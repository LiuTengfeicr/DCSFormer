"""
Point Transformer - V3 Mode1

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from functools import partial
from addict import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
import torch_scatter
from timm.layers import DropPath

from pointcept.models.point_transformer_v3.point_transformer_v3m2_sonata import SerializedAttention

try:
    import flash_attn
except ImportError:
    flash_attn = None

from pointcept.models.point_prompt_training import PDNorm
from pointcept.models.builder import MODELS
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential
from pointcept.models.utils.knn_cuda import knn as knn_utils
from typing import List
from torch.nn import LayerNorm

class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
                coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
                + self.pos_bnd  # relative position to positive index
                + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out

class DynamicSerializedAttention(SerializedAttention):

    def __init__(
            self,
            channels: int,
            num_heads: int,
            patch_size: int,
            *,
            num_experts: int = 6,
            k_active: int = 2,
            balance_loss_weight: float = 0.01,
            router_hidden_ratio: float = 0.25,
            max_patch_size_non_flash: int = 128,
            **kwargs,
    ) -> None:
        super().__init__(channels=channels, num_heads=num_heads, patch_size=patch_size, **kwargs)

        hid = int(channels * router_hidden_ratio)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # (B,C,N)->(B,C,1)
            nn.Flatten(),
            nn.Linear(channels, hid),
            nn.GELU(),
            nn.Linear(hid, num_experts),
        )
        self.num_experts = num_experts
        self.k_active = k_active
        self.balance_w = balance_loss_weight
        self.max_patch_size_non_flash = max_patch_size_non_flash

        self.router_mlp = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, self.num_experts)
        )
        # --- Expert mapping ---
        self._expert_impls = [
            self._expert_softmax,
            self._expert_flash,
            self._expert_topk_sparse,
        ]
        self._expert_impls = self._expert_impls[:num_experts]
        assert len(self._expert_impls) == num_experts, "error"

    @staticmethod
    def _pad_batch_feats(feat: torch.Tensor, batch: torch.Tensor):
        """Pad variable‑length point clouds in the same batch to max length."""
        device = feat.device
        B = batch.max().item() + 1
        counts = torch.bincount(batch, minlength=B)
        K = counts.max().item()
        C = feat.shape[1]

        feat_padded = torch.zeros((B, K, C), device=device, dtype=feat.dtype)
        mask = torch.zeros((B, K), device=device, dtype=torch.bool)
        for b in range(B):
            idx = (batch == b).nonzero(as_tuple=False).squeeze(-1)
            n = idx.numel()
            if n:
                feat_padded[b, :n] = feat[idx]
                mask[b, :n] = 1
        return feat_padded, mask, K

    @staticmethod
    def safe_softmax(x, dim=-1):
        is_all_inf = torch.isinf(x).all(dim=dim, keepdim=True)
        x = torch.where(is_all_inf, torch.zeros_like(x), x)
        return torch.softmax(x, dim=dim)

    def _route(self, x):
        x = x.squeeze(-1)
        logits = self.router_mlp(x).squeeze(-1)

        logits = torch.clamp(logits, -30, 30)
        alpha = torch.softmax(logits, dim=-1)

        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)

        return alpha

    def _expert_softmax(self, q, k, v, rel_pos=None, mask=None):
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if rel_pos is not None:
            attn = attn + rel_pos
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1), float('-inf'))
        attn = self.safe_softmax(attn, dim=-1)
        return attn @ v

    def _expert_flash(self, qkv_packed, cu_seqlens, max_seqlen):

        packed = qkv_packed  # shape: [B * K, 3, H * Dh]

        cu_seqlens_b = torch.tensor([0, max_seqlen], dtype=torch.int32, device=packed.device)

        out = flash_attn.flash_attn_varlen_qkvpacked_func(
            packed.half(), cu_seqlens_b, max_seqlen=max_seqlen,
            dropout_p=self.attn_drop if self.training else 0.0, softmax_scale=self.scale
        )

        return out

    def _expert_cosine(self, q, k, v, rel_pos=None, mask=None, eps=1e-6):
        # 防 0 范数
        q = F.normalize(q.float(), dim=-1, eps=eps).type_as(q)
        k = F.normalize(k.float(), dim=-1, eps=eps).type_as(k)

        attn = F.relu((q @ k.transpose(-2, -1)) * self.scale)

        if rel_pos is not None:
            attn = attn + rel_pos
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1), float('-inf'))

        attn = self.safe_softmax(attn, dim=-1)
        return attn @ v


    def _expert_topk_sparse(self, q, k, v, rel_pos=None,
                            mask=None, topk=8):
        attn = (q * self.scale) @ k.transpose(-2, -1)

        if rel_pos is not None:
            attn = attn + rel_pos
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1), float('-inf'))

        topk_val, _ = torch.topk(attn, topk, dim=-1)
        thresh = topk_val[:, :, -1:].detach()
        sparse_mask = attn >= thresh
        attn = attn.masked_fill(~sparse_mask, float('-inf'))

        attn = self.safe_softmax(attn, dim=-1)
        return attn @ v

    @staticmethod
    def _expert_identity(q, k, v, **kwargs):
        return v

    # ---------------------------------------------------------------------
    # forward
    # ---------------------------------------------------------------------
    def forward(self, point):

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)
        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        feat_ord = point.feat[order]
        batch_ord = point.batch[order]
        feat_pad, mask, K = self._pad_batch_feats(feat_ord, batch_ord)

        B, C = feat_pad.shape[0], feat_pad.shape[2]
        H, Dh = self.num_heads, C // self.num_heads

        qkv = self.qkv(feat_pad).view(B, K, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = [x.reshape(B * H, K, Dh) for x in qkv]
        attn_mask = (~mask).unsqueeze(1).expand(B, H, K).reshape(B * H, K)

        rel_pos = None
        if self.enable_rpe:
            rp = self.get_rel_pos(point, order)
            if rp is not None:
                rel_pos = self.rpe(rp)

        alpha = self._route(feat_pad.mean(1).unsqueeze(-1))

        expert_outs = [[None] * B for _ in range(self.num_experts)]
        K_nf = min(K, self.max_patch_size_non_flash)

        for b in range(B):
            bh = slice(b * H, (b + 1) * H)
            active_idx = (alpha[b] > 0).nonzero().squeeze(1)
            for idx in active_idx:
                idx = idx.item()
                fn = self._expert_impls[idx]

                if idx == 1 and flash_attn is not None:
                    qkv_b = qkv[:, b].unsqueeze(1)  # (3, 1, H, K, Dh)
                    _, _, H_b, L_b, Dh_b = qkv_b.shape
                    total_tokens = L_b

                    qkv_packed = (
                        qkv_b.permute(3, 0, 2, 4, 1)  # (K, 3, H, Dh, 1)
                        .reshape(total_tokens, 3, H_b, Dh_b)
                    )

                    cu = torch.tensor([0, total_tokens],
                                      dtype=torch.int32,
                                      device=qkv_packed.device)

                    out_tok = flash_attn.flash_attn_varlen_qkvpacked_func(
                        qkv_packed.half(), cu,
                        max_seqlen=L_b,
                        dropout_p=self.attn_drop if self.training else 0.0,
                        softmax_scale=self.scale,
                    )

                    out = (out_tok
                           .reshape(L_b, H_b, Dh_b)  # (K, H, Dh)
                           .permute(1, 0, 2))  # (H, K, Dh)

                    expert_outs[idx][b] = out
                else:  # 其余 expert
                    out = fn(
                        q[bh, :K_nf], k[bh, :K_nf], v[bh, :K_nf],
                        rel_pos=None if rel_pos is None else rel_pos[bh],
                        mask=attn_mask[bh, :K_nf]
                    )

                    if K_nf < K:
                        pad_len = K - K_nf
                        pad_zeros = torch.zeros(out.size(0), pad_len, Dh,
                                                dtype=out.dtype, device=out.device)
                        out = torch.cat([out, pad_zeros], dim=1)

                expert_outs[idx][b] = out

        fused_chunks = []
        for b in range(B):
            bh = slice(b * H, (b + 1) * H)
            fused_b = None
            for idx in (alpha[b] > 0).nonzero(as_tuple=False).squeeze(1):
                idx = idx.item()
                out_slice = expert_outs[idx][b]
                if out_slice is None:
                    continue
                if fused_b is None:
                    fused_b = out_slice * alpha[b, idx]
                else:
                    fused_b = fused_b + out_slice * alpha[b, idx]

            if fused_b is None:
                print(f"[Warning] Batch {b} had no expert output; using identity fallback.")
                fused_b = v[bh]

            fused_chunks.append(fused_b)

        fused = torch.cat(fused_chunks, 0)  # (B*H, K, Dh)
        fused = fused.transpose(1, 2).reshape(-1, C)
        point.feat = self.proj_drop(self.proj(fused)[inverse])

        return point

class SerializedAttention(PointModule):
    def __init__(
            self,
            channels,
            num_heads,
            patch_size,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            order_index=0,
            enable_rpe=False,
            enable_flash=True,
            upcast_attention=True,
            upcast_softmax=True,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash
        if enable_flash:
            assert (
                    enable_rpe is False
            ), "Set enable_rpe to False when enable Flash Attention"
            assert (
                    upcast_attention is False
            ), "Set upcast_attention to False when enable Flash Attention"
            assert (
                    upcast_softmax is False
            ), "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
                pad_key not in point.keys()
                or unpad_key not in point.keys()
                or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                    torch.div(
                        bincount + self.patch_size - 1,
                        self.patch_size,
                        rounding_mode="trunc",
                    )
                    * self.patch_size
            )
            # only pad point when num of points larger than patch_size
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i]: _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                    _offset_pad[i + 1]
                    - self.patch_size
                    + (bincount[i] % self.patch_size): _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size): _offset_pad[i + 1]
                                                           - self.patch_size
                        ]
                pad[_offset_pad[i]: _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # padding and reshape feat and batch for serialized point patch
        qkv = self.qkv(point.feat)[order]

        if not self.enable_flash:
            # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            # attn
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        else:
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half().reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)
        feat = feat[inverse]

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class DualAttentionMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, drop=0.0):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels

        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.norm2 = nn.LayerNorm(out_channels)

        self.fc_cca1 = nn.Linear(hidden_channels, hidden_channels // 4)
        self.fc_cca2 = nn.Linear(hidden_channels // 4, hidden_channels)
        self.norm_cca = nn.LayerNorm(hidden_channels)

        self.conv_sca1 = nn.Conv1d(1, 4, kernel_size=1)
        self.conv_sca2 = nn.Conv1d(4, 1, kernel_size=1)
        self.norm_sca = nn.LayerNorm(hidden_channels)

        self.alpha = nn.Parameter(torch.tensor(0.0))

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        residual = x

        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.drop(x)

        cca = x.mean(dim=0, keepdim=True)
        cca = self.fc_cca1(cca)
        cca = F.relu(cca)
        cca = self.fc_cca2(cca)
        cca = torch.sigmoid(cca)
        x_cca = x * cca
        x_cca = self.norm_cca(x_cca)

        sca = x.mean(dim=1, keepdim=True)
        sca = sca.unsqueeze(0).transpose(1, 2)  # [1, N, 1]
        sca = self.conv_sca1(sca)
        sca = F.relu(sca)
        sca = self.conv_sca2(sca)
        sca = torch.sigmoid(sca).squeeze(0).squeeze(0)
        x_sca = x * sca.unsqueeze(1)
        x_sca = self.norm_sca(x_sca)

        gate = torch.sigmoid(self.alpha)  # scalar
        x = x + gate * x_cca + (1 - gate) * x_sca

        x = self.fc2(x)
        x = self.norm2(x)
        x = self.drop(x)

        return x + residual

class MLP(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels=None,
            out_channels=None,
            act_layer=nn.GELU,
            drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(PointModule):
    def __init__(
            self,
            channels,
            num_heads,
            patch_size=48,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
            pre_norm=True,
            order_index=0,
            cpe_indice_key=None,
            enable_rpe=False,
            enable_flash=True,
            upcast_attention=True,
            upcast_softmax=True,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = DynamicSerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_experts=3,
            k_active=1,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            DualAttentionMLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class GlobalContextAttention(PointModule):
    def __init__(self, channels):
        super().__init__()
        self.query_proj = nn.Linear(channels, channels)
        self.key_proj = nn.Linear(channels, channels)
        self.value_proj = nn.Linear(channels, channels)
        self.softmax = nn.Softmax(dim=-1)
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, point: Point):
        batch_size = point.offset.size(0)
        start_idx = torch.cat([point.offset.new_zeros(1), point.offset[:-1]])
        semantic_tokens = []
        for b in range(batch_size):
            feat_b = point.feat[start_idx[b]:point.offset[b]]
            semantic_tokens.append(feat_b.mean(dim=0, keepdim=True))  # (1, C)
        semantic_token = torch.cat(semantic_tokens, dim=0)  # (B, C)

        q = self.query_proj(point.feat)  # (N, C)
        k = self.key_proj(semantic_token)  # (B, C)
        v = self.value_proj(semantic_token)  # (B, C)

        batch_indices = torch.searchsorted(point.offset, torch.arange(point.feat.size(0), device=point.feat.device),
                                           right=True)
        k_expand = k[batch_indices - 1]  # (N, C)
        v_expand = v[batch_indices - 1]  # (N, C)

        attn_scores = (q * k_expand).sum(dim=-1, keepdim=True)  # (N, 1)
        attn_probs = self.softmax(attn_scores)

        out = attn_probs * v_expand  # (N, C)
        out = self.out_proj(out)

        point.feat = out
        return point


class GeometryEnhance(PointModule):

    def __init__(self, channels, k=16):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )

    def forward(self, point: Point):
        # CUDA KNN
        knn_idx = knn_utils(self.k, point.coord, point.offset)  # (N, k)

        neigh_feat = point.feat[knn_idx].mean(1)  # (N, C)
        high_freq = point.feat - neigh_feat

        point.feat = point.feat + self.mlp(high_freq)
        return point


# ---------------- Geo‑Semantic Joint Enhancement ----------------
class GeoSemEnhance(PointModule):
    def __init__(self, channels, k=16):
        super().__init__()
        self.geo_enhance = GeometryEnhance(channels, k=k)
        self.sem_attn = GlobalContextAttention(channels)
        self.gate = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.Sigmoid()
        )
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, point: Point):
        geo_input = point.clone(safe=True)
        geo = self.geo_enhance(geo_input)

        sem = self.sem_attn(Point(
            coord=point.coord,
            feat=point.feat.clone(),
            offset=point.offset
        ))

        fuse = torch.cat([geo.feat, sem.feat], dim=-1)
        alpha = self.gate(fuse)
        point.feat = self.out_proj(alpha * geo.feat + (1 - alpha) * sem.feat)
        return point


# ---------------- Semantic Enhancement ----------------
class SemanticEnhance(GlobalContextAttention):
    def __init__(self, channels):
        super().__init__(channels)


class SerializedPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
        # TODO: add support to grid pool (any stride)
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(
            point.keys()
        ), "Run point.serialization() point cloud before SerializedPooling"

        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        # generate down code, order, inverse
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # collect information
        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()
        return point


def interpolate_feature(source_point: "Point", target_point: "Point", k: int = 3):
    knn_idx = knn_utils(k, source_point.coord, source_point.offset,
                        new_xyz=target_point.coord, new_offset=target_point.offset)

    neighbor_feat = source_point.feat[knn_idx]  # (N_target, k, C)
    neighbor_coord = source_point.coord[knn_idx]  # (N_target, k, 3)
    center = target_point.coord.unsqueeze(1)  # (N_target, 1, 3)

    dist = ((center - neighbor_coord) ** 2).sum(dim=-1) + 1e-8  # (N_target, k)
    weight = 1.0 / dist
    weight = weight / weight.sum(dim=1, keepdim=True)  # (N_target, k)

    interpolated_feat = (weight.unsqueeze(-1) * neighbor_feat).sum(dim=1)  # (N_target, C)
    return interpolated_feat


class FrequencyEnhance(nn.Module):
    def __init__(self, channels, hidden_ratio=1.0):
        super().__init__()
        hidden_dim = int(channels * hidden_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, channels)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_fft = torch.fft.fft(x, dim=0)  # (N, C), complex64

        N = x.shape[0]
        low = x_fft[:N//4].real
        mid = x_fft[N//4:N*3//4].real
        high = x_fft[N*3//4:].real

        low_enh = self.mlp(low)
        mid_enh = self.mlp(mid)
        high_enh = self.mlp(high)

        x_enh_real = torch.cat([low_enh, mid_enh, high_enh], dim=0)

        x_fft_enh = torch.complex(x_enh_real, x_fft.real[:x_enh_real.shape[0]] * 0)

        x_rec = torch.fft.ifft(x_fft_enh, dim=0).real  # (N, C)

        weight = self.sigmoid(x_rec)

        return weight


class CrossLayerFusion(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.projs = nn.ModuleList([nn.Linear(in_c, out_channels) for in_c in in_channels_list])
        self.fuse = nn.Sequential(
            nn.Linear(out_channels * len(in_channels_list), out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels)
        )

        self.freq_enhance = FrequencyEnhance(out_channels)
    def forward(self, points: List[Point], target_point: Point):
        fused_feats = []
        for i, (proj, p) in enumerate(zip(self.projs, points)):
            aligned_feat = interpolate_feature(p, target_point)
            aligned_feat = proj(aligned_feat)
            fused_feats.append(aligned_feat)
        feat = torch.cat(fused_feats, dim=-1)  # (N_target, out_c * n)
        fused_feat = self.fuse(feat)  # (N_target, out_c)

        weight = self.freq_enhance(fused_feat)
        fused_feat = fused_feat + fused_feat * weight
        return fused_feat


class SerializedUnpooling(PointModule):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            norm_layer=None,
            act_layer=None,
            traceable=False,  # record parent and cluster
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))

        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.skip_channel = skip_channels
        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_inverse" in point.keys()
        assert "pooling_parent" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent = self.proj_skip(parent)
        point = self.proj(point)
        # 融合
        parent.feat = parent.feat + point.feat[inverse]
        if self.traceable:
            parent["unpooling_parent"] = point
        return parent


class Embedding(PointModule):
    def __init__(
            self,
            in_channels,
            embed_channels,
            norm_layer=None,
            act_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        # TODO: check remove spconv
        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point


@MODELS.register_module("PT-v3m1")
class PointTransformerV3(PointModule):
    def __init__(
            self,
            in_channels=6,
            order=("z", "z-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(32, 64, 128, 256, 512),
            enc_num_head=(2, 4, 8, 16, 32),
            enc_patch_size=(48, 48, 48, 48, 48),
            dec_depths=(2, 2, 2, 2),
            dec_channels=(64, 64, 128, 256),
            dec_num_head=(4, 4, 8, 16),
            dec_patch_size=(48, 48, 48, 48),
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=0.3,
            pre_norm=True,
            shuffle_orders=True,
            enable_rpe=False,
            enable_flash=True,
            upcast_attention=False,
            upcast_softmax=False,
            cls_mode=False,
            pdnorm_bn=False,
            pdnorm_ln=False,
            pdnorm_decouple=True,
            pdnorm_adaptive=False,
            pdnorm_affine=True,
            pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1

        # norm layers
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    LayerNorm,
                    elementwise_affine=True
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(LayerNorm, elementwise_affine=True)

        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                             sum(enc_depths[:s]): sum(enc_depths[: s + 1])
                             ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )

            if s in (0, 1):
                enc.add(GeometryEnhance(enc_channels[s]), name="geo_enh")
            elif s == 2:
                enc.add(GeoSemEnhance(enc_channels[s]), name="geosem_enh")
            elif s in (3, 4):
                enc.add(SemanticEnhance(enc_channels[s]), name="sem_enh")

        self.fusion0 = CrossLayerFusion(
            in_channels_list=[enc_channels[0], enc_channels[2], enc_channels[4]],  # [32, 128, 512]
            out_channels=512
        )
        self.fusion1 = CrossLayerFusion(
            in_channels_list=[enc_channels[1], enc_channels[2], enc_channels[3]],  # [64, 128, 256]
            out_channels=256
        )
        self.fusion2 = CrossLayerFusion(
            in_channels_list=[enc_channels[0], enc_channels[2], enc_channels[4]],  # [32, 128, 512]
            out_channels=128
        )
        self.fusion3 = CrossLayerFusion(
            in_channels_list=[enc_channels[1], enc_channels[2], enc_channels[3]],  # [64, 128, 256]
            out_channels=64
        )

        # decoder
        if not self.cls_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            self.dec_channels1 = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                                 sum(dec_depths[:s]): sum(dec_depths[: s + 1])
                                 ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

    def forward(self, data_dict):

        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        enc_points = []

        point_temp = point.clone(safe=True)
        for layer in self.enc:
            point_temp = layer(point_temp)

            if torch.isnan(point_temp.feat).any():
                print(f"[ERROR] NaN detected at {layer}")
                raise ValueError("NaN in point_temp.feat")

            if point_temp.feat.shape[0] != point_temp.coord.shape[0]:
                print(f"[ERROR] Shape mismatch at {layer}")
                raise ValueError("Shape mismatch")

            if hasattr(point_temp, "pooling_parent") and point_temp.pooling_parent is point_temp:
                print(f"[ERROR] Recursive parent detected at {layer}")
                raise ValueError("Recursive parent reference")

            enc_points.append(point_temp.clone(safe=True))

        point = self.enc(point)

        if not self.cls_mode:
            dec_point_list = []
            current_point = point
            dec_point_list.append(current_point)

            fusion_inputs = [
                [enc_points[0], enc_points[2], enc_points[4]],  # for s = 0
                [enc_points[1], enc_points[2], enc_points[3]],  # for s = 1
                [enc_points[0], enc_points[2], enc_points[4]],  # for s = 2
                [enc_points[1], enc_points[2], enc_points[3]],  # for s = 3
            ]
            fusion_modules = [self.fusion0, self.fusion1, self.fusion2, self.fusion3]

            for s in reversed(range(self.num_stages - 1)):
                skip_feat = fusion_modules[s](fusion_inputs[s], current_point)

                current_point = current_point.clone_with(
                    feat=skip_feat,
                    pooling_inverse=current_point.pooling_inverse,
                    pooling_parent=current_point.pooling_parent
                )
                dec_point_list.append(current_point)

            point = dec_point_list[-1]

            point = self.dec(point)

        return point

