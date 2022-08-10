# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.registry import DROPOUT_LAYERS
from mmcv.runner.base_module import BaseModule

from mmrazor.models.architectures.dynamic_op import RelativePosition2D
from mmrazor.models.mutables.base_mutable import BaseMutable
from .base import ChannelDynamicOP


class MultiheadAttention(BaseModule):
    """Multi-head Attention Module with iRPE.
    This module implements multi-head attention that supports different input
    dims and embed dims. And it also supports a shortcut from ``value``, which
    is useful if input dims is not the same with embed dims.
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        input_dims (int, optional): The input dimension, and if None,
            use ``embed_dims``. Defaults to None.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        dropout_layer (dict): The dropout config before adding the shortcut.
            Defaults to ``dict(type='Dropout', drop_prob=0.)``.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        v_shortcut (bool): Add a shortcut from value to output. It's usually
            used if ``input_dims`` is different from ``embed_dims``.
            Defaults to False.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 input_dims=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 relative_position=True,
                 max_relative_position=14,
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 v_shortcut=False,
                 init_cfg=None):
        super(MultiheadAttention, self).__init__(init_cfg=init_cfg)

        self.input_dims = input_dims or embed_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.v_shortcut = v_shortcut
        self.relative_position = relative_position
        self.max_relative_position = max_relative_position

        self.head_dims = embed_dims // num_heads  # unit
        self.scale = qk_scale or self.head_dims**-0.5

        self.w_qs = nn.Linear(
            self.input_dims, num_heads * self.head_dims, bias=qkv_bias)
        self.w_ks = nn.Linear(
            self.input_dims, num_heads * self.head_dims, bias=qkv_bias)

        self.w_vs = nn.Linear(
            self.input_dims, num_heads * self.head_dims, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.out_drop = DROPOUT_LAYERS.build(dropout_layer)

        # image relative position encoding
        if self.relative_position:
            self.rel_pos_embed_k = RelativePosition2D(
                self.num_heads, self.max_relative_position)
            self.rel_pos_embed_v = RelativePosition2D(
                self.num_heads, self.max_relative_position)

    def forward(self, x):
        B, N, _ = x.shape

        q = self.w_qs(x).view(B, N, self.num_heads, self.head_dims)
        k = self.w_ks(x).view(B, N, self.num_heads, self.head_dims)
        v = self.w_vs(x).view(B, N, self.num_heads, self.head_dims)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_position:
            r_p_k = self.rel_pos_embed_k(N, N)
            attn = attn + (q.permute(2, 0, 1, 3).reshape(N, self.num_heads * B, -1)
                           @ r_p_k.transpose(2, 1)) \
                .transpose(1, 0).reshape(B, self.num_heads, N, N) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.embed_dims)

        if self.relative_position:
            r_p_v = self.rel_pos_embed_v(N, N)
            t_attn = attn.permute(2, 0, 1, 3).reshape(N, B * self.num_heads,
                                                      -1)
            x = x + (t_attn @ r_p_v).transpose(1, 0).reshape(
                B, self.num_heads, N, -1).transpose(2, 1).reshape(B, N, -1)

        x = self.proj(x)
        x = self.out_drop(self.proj_drop(x))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x


class DynamicMultiheadAttention(MultiheadAttention, ChannelDynamicOP):
    """_summary_"""

    accpeted_mutables = {
        'mutable_head_dims',
        'mutable_embed_dims',
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.mutable_head_dims: Optional[BaseMutable] = None
        self.mutable_embed_dims: Optional[BaseMutable] = None

    def _get_q_out_mask(self):
        # TODO sample, min is just for test.
        active_num_heads = min(self.mutable_num_heads)
        active_heads_mask = torch.cat(
            [self.mutable_head_dims.mask] * active_num_heads, dim=0)

        inactive_num_heads = max(self.mutable_num_heads) - active_num_heads
        inactive_mask = torch.zeros_like(self.mutable_head_dims.mask).bool()
        inactive_heads_mask = torch.cat(
            [inactive_mask] * inactive_num_heads, dim=0)

        q_out_mask = torch.cat([active_heads_mask, inactive_heads_mask], dim=0)

        return q_out_mask

    def _get_qkv_weight_bias(self):
        q_out_mask = self._get_q_out_mask()
        out_mask = torch.cat([q_out_mask] * 3, dim=0)
        in_mask = self.mutable_embed_dims.mask

        weight = self.qkv.weight[out_mask][:, in_mask]
        bias = self.qkv.bias[out_mask] if self.qkv.bias is not None else None
        return weight, bias

    def _get_proj_weight_bias(self):
        out_mask = self.mutable_embed_dims.mask
        in_mask = self._get_q_out_mask()

        weight = self.proj.weight[out_mask][:, in_mask]
        bias = self.proj.bias[out_mask] if self.qkv.bias is not None else None
        return weight, bias

    def forward(self, x):
        B, N, _ = x.shape

        qkv_weight, qkv_bias = self._get_qkv_weight_bias()
        qkv = F.linear(x, qkv_weight, qkv_bias)

        # TODO mutable value, min is just for test
        current_num_heads = min(self.mutable_num_heads)

        current_head_dims = self.mutable_head_dims.mask.sum()
        current_embed_dims = self.mutable_embed_dims.mask.sum()
        qkv = qkv.reshape(B, N, 3, current_num_heads,
                          current_head_dims).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(
            B, N, current_num_heads * current_head_dims)
        proj_weight, proj_bias = self._get_proj_weight_bias()
        x = F.linear(x, proj_weight, proj_bias)

        x = self.out_drop(self.proj_drop(x))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x

    def forward(self, x):
        num_heads = self.get_value(self.num_heads)
        # super_embed_dim = self.get_value(self.super_embed_dim)
        self.scale = 0.125
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, num_heads,
                                  -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_position:
            r_p_k = self.rel_pos_embed_k(N, N)
            attn = attn + (q.permute(2, 0, 1, 3).reshape(N, num_heads * B, -1)
                           @ r_p_k.transpose(2, 1)) \
                .transpose(1, 0).reshape(B, num_heads, N, N) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        if self.relative_position:
            r_p_v = self.rel_pos_embed_v(N, N)
            attn_1 = attn.permute(2, 0, 1, 3).reshape(N, B * num_heads, -1)
            x = x + (attn_1 @ r_p_v).transpose(1, 0).reshape(
                B, num_heads, N, -1).transpose(2, 1).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# torch 的 MHA 中逻辑过于复杂，这里是基于 mmcls 的 MHA 开发的
class DynamicMHA(MultiheadAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # auto former 的官方实现中 head dims 是固定的 64
        # 这里的实现是 head dims 可搜索
        self.mutable_head_dims = OrderChannelMutable(
            self.head_dims,
            candidate_choices=[int(0.5 * self.head_dims), self.head_dims])
        self.mutable_embed_dims = OrderChannelMutable(
            self.embed_dims, candidate_choices=[320, 384, 448])
        # 还没设计实现 MutableValue，先拿一个 list 简单测试一下
        self.mutable_num_heads = [5, 6, 7]
        # TODO del, just for test
        self.mutable_head_dims.current_choice = int(0.5 * self.head_dims)

    def _get_q_out_mask(self):
        # TODO sample, min is just for test.
        active_num_heads = min(self.mutable_num_heads)
        active_heads_mask = torch.cat(
            [self.mutable_head_dims.mask] * active_num_heads, dim=0)
        inactive_num_heads = max(self.mutable_num_heads) - active_num_heads
        inactive_mask = torch.zeros_like(self.mutable_head_dims.mask).bool()
        inactive_heads_mask = torch.cat(
            [inactive_mask] * inactive_num_heads, dim=0)

        q_out_mask = torch.cat([active_heads_mask, inactive_heads_mask], dim=0)

        return q_out_mask

    def _get_qkv_weight_bias(self):
        q_out_mask = self._get_q_out_mask()
        out_mask = torch.cat([q_out_mask] * 3, dim=0)
        in_mask = self.mutable_embed_dims.mask

        weight = self.qkv.weight[out_mask][:, in_mask]
        bias = self.qkv.bias[out_mask] if self.qkv.bias is not None else None
        return weight, bias

    def _get_proj_weight_bias(self):
        out_mask = self.mutable_embed_dims.mask
        in_mask = self._get_q_out_mask()

        weight = self.proj.weight[out_mask][:, in_mask]
        bias = self.proj.bias[out_mask] if self.qkv.bias is not None else None
        return weight, bias

    def forward(self, x):
        B, N, _ = x.shape

        qkv_weight, qkv_bias = self._get_qkv_weight_bias()
        qkv = F.linear(x, qkv_weight, qkv_bias)

        # TODO mutable value, min is just for test
        current_num_heads = min(self.mutable_num_heads)

        current_head_dims = self.mutable_head_dims.mask.sum()
        current_embed_dims = self.mutable_embed_dims.mask.sum()
        qkv = qkv.reshape(B, N, 3, current_num_heads,
                          current_head_dims).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(
            B, N, current_num_heads * current_head_dims)
        proj_weight, proj_bias = self._get_proj_weight_bias()
        x = F.linear(x, proj_weight, proj_bias)

        x = self.out_drop(self.proj_drop(x))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x
