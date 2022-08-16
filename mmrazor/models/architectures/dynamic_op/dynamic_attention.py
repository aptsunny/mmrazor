# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.registry import DROPOUT_LAYERS
from mmcv.runner.base_module import BaseModule
from torch import Tensor

from mmrazor.models.architectures.dynamic_op import (DynamicRelativePosition2D,
                                                     RelativePosition2D)
from mmrazor.models.mutables.base_mutable import BaseMutable
from .base import DynamicOP


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
            attn = attn + (q.permute(2, 0, 1, 3).reshape(N, self.num_heads * B, -1)  # noqa: E501
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


class DynamicMultiheadAttention(MultiheadAttention, DynamicOP):
    """Dynamic Multihead Attention with iRPE"""

    accpeted_mutables = {
        # 'mutable_head_dims', = embed_dims / num_heads
        'mutable_num_heads',
        'mutable_embed_dims',
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.mutable_num_heads: Optional[BaseMutable] = None
        self.mutable_embed_dims: Optional[BaseMutable] = None
        self.mutable_head_dims: Optional[BaseMutable] = None
        # DerivedMutable

        # dynamic image relative position encoding
        if self.relative_position:
            self.rel_pos_embed_k = DynamicRelativePosition2D(
                self.num_heads, self.max_relative_position)
            self.rel_pos_embed_v = DynamicRelativePosition2D(
                self.num_heads, self.max_relative_position)

    def _get_dynamic_proj_params(
            self, w: nn.Linear) -> Tuple[Tensor, Optional[Tensor]]:
        # TODO support mask later
        if self.mutable_embed_dims is None:
            return w.weight, w.bias

        if self.mutable_embed_dims is not None:
            in_features = self.mutable_embed_dims.current_choice.to(
                self.weight.device)
        else:
            in_features = self.embed_dims

        out_features = in_features

        weight = self.weight[:out_features][:, in_features]
        bias = self.bias[:out_features] if self.bias is not None else None

        return weight, bias

    def _get_dynamic_qkv_params(
            self, w: nn.Linear) -> Tuple[Tensor, Optional[Tensor]]:
        # TODO support mask later
        if self.mutable_num_heads is None and self.mutable_embed_dims is None:
            return w.weight, w.bias

        if self.mutable_embed_dims is not None:
            in_features = self.mutable_embed_dims.current_choice.to(
                self.weight.device)
        else:
            in_features = self.embed_dims

        if self.mutable_num_heads is not None:
            out_features = self.mutable_num_heads * self.mutable_head_dims.to(
                self.weight.device)
        else:
            out_features = self.num_heads * self.head_dims

        weight = self.weight[:out_features][:, in_features]
        bias = self.bias[:out_features] if self.bias is not None else None

        return weight, bias

    def forward(self, x: Tensor) -> Tensor:
        B, N = x.shape(0), x.shape1(1)
        num_heads = self.mutable_num_heads.current_choice.to(
            self.weight.device)

        q_w, q_b = self._get_dynamic_params(self.w_qs)
        k_w, k_b = self._get_dynamic_params(self.k_qs)
        v_w, v_b = self._get_dynamic_params(self.v_qs)

        q = F.linear(x, q_w, q_b).view(B, N, num_heads, self.unit)
        k = F.linear(x, k_w, k_b).view(B, N, num_heads, self.unit)
        v = F.linear(x, v_w, v_b).view(B, N, num_heads, self.unit)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_position:
            r_p_k = self.rel_pos_embed_k(N, N)
            attn = attn + (q.permute(2, 0, 1, 3).reshape(N, num_heads * B, -1)  # noqa: E501
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
        x = self.proj_drop(x),
        return x

    def to_static_op(self) -> nn.Module:
        self.check_if_mutables_fixed()

        embed_dims = self.mutable_embed_dims.current_choice
        num_heads = self.mutable_num_heads.current_choice

        q_w, q_b = self._get_dynamic_qkv_params(self.w_qs)
        k_w, k_b = self._get_dynamic_qkv_params(self.k_qs)
        v_w, v_b = self._get_dynamic_qkv_params(self.v_qs)

        proj_w, proj_b = self._get_dynamic_proj_params(self.proj)

        static_mha = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            input_dims=None,
            attn_drop=self.attn_drop,
            relative_position=self.relative_position,
            max_relative_position=self.max_relative_position)

        static_mha.w_qs.weight = nn.Parameter(q_w.clone())
        static_mha.w_qs.bias = nn.Parameter(q_b.clone())

        static_mha.w_ks.weight = nn.Parameter(k_w.clone())
        static_mha.w_ks.bias = nn.Parameter(k_b.clone())

        static_mha.w_vs.weight = nn.Parameter(v_w.clone())
        static_mha.w_vs.bias = nn.Parameter(v_b.clone())

        static_mha.proj.weight = nn.Parameter(proj_w.clone())
        static_mha.proj.bias = nn.Parameter(proj_b.clone())

        if self.relative_position:
            static_mha.rel_pos_embed_k = self.rel_pos_embed_k.to_static_op()
            static_mha.rel_pos_embed_v = self.rel_pos_embed_v.to_static_op()

        return static_mha
