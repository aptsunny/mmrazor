# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmcv.cnn.bricks.registry import CONV_LAYERS
from mmcv.utils import to_2tuple
from torch import Tensor

from mmrazor.models.mutables.base_mutable import BaseMutable
from .base import ChannelDynamicOP


class Attention(nn.Module):
    def __init__(self, super_embed_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., , relative_position=False,
                 max_relative_position=14, scale=False, change_qkv=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = super_embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.super_embed_dim = super_embed_dim

        self.fc_scale = scale
        self.change_qkv = change_qkv
        if change_qkv:
            self.qkv = qkv_super(super_embed_dim, 3 *
                                 super_embed_dim, bias=qkv_bias)
        else:
            self.qkv = LinearSuper(
                super_embed_dim, 3 * super_embed_dim, bias=qkv_bias)

        self.relative_position = relative_position
        if self.relative_position:
            self.rel_pos_embed_k = RelativePosition2D_super(
                super_embed_dim // num_heads, max_relative_position)
            self.rel_pos_embed_v = RelativePosition2D_super(
                super_embed_dim // num_heads, max_relative_position)
        self.max_relative_position = max_relative_position
        self.sample_qk_embed_dim = None
        self.sample_v_embed_dim = None
        self.sample_num_heads = None
        self.sample_scale = None
        self.sample_in_embed_dim = None

        self.proj = LinearSuper(super_embed_dim, super_embed_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def set_sample_config(self, sample_q_embed_dim=None, sample_num_heads=None, sample_in_embed_dim=None):

        self.sample_in_embed_dim = sample_in_embed_dim
        self.sample_num_heads = sample_num_heads
        if not self.change_qkv:
            self.sample_qk_embed_dim = self.super_embed_dim
            self.sample_scale = (sample_in_embed_dim //
                                 self.sample_num_heads) ** -0.5

        else:
            self.sample_qk_embed_dim = sample_q_embed_dim
            self.sample_scale = (self.sample_qk_embed_dim //
                                 self.sample_num_heads) ** -0.5

        self.qkv.set_sample_config(
            sample_in_dim=sample_in_embed_dim, sample_out_dim=3*self.sample_qk_embed_dim)
        self.proj.set_sample_config(
            sample_in_dim=self.sample_qk_embed_dim, sample_out_dim=sample_in_embed_dim)
        if self.relative_position:
            self.rel_pos_embed_k.set_sample_config(
                self.sample_qk_embed_dim // sample_num_heads)
            self.rel_pos_embed_v.set_sample_config(
                self.sample_qk_embed_dim // sample_num_heads)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B, N, 3, self.sample_num_heads, -1).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.sample_scale
        if self.relative_position:
            r_p_k = self.rel_pos_embed_k(N, N)
            attn = attn + (q.permute(2, 0, 1, 3).reshape(N, self.sample_num_heads * B, -1) @ r_p_k.transpose(2, 1)) \
                .transpose(1, 0).reshape(B, self.sample_num_heads, N, N) * self.sample_scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        if self.relative_position:
            r_p_v = self.rel_pos_embed_v(N, N)
            attn_1 = attn.permute(2, 0, 1, 3).reshape(
                N, B * self.sample_num_heads, -1)
            # The size of attention is (B, num_heads, N, N), reshape it to (N, B*num_heads, N) and do batch matmul with
            # the relative position embedding of V (N, N, head_dim) get shape like (N, B*num_heads, head_dim). We reshape it to the
            # same size as x (B, num_heads, N, hidden_dim)
            x = x + (attn_1 @ r_p_v).transpose(1, 0).reshape(B,
                                                             self.sample_num_heads, N, -1).transpose(2, 1).reshape(B, N, -1)

        if self.fc_scale:
            x = x * (self.super_embed_dim / self.sample_qk_embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DynamicAttention(ChannelDynamicOP):
    """_summary_

    Args:
        embed_dim (int): _description_
        num_heads (int): _description_
        qkv_bias (bool, optional): _description_. Defaults to False.
        attn_drop (float, optional): _description_. Defaults to 0..
        proj_drop (float, optional): _description_. Defaults to 0..
        relative_position (bool, optional): _description_. Defaults to True.
        max_relative_position (int, optional): _description_. Defaults to 14.
        change_qkv (bool, optional): _description_. Defaults to True.
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 qkv_bias: bool = False,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 relative_position: bool = True,
                 max_relative_position: int = 14,
                 change_qkv: bool = True) -> None:

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.change_qkv = change_qkv
        if change_qkv:
            self.qkv = QkvSlice(
                embed_dim, self.num_heads, unit=64, bias=qkv_bias)
            # self.qkv = QKV_Super(640, 1920, bias=qkv_bias)
        else:
            self.qkv = LinearSlice_byhead(
                embed_dim, 3 * embed_dim, bias=qkv_bias)

        self.relative_position = relative_position
        if self.relative_position:
            self.rel_pos_embed_k = RelativePositionSlice2D(
                self.num_heads,
                unit=64,
                max_relative_position=max_relative_position)
            self.rel_pos_embed_v = RelativePositionSlice2D(
                self.num_heads,
                unit=64,
                max_relative_position=max_relative_position)
        self.max_relative_position = max_relative_position
        self.proj = LinearSlice_byhead(
            embed_dim,
            self.num_heads,
            unit=64,
        )

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
