# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from .base import ChannelDynamicOP


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
    accpeted_mutables = {
        'mutable_num_heads',
        'mutable_in_features',
        'mutable_heads_dim',
    }

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
        self.proj = LinearSlice_byhead(embed_dim, self.num_heads, unit=64)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

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

    def export(self, **kwargs):
        return self
