# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcv.cnn import build_activation_layer, build_norm_layer
from torch import Tensor

from mmrazor.models.architectures.dynamic_op import (DynamicLinear,
                                                     DynamicMultiheadAttention,
                                                     DynamicPatchEmbed,
                                                     DynamicSequential)
from mmrazor.models.mutables import OneShotMutableChannel, OneShotMutableValue
from mmrazor.registry import MODELS


class TransformerEncoderLayer(BaseBackbone):
    """_summary_

    Args:
        embed_dims (int): _description_
        num_heads (int): _description_
        mlp_ratio (List): _description_
        drop_rate (float): _description_
        attn_drop_rate (float): _description_
        qkv_bias (bool, optional): _description_. Defaults to True.
        act_cfg (Dict, optional): _description_. Defaults to dict(type='GELU').
        norm_cfg (Dict, optional): _description_.
        init_cfg=None.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 mlp_ratio: List,
                 drop_rate: float,
                 attn_drop_rate: float,
                 qkv_bias: bool = True,
                 act_cfg: Dict = dict(type='GELU'),
                 norm_cfg: Dict = dict(type='DynamicLayerNorm'),
                 init_cfg: Dict = None) -> None:
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, self.embed_dims)
        self.add_module(self.norm1_name, norm1)
        self.attn = DynamicMultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            qkv_bias=qkv_bias)

        self.norm2_name, norm2 = build_norm_layer(norm_cfg, self.embed_dims)
        self.add_module(self.norm2_name, norm2)

        # derived mutable
        self.middle_channels = embed_dims * mlp_ratio
        self.fc1 = DynamicLinear(embed_dims, self.middle_channels)
        self.fc2 = DynamicLinear(self.middle_channels, embed_dims)
        self.act = build_activation_layer(act_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        x = residual + x
        residual = x
        x = self.norm2(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return residual + x


def _mutate_layer(layer: TransformerEncoderLayer,
                  derived_embed_dims=None,
                  derived_num_heads=None,
                  derived_mlp_ratios=None,
                  derived_depth=None):
    """mainly used for embed dims"""

    if derived_embed_dims is not None:
        layer.attn.mutate_embed_dims(derived_embed_dims)
        layer.norm1.mutate_num_channels(derived_embed_dims)
        layer.norm2.mutate_num_channels(derived_embed_dims)

    if derived_num_heads is not None:
        layer.attn.mutate_num_heads(derived_num_heads)


@MODELS.register_module
class Autoformer(BaseBackbone):

    # 3 parameters are needed to construct a layer,
    # from left to right: embed_dim, num_head, mlp_ratio, repeat_num
    arch_settings = {
        'embed_dims': 624,
        'num_heads': 10,
        'mlp_ratios': 4.0,
        'depth': 16,
    }
    # mlp_ratio, num_heads
    mutable_settings = {
        'mlp_ratios': [3.0, 3.5, 4.0],  # mutable value
        'num_heads': [8, 9, 10],  # mutable value
        'depth': [14, 15, 16],  # mutable value
        'embed_dims': [528, 576, 624],  # mutable channel
    }

    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            in_channels: int = 3,
            qkv_bias: bool = True,
            conv_cfg: Dict = dict(type='DynamicConv2d'),  # TODO check
            norm_cfg: Dict = dict(type='DynamicLayerNorm'),
            act_cfg: Dict = dict(type='GELU'),
            final_norm: bool = True,
            init_cfg=None) -> None:
        super().__init__(init_cfg)

        self.img_size = img_size
        self.patch_size = patch_size
        self.qkv_bias = qkv_bias
        self.in_channels = in_channels
        self.dropout = 0.5

        self.embed_dims = self.arch_settings['embed_dims']

        # adopt mutable settings
        mlp_ratio_range = self.mutable_settings['mlp_ratios']
        num_head_range = self.mutable_settings['num_heads']
        depth_range = self.mutable_settings['depth']
        embed_dim_range = self.mutable_settings['embed_dims']

        # mutable value or channel
        self.mutable_embed_dims = OneShotMutableChannel(
            num_channels=max(embed_dim_range),
            candidate_mode='number',
            candidate_choices=embed_dim_range)

        self.mutable_num_heads = OneShotMutableValue(
            value_list=num_head_range, default_value=max(num_head_range))

        self.mutable_depth_range = OneShotMutableValue(
            value_list=depth_range, default_value=max(depth_range))

        self.mutable_mlp_ratio_range = OneShotMutableValue(
            value_list=mlp_ratio_range, default_value=max(mlp_ratio_range))

        # patch embeddings
        self.patch_embed = DynamicPatchEmbed(
            img_size=self.img_size,
            in_channels=self.in_channels,
            embed_dims=self.embed_dims,
            norm_cfg=norm_cfg,
            conv_cfg=conv_cfg,
            init_cfg=init_cfg)

        # num of patches
        self.patch_resolution = [
            img_size // patch_size, img_size // patch_size
        ]
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # cls token and pos embed
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, self.embed_dims))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))

        # main body
        self.blocks = nn.ModuleList()
        for layer_cfg in self.arch_settings:
            embed_dims, num_heads, mlp_ratios, repeat_num = layer_cfg
            layers = self.make_layers(
                embed_dims=embed_dims,
                num_heads=num_heads,
                mlp_ratios=mlp_ratios,
                repeat_num=repeat_num)
            self.blocks.append(layers)

        self.final_norm = final_norm
        if self.final_norm:
            self.norm1_name, norm1 = build_norm_layer(norm_cfg,
                                                      self.embed_dims)
            self.add_module(self.norm1_name, norm1)

    def make_layers(self, embed_dims, num_heads, mlp_ratios, depth):
        layers = []
        for _ in range(depth):
            layer = TransformerEncoderLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratios,
                drop_rate=0.,
                attn_drop_rate=self.dropout)
            layers.append(layer)

        return DynamicSequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        embed_dims = self.mutable_embed_dims.current_choice
        # cls token
        cls_tokens = self.cls_token[..., :embed_dims].expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # pos embed
        x = x + self.pos_embed[..., :embed_dims]

        x = F.dropout(x, p=self.dropout)

        x = x + self.pos_embed()

        for block in self.blocks:
            x = block(x)

        return torch.mean(x[:, 1:], dim=1)


if __name__ == '__main__':
    m = Autoformer()
