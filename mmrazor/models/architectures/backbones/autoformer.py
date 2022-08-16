# Copyright (c) OpenMMLab. All rights reserved.
from timeit import repeat
from typing import Dict, List

import torch.nn as nn
from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcv.cnn import build_activation_layer, build_norm_layer
from torch import Tensor

from mmrazor.models.architectures.dynamic_op import (DynamicLinear,
                                                     DynamicMultiheadAttention,
                                                     DynamicSequential,
                                                     DynamicPatchEmbed)
from mmrazor.registry import MODELS


@MODELS.register_module
class Autoformer(BaseBackbone):

    # 3 parameters are needed to construct a layer,
    # from left to right: embed_dim, num_head, mlp_ratio, repeat_num
    arch_settings = [[624, 10, 4.0, 16]]
    # mlp_ratio, num_heads
    mutable_settings = {
        'mlp_ratios': [3.0, 3.5, 4.0], # mutable value
        'num_heads': [8, 9, 10], # mutable value
        'depth': [14, 15, 16], # mutable value
        'embed_dims': [528, 576, 624], # mutable channel
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
        
        
        self.mutable_embed_dim
        # patch embeddings
        self.patch_embed = DynamicPatchEmbed(
            
        )

        self.blocks = nn.ModuleList()
        for layer_cfg in self.arch_settings:
            embed_dims, num_heads, mlp_ratios, repeat_num = layer_cfg
            layers = self.make_layers(embed_dims=embed_dims, 
                                      num_heads=num_heads, 
                                      mlp_ratios=mlp_ratios,
                                      repeat_num=repeat_num)
            self.blocks.append(layers)
        
        
            
        

    def make_layers(self, embed_dims, num_heads, mlp_ratios, depth):
        layers = []
        for _ in range(depth):
            layer = TransformerEncoderLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratios,
                drop_rate=0.,
                attn_drop_rate=0.)
            layers.append(layer)

        return DynamicSequential(*layers)


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
