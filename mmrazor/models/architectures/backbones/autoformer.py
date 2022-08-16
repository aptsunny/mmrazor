# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List

import torch
import torch.nn as nn
from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.utils import to_2tuple

from mmrazor.models.architectures.dynamic_op import (DynamicLinear,
                                                     DynamicMultiheadAttention)
from mmrazor.registry import MODELS


@MODELS.register_module
class Autoformer(BaseBackbone):

    def __init__(self, init_cfg=None) -> None:
        super().__init__(init_cfg)


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
        norm_cfg (Dict, optional): _description_. Defaults to dict(type='DynamicLayerNorm')
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
        self.fc1 = DynamicLinear()
