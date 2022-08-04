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


class Patchembed(nn.Module):
    """AI is creating summary for __init__

    Args:
        img_size (int, optional): [description]. Defaults to 224.
        patch_size (int, optional): [description]. Defaults to 16.
        in_channels (int, optional): [description]. Defaults to 3.
        embed_dim (int, optional): [description]. Defaults to 576.
        scale (bool, optional): [description]. Defaults to False.
        sampled_scale (float, optional): [description]. Defaults to 1.0.
        conv_cfg ([type], optional): [description]. Defaults to None.
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=576,
                 conv_cfg=dict(type='Conv2d')):

        super(Patchembed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.in_channels = in_channels
        self.conv_cfg = conv_cfg
        
        kwargs = dict(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size)
        
        self.proj = build_conv_layer(cfg=conv_cfg, **kwargs)

        self.embed_dim = embed_dim
        
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

@CONV_LAYERS.register_module()
class DynamicPatchembed(Patchembed, ChannelDynamicOP):
    """AI is creating summary for __init__

    Args:
        img_size (int, optional): [description]. Defaults to 224.
        patch_size (int, optional): [description]. Defaults to 16.
        in_channels (int, optional): [description]. Defaults to 3.
        embed_dim ([type], optional): [description]. Defaults to None.
        scale (bool, optional): [description]. Defaults to False.
        sampled_scale (float, optional): [description]. Defaults to 1.0.
        conv_cfg ([type], optional): [description]. Defaults to None.
    """
    accpeted_mutables = {'mutable_embed_dim'}
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mutable_embed_dim: Optional[BaseMutable] = None
        
    def mutate_embed_dim(self, mutable_embed_dim: BaseMutable) -> None:
        self.check_mutable_channels(mutable_embed_dim)
        self.mutable_embed_dim = mutable_embed_dim
        
    @property
    def mutable_in(self) -> Optional[BaseMutable]:
        return self.mutable_embed_dim 
    
    @property
    def mutable_out(self) -> Optional[BaseMutable]:
        return self.mutable_embed_dim

    def _get_dynamic_params(self) -> Tuple[Tensor, Optional[Tensor]]:
        if self.mutable_embed_dim is None:
            return self.proj.weight, self.proj.bias 
        
        mask = self.mutable_embed_dim.current_mask.to(self.proj.weight.device)
        
        weight = self.proj.weight[mask, ...]
        bias = self.proj.bias[mask] if self.proj.bias is not None else None 

        return weight, bias
    
    def forward(self, x: Tensor):
        weight, bias = self._get_dynamic_params()
        x = F.conv2d(
            x,
            weight,
            bias,
            stride=self.patch_size,
            padding=self.proj.padding,
            dilation=self.proj.dilation).flatten(2).transpose(1, 2)

        return x
        
    def to_static_op(self) -> nn.Module:
        self.check_if_mutables_fixed()
        
        weight, bias = self._get_dynamic_params()
        static_patch_embed = Patchembed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=sum(self.mutable_embed_dim.current_mask.item()),
            conv_cfg=self.conv_cfg
        )
        static_patch_embed.proj.weight = nn.Parameter(weight)
        static_patch_embed.proj.bias = nn.Parameter(bias)
        
        return static_patch_embed