# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch.nn as nn
import torch.nn.functional as F
from mmcls.models.utils import PatchEmbed
from mmcv.cnn.bricks.registry import CONV_LAYERS
from torch import Tensor

from mmrazor.models.mutables.base_mutable import BaseMutable
from .base import ChannelDynamicOP


@CONV_LAYERS.register_module()
class DynamicPatchEmbed(PatchEmbed, ChannelDynamicOP):
    """AI is creating summary for __init__

    Args:
        img_size (int, optional): [description]. Defaults to 224.
        in_channels (int, optional): [description]. Defaults to 3.
        embed_dims ([type], optional): [description]. Defaults to None.
        scale (bool, optional): [description]. Defaults to False.
        sampled_scale (float, optional): [description]. Defaults to 1.0.
        conv_cfg ([type], optional): [description]. Defaults to None.
    """
    accpeted_mutables = {'mutable_embed_dim'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mutable_embed_dim: Optional[BaseMutable] = None
        self.patch_size = 16

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
            return self.projection.weight, self.projection.bias

        choice = self.mutable_embed_dim.current_choice.to(
            self.projection.weight.device)

        weight = self.projection.weight[:choice, ...]
        bias = self.projection.bias[:
                                    choice] if self.projection.bias is not None else None

        return weight, bias

    def forward(self, x: Tensor):
        weight, bias = self._get_dynamic_params()
        x = F.conv2d(
            x,
            weight,
            bias,
            stride=self.patch_size,
            padding=self.projection.padding,
            dilation=self.projection.dilation).flatten(2).transpose(1, 2)

        return x

    def to_static_op(self) -> nn.Module:
        self.check_if_mutables_fixed()
        assert self.mutable_embed_dim is not None

        weight, bias = self._get_dynamic_params()
        static_patch_embed = PatchEmbed(
            img_size=self.img_size,
            in_channels=self.in_channels,
            embed_dims=sum(self.mutable_embed_dim.current_choice.item()),
            conv_cfg=self.conv_cfg)

        static_patch_embed.projection.weight = nn.Parameter(weight)
        static_patch_embed.projection.bias = nn.Parameter(bias)

        return static_patch_embed
