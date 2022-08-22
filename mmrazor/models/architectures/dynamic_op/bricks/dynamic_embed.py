# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import torch.nn as nn
import torch.nn.functional as F
from mmcls.models.utils import PatchEmbed
from mmcv.cnn.bricks.registry import CONV_LAYERS
from torch import Tensor

from mmrazor.models.mutables.base_mutable import BaseMutable
from .dynamic_mixins import DynamicPatchEmbedMixin


@CONV_LAYERS.register_module()
class DynamicPatchEmbed(PatchEmbed, DynamicPatchEmbedMixin):
    """AI is creating summary for __init__

    Args:
        img_size (int, optional): [description]. Defaults to 224.
        in_channels (int, optional): [description]. Defaults to 3.
        embed_dims ([type], optional): [description]. Defaults to None.
        scale (bool, optional): [description]. Defaults to False.
        sampled_scale (float, optional): [description]. Defaults to 1.0.
        conv_cfg ([type], optional): [description]. Defaults to None.
    """
    accpeted_mutables = {'embed_dims'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

    @property
    def static_op_factory(self):
        return PatchEmbed

    @classmethod
    def convert_from(cls, module):
        """Convert a PatchEmbed to a DynamicPatchEmbed."""

        dynamic_patch_embed = cls(
            img_size=module.img_size,
            in_channels=3,
            embed_dims=module.embed_dims,
            norm_cfg=None,
            conv_cfg=None,
            init_cfg=None)

        return dynamic_patch_embed

    def forward(self, x: Tensor):
        weight, bias = self._get_dynamic_params()
        x = F.conv2d(
            x,
            weight,
            bias,
            stride=16,
            padding=self.projection.padding,
            dilation=self.projection.dilation).flatten(2).transpose(1, 2)

        return x
