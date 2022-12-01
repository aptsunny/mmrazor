# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

import torch
import torch.nn as nn

from mmrazor.models.mutables.base_mutable import BaseMutable
from mmrazor.registry import MODELS
from ...ops import InputResizer
from ..mixins.dynamic_mixins import DynamicResizeMixin


# TODO
# consider use data preprocessor
@MODELS.register_module()
class DynamicInputResizer(InputResizer, DynamicResizeMixin):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.mutable_attrs: Dict[str, Optional[BaseMutable]] = nn.ModuleDict()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._size = self.get_dynamic_shape()

        return super().forward(x, self._size)

    @property
    def static_op_factory(self):
        """Corresponding Pytorch OP."""
        return InputResizer

    @classmethod
    def convert_from(cls, module: InputResizer):
        """Convert a Sequential module to a DynamicSequential.

        Args:
            module (:obj:`torch.nn.Sequential`): The original Sequential
                module.
        """
        dynamic_seq = cls(
            size=module._size,
            interpolation_type=module._interpolation_type,
            align_corners=module._align_corners,
            scale_factor=module._scale_factor,
            recompute_scale_factor=module._recompute_scale_factor)

        return dynamic_seq
