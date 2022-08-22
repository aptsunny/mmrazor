# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Iterator, Optional, Sequence

import torch.nn as nn
from mmengine.model import Sequential
from torch import Tensor
from torch.nn import Module

from mmrazor.models.mutables import DerivedMutable, MutableValue
from mmrazor.models.mutables.base_mutable import BaseMutable
from .dynamic_mixins import DynamicSequentialMixin


class DynamicSequential(Sequential, DynamicSequentialMixin):
    accepted_mutable_attrs = {'depth'}

    forward_ignored_module = (MutableValue, DerivedMutable)

    def __init__(self, *args, init_cfg: Optional[dict] = None):
        super().__init__(*args, init_cfg=init_cfg)

        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

    def mutate_depth(self,
                     mutable_depth: BaseMutable,
                     depth_seq: Optional[Sequence[int]] = None) -> None:

        if depth_seq is None:
            depth_seq = getattr(mutable_depth, 'choices')

        if depth_seq is None:
            raise ValueError('depth sequence must be provided')

        depth_list = list(sorted(depth_seq))
        if depth_list[-1] != len(self):
            raise ValueError(f'Expect max depth to be: {len(self)}, '
                             f'but got: {depth_list[-1]}')

        self.depth_list = depth_list
        self.mutable_depth = mutable_depth

    def forward(self, x: Tensor) -> Tensor:
        if self.mutable_depth is None:
            return self(x)

        current_depth = self.get_current_choice(self.mutable_depth)
        passed_module_nums = 0
        for module in self.pure_modules():
            passed_module_nums += 1
            if passed_module_nums > current_depth:
                break
            x = module(x)
        return x

    @property
    def pure_module_nums(self) -> int:
        return sum(1 for _ in self.pure_modules())

    def pure_modules(self) -> Iterator[Module]:
        for module in self._modules.values():
            if isinstance(module, self.forward_ignored_module):
                continue
            yield module
