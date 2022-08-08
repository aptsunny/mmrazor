# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmrazor.models.mutables.base_mutable import BaseMutable
from .base import ChannelDynamicOP


class DynamicQKV(nn.Linear, ChannelDynamicOP):
    """Sliceable Linear module.

    Args:
        in_features (int/Int): The same with Linear.
        num_heads (int/Int): Parallel attention heads.
        bias (bool): The same with Linear. Defaults to True.
    """
    accpeted_mutables = {'mutable_num_heads', 'mutable_in_features'}

    def __init__(self, in_features, num_heads, unit=64, bias=True, key=None):
        # Must initionalize first for multiple inheritance.

        self.mutable_num_heads: Optional[BaseMutable] = None
        self.mutable_in_features: Optional[BaseMutable] = None
        out_features = num_heads * unit * 3
        super().__init__(in_features, out_features, bias)

        self.in_features = in_features
        self.num_heads = num_heads
        self.unit = unit

    def mutate_num_heads(self, mutable_num_heads):
        self.mutable_num_heads = mutable_num_heads

    @property
    def mutable_in(self) -> Optional[BaseMutable]:
        return self.mutable_in_features

    @property
    def mutable_out(self) -> Optional[BaseMutable]:
        return self.mutable_num_heads * self.unit * 3

    def _get_dynamic_params(self) -> Tuple[Tensor, Optional[Tensor]]:
        if self.mutable_in_features is None and self.mutable_num_heads is None:
            return self.weight, self.bias

        if self.mutable_in_features is not None:
            in_features = self.mutable_in_features.current_choice.to(
                self.weight.device)
        else:
            in_features = self.in_features

        if self.mutable_num_heads is not None:
            out_features = self.mutable_num_heads.current_choice.repeat(3).to(
                self.weight.device)
        else:
            out_features = self.num_heads * self.unit * 3

        weight = self.weight[:, :in_features]
        weight = torch.cat([weight[i:out_features:3, :] for i in range(3)],
                           dim=0)
        bias = self.bias[:out_features] if self.bias is not None else None
        return weight, bias

    def forward(self, input: Tensor) -> Tensor:
        """Slice the parameters according to `mutable_in_features` and
        `mutable_out_features`, and forward."""
        weight, bias = self._get_dynamic_params()

        return F.linear(input, weight, bias)

    def to_static_op(self) -> nn.Module:
        self.check_if_mutables_fixed()
        weight, bias = self._get_dynamic_params()
        static_linear = nn.Linear(
            in_features=weight.size(1),
            out_features=weight.size(0),
            bias=True if bias is not None else False)
        static_linear.weight = nn.Parameter(weight)
        static_linear.bias = nn.Parameter(bias)
        return static_linear
