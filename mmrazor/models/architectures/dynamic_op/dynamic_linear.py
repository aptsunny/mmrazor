# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmrazor.models.mutables.base_mutable import BaseMutable
from .base import ChannelDynamicOP


class DynamicLinear(nn.Linear, ChannelDynamicOP):
    """Applies a linear transformation to the incoming data according to the
    `mutable_in_features` and `mutable_out_features` dynamically.

    Note that in_features/out_features and mlp_ratios can not be mutable at
    the same time.

    Args:
        in_features_cfg (Dict): Config related to `in_features`.
        out_features_cfg (Dict): Config related to `out_features`.
    """

    accpeted_mutables = {
        'mutable_in_features', 'mutable_out_features', 'mutable_mlp_ratios'
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.mutable_in_features: Optional[BaseMutable] = None
        self.mutable_out_features: Optional[BaseMutable] = None
        self.mutable_mlp_ratios: Optional[BaseMutable] = None

    def mutate_in_features(self, mutable_in_features: BaseMutable) -> None:
        self.check_mutable_channels(mutable_in_features)

        self.mutable_in_features = mutable_in_features

    def mutate_out_features(self, mutable_out_features: BaseMutable) -> None:
        self.check_mutable_channels(mutable_out_features)
        self.mutable_out_features = mutable_out_features

    def mutate_mlp_ratios(self, mutable_mlp_ratios: BaseMutable) -> None:
        self.check_if_mutables_fixed(mutable_mlp_ratios)
        self.mutable_mlp_ratios = mutable_mlp_ratios

    @property
    def mutable_in(self) -> Optional[BaseMutable]:
        """Mutable `in_features`."""
        if isinstance(self.mutable_in_features, BaseMutable) and \
                isinstance(self.mutable_mlp_ratios, BaseMutable):
            raise '`in_features` and `mlp_ratios` can not be mutable' \
                'at the same time.'

        if isinstance(self.mutable_in_features, BaseMutable):
            return self.mutable_in_features
        elif isinstance(self.mutable_mlp_ratios, BaseMutable):
            return self.in_features * self.mutable_mlp_ratios
        else:
            raise '`in_features` and `mlp_ratios` are both not mutable.'

    @property
    def mutable_out(self) -> Optional[BaseMutable]:
        """Mutable `out_features`."""
        if isinstance(self.mutable_out_features, BaseMutable) and \
            isinstance(self.mutable_mlp_ratios, BaseMutable):
            raise '`out_features` and `mlp_ratios` can not be mutable' \
                'at the same time.'

        if isinstance(self.mutable_out_features, BaseMutable):
            return self.mutable_out_features
        elif isinstance(self.mutable_mlp_ratios, BaseMutable):
            return self.out_features * self.mutable_mlp_ratios
        else:
            raise '`out_features` and `mlp_ratios` are both not mutable.'

    def _get_dynamic_params(self) -> Tuple[Tensor, Optional[Tensor]]:
        if self.mutable_in_features is None and \
                self.mutable_out_features is None:
            return self.weight, self.bias

        if self.mutable_in_features is not None:
            in_mask = self.mutable_in_features.current_mask.to(
                self.weight.device)
        else:
            in_mask = torch.ones(self.weight.size(1)).bool().to(
                self.weight.device)
        if self.mutable_out_features is not None:
            out_mask = self.mutable_out_features.current_mask.to(
                self.weight.device)
        else:
            out_mask = torch.ones(self.weight.size(0)).bool().to(
                self.weight.device)

        weight = self.weight[out_mask][:, in_mask]
        bias = self.bias[out_mask] if self.bias is not None else None

        return weight, bias

    def forward(self, input: Tensor) -> Tensor:
        """Slice the parameters according to `mutable_in_features` and
        `mutable_out_features`, and forward."""
        weight, bias = self._get_dynamic_params()

        return F.linear(input, weight, bias)

    def to_static_op(self) -> nn.Module:
        self.check_if_mutables_fixed()

        weight, bias = self._get_dynamic_params()
        out_features = weight.size(0)
        in_features = weight.size(1)

        static_linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=True if bias is not None else False)

        static_linear.weight = nn.Parameter(weight)
        if bias is not None:
            static_linear.bias = nn.Parameter(bias)

        return static_linear
