# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.registry import NORM_LAYERS
from torch import Tensor
from torch.nn.modules import GroupNorm, LayerNorm
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm

from mmrazor.models.mutables.base_mutable import BaseMutable
from mmrazor.registry import MODELS
from ...mutables import MutableManageMixIn
from .base import ChannelDynamicOP


class _DynamicBatchNorm(_BatchNorm, ChannelDynamicOP):
    """Applies Batch Normalization over an input according to the
    `mutable_num_features` dynamically.

    Args:
        num_features_cfg (Dict): Config related to `num_features`.
    """
    accpeted_mutables = {'mutable_num_features'}
    batch_norm_type: str

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.mutable_num_features: Optional[BaseMutable] = None

    def mutate_num_features(self, mutable_num_features: BaseMutable) -> None:
        self.check_mutable_channels(mutable_num_features)

        self.mutable_num_features = mutable_num_features

    @property
    def mutable_in(self) -> Optional[BaseMutable]:
        """Mutable `num_features`."""
        return self.mutable_num_features

    @property
    def mutable_out(self) -> Optional[BaseMutable]:
        """Mutable `num_features`."""
        return self.mutable_num_features

    def _get_dynamic_params(
        self
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor],
               Optional[Tensor]]:
        if self.mutable_num_features is not None:
            out_mask = self.mutable_num_features.current_mask.to(
                self.weight.device)
        else:
            out_mask = torch.ones_like(self.weight).bool()

        if self.affine:
            weight = self.weight[out_mask]
            bias = self.bias[out_mask]
        else:
            weight, bias = self.weight, self.bias

        if self.track_running_stats:
            running_mean = self.running_mean[out_mask] \
                if not self.training or self.track_running_stats else None
            running_var = self.running_var[out_mask] \
                if not self.training or self.track_running_stats else None
        else:
            running_mean, running_var = self.running_mean, self.running_var

        return running_mean, running_var, weight, bias

    def forward(self, input: Tensor) -> Tensor:
        """Slice the parameters according to `mutable_num_features`, and
        forward."""
        self._check_input_dim(input)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:  # type: ignore
                self.num_batches_tracked = \
                    self.num_batches_tracked + 1  # type: ignore
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(
                        self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is
                                                           None)

        running_mean, running_var, weight, bias = self._get_dynamic_params()

        return F.batch_norm(input, running_mean, running_var, weight, bias,
                            bn_training, exponential_average_factor, self.eps)

    def to_static_op(self) -> nn.Module:
        self.check_if_mutables_fixed()

        running_mean, running_var, weight, bias = self._get_dynamic_params()
        if self.mutable_num_features is not None:
            num_features = self.mutable_num_features.current_mask.sum().item()
        else:
            num_features = self.num_features

        static_bn = getattr(nn, self.batch_norm_type)(
            num_features=num_features,
            eps=self.eps,
            momentum=self.momentum,
            affine=self.affine,
            track_running_stats=self.track_running_stats)

        if running_mean is not None:
            static_bn.running_mean.copy_(running_mean)
        if running_var is not None:
            static_bn.running_var.copy_(running_var)
        if weight is not None:
            static_bn.weight = nn.Parameter(weight)
        if bias is not None:
            static_bn.bias = nn.Parameter(bias)

        return static_bn


@NORM_LAYERS.register_module()
class DynamicBatchNorm1d(_DynamicBatchNorm):
    batch_norm_type: str = 'BatchNorm1d'

    def _check_input_dim(self, input: Tensor) -> None:
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(
                input.dim()))


@NORM_LAYERS.register_module()
class DynamicBatchNorm2d(_DynamicBatchNorm):
    batch_norm_type: str = 'BatchNorm2d'

    def _check_input_dim(self, input: Tensor) -> None:
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(
                input.dim()))


@NORM_LAYERS.register_module()
class DynamicBatchNorm3d(_DynamicBatchNorm):
    batch_norm_type: str = 'BatchNorm3d'

    def _check_input_dim(self, input: Tensor) -> None:
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(
                input.dim()))


class DynamicInstanceNorm(_InstanceNorm, MutableManageMixIn):
    """Applies Instance Normalization over an input according to the
    `mutable_num_features` dynamically.

    Args:
        num_features_cfg (Dict): Config related to `num_features`.
    """

    def __init__(self, num_features_cfg, *args, **kwargs):
        super(DynamicInstanceNorm, self).__init__(*args, **kwargs)

        num_features_cfg_ = copy.deepcopy(num_features_cfg)
        num_features_cfg_.update(dict(num_channels=self.num_features))
        self.mutable_num_features = MODELS.build(num_features_cfg_)

    @property
    def mutable_in(self):
        """Mutable `num_features`."""
        return self.mutable_num_features

    @property
    def mutable_out(self):
        """Mutable `num_features`."""
        return self.mutable_num_features

    def forward(self, input: Tensor) -> Tensor:
        """Slice the parameters according to `mutable_num_features`, and
        forward."""
        if self.affine:
            out_mask = self.mutable_num_features.current_mask.to(
                self.weight.device)
            weight = self.weight[out_mask]
            bias = self.bias[out_mask]
        else:
            weight, bias = self.weight, self.bias

        if self.track_running_stats:
            out_mask = self.mutable_num_features.current_mask.to(
                self.running_mean.device)
            running_mean = self.running_mean[out_mask]
            running_var = self.running_var[out_mask]
        else:
            running_mean, running_var = self.running_mean, self.running_var

        return F.instance_norm(input, running_mean, running_var, weight, bias,
                               self.training or not self.track_running_stats,
                               self.momentum, self.eps)


class DynamicGroupNorm(GroupNorm, MutableManageMixIn):
    """Applies Group Normalization over a mini-batch of inputs according to the
    `mutable_num_channels` dynamically.

    Args:
        num_channels_cfg (Dict): Config related to `num_channels`.
    """

    def __init__(self, num_channels_cfg, *args, **kwargs):
        super(DynamicGroupNorm, self).__init__(*args, **kwargs)

        num_channels_cfg_ = copy.deepcopy(num_channels_cfg)
        num_channels_cfg_.update(dict(num_channels=self.num_channels))
        self.mutable_num_channels = MODELS.build(num_channels_cfg_)

    @property
    def mutable_in(self):
        """Mutable `num_channels`."""
        return self.mutable_num_channels

    @property
    def mutable_out(self):
        """Mutable `num_channels`."""
        return self.mutable_num_channels

    def forward(self, input: Tensor) -> Tensor:
        """Slice the parameters according to `mutable_num_channels`, and
        forward."""
        if self.affine:
            out_mask = self.mutable_num_channels.current_mask.to(
                self.weight.device)
            weight = self.weight[out_mask]
            bias = self.bias[out_mask]
        else:
            weight, bias = self.weight, self.bias

        return F.group_norm(input, self.num_groups, weight, bias, self.eps)


class DynamicLayerNorm(LayerNorm, MutableManageMixIn):
    """Applies Layer Normalization over a mini-batch of inputs according to the
    `mutable_num_channels` dynamically.

    Args:
        num_channels_cfg (Dict): Config related to `num_channels`.
    """

    def __init__(self, num_channels_cfg, *args, **kwargs):
        super(DynamicLayerNorm, self).__init__(*args, **kwargs)

        num_channels_cfg_ = copy.deepcopy(num_channels_cfg)
        num_channels_cfg_.update(dict(num_channels=self.num_channels))
        self.mutable_num_channels = MODELS.build(num_channels_cfg_)

    def mutate_num_channels(self, mutable_num_channels):
        self.mutable_num_channels = mutable_num_channels

    @property
    def mutable_in(self):
        """Mutable `num_channels`."""
        return self.mutable_num_channels

    @property
    def mutable_out(self):
        """Mutable `num_channels`."""
        return self.mutable_num_channels

    def forward(self, input: Tensor) -> Tensor:
        """Slice the parameters according to `mutable_num_channels`, and
        forward."""
        if self.affine:
            out_mask = self.mutable_num_channels.current_mask.to(
                self.weight.device)
            weight = self.weight[out_mask]
            bias = self.bias[out_mask]
        else:
            weight, bias = self.weight, self.bias

        return F.layer_norm(input, self.num_groups, weight, bias, self.eps)
