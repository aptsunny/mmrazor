# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.models.utils import make_divisible
from mmcv.cnn import ConvModule, build_activation_layer
from mmengine.logging import MMLogger
from mmengine.model import Sequential, constant_init
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.architectures.dynamic_ops import (BigNasConv2d,
                                                      DynamicBatchNorm2d)
from mmrazor.models.architectures.dynamic_ops.bricks import DynamicSequential
from mmrazor.models.architectures.ops.gml_mobilenet_series import GMLMBBlock
from mmrazor.models.mutables import (MutableChannelContainer,
                                     OneShotMutableChannel,
                                     OneShotMutableChannelUnit,
                                     OneShotMutableValue)
from mmrazor.models.mutables.mutable_register import (mutate_conv_module,
                                                      mutate_mobilenet_layer)
from mmrazor.models.utils.parse_values import parse_values
from mmrazor.registry import MODELS

logger = MMLogger.get_current_instance()


@MODELS.register_module()
class AttentiveMobileNet(BaseBackbone):
    """Searchable MobileNetV3 backbone defined by AttentiveNAS."""

    def __init__(self,
                 arch_setting,
                 widen_factor=1.,
                 out_indices=(7, ),
                 frozen_stages=-1,
                 dropout_stages=6,
                 conv_cfg=dict(type='BigNasConv2d'),
                 norm_cfg=dict(type='DynamicBatchNorm2d'),
                 act_cfg=dict(type='MemoryEfficientSwish'),
                 norm_eval=False,
                 zero_init_residual=True,
                 with_cp=False,
                 init_cfg=None):

        super().__init__(init_cfg)

        self.widen_factor = widen_factor
        self.out_indices = out_indices
        for index in out_indices:
            if index not in range(0, 8):
                raise ValueError('the item in out_indices must in '
                                 f'range(0, 8). But received {index}')
        if frozen_stages not in range(-1, 8):
            raise ValueError('frozen_stages must be in range(-1, 8). '
                             f'But received {frozen_stages}')
        if dropout_stages not in range(-1, 8):
            raise ValueError('dropout_stages must be in range(-1, 8). '
                             f'But received {dropout_stages}')
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.dropout_stages = dropout_stages

        self.arch_setting = arch_setting
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.with_cp = with_cp

        self.stride = [1, 2, 2, 2, 1, 2, 1]
        self.with_se_cfg = [False, False, True, False, True, True, True]

        # adapt mutable settings
        self.kernel_size_list = parse_values(self.arch_setting['kernel_size'])
        self.num_blocks_list = parse_values(self.arch_setting['num_blocks'])
        self.expand_ratio_list = \
            parse_values(self.arch_setting['expand_ratio'])
        self.num_channels_list = \
            parse_values(self.arch_setting['num_out_channels'])

        self.num_channels_list = [[
            make_divisible(c * widen_factor, 8) for c in channels
        ] for channels in self.num_channels_list]
        self.num_se_channels_list = [[int(c / 4) for c in channel]
                                     for channel in self.num_channels_list]

        self.first_out_channels_list = self.num_channels_list.pop(0)
        self.last_out_channels_list = self.num_channels_list.pop(-1)
        self.last_expand_ratio_list = self.expand_ratio_list.pop(-1)
        assert len(self.kernel_size_list) == len(self.num_blocks_list) == \
            len(self.expand_ratio_list) == len(self.num_channels_list)

        self.in_channels = max(self.first_out_channels_list)
        self.first_conv = Sequential(
            OrderedDict([('conv',
                          BigNasConv2d(
                              in_channels=3,
                              out_channels=self.in_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1)),
                         ('bn',
                          DynamicBatchNorm2d(num_features=self.in_channels)),
                         ('act', build_activation_layer(self.act_cfg))]))

        self.last_mutable = OneShotMutableChannel(
            num_channels=self.in_channels,
            candidate_choices=self.first_out_channels_list)

        self.layers = self.make_layers()

        last_expand_channels = \
            self.in_channels * max(self.last_expand_ratio_list)
        self.out_channels = max(self.last_out_channels_list)
        last_layers = Sequential(
            OrderedDict([('final_expand_layer',
                          ConvModule(
                              in_channels=self.in_channels,
                              out_channels=last_expand_channels,
                              kernel_size=1,
                              padding=0,
                              conv_cfg=self.conv_cfg,
                              norm_cfg=self.norm_cfg,
                              act_cfg=self.act_cfg)),
                         ('pool', nn.AdaptiveAvgPool2d((1, 1))),
                         ('feature_mix_layer',
                          ConvModule(
                              in_channels=last_expand_channels,
                              out_channels=self.out_channels,
                              kernel_size=1,
                              padding=0,
                              bias=False,
                              conv_cfg=self.conv_cfg,
                              norm_cfg=None,
                              act_cfg=self.act_cfg))]))
        self.add_module('last_conv', last_layers)
        self.layers.append(last_layers)
        self.blocks = self.layers[:-1]

        self.register_mutables()

    def make_layers(self):
        """Build multiple mobilenet layers."""
        layers = []
        for i, (num_blocks, kernel_sizes, expand_ratios, num_channels) in \
            enumerate(zip(self.num_blocks_list, self.kernel_size_list,
                          self.expand_ratio_list, self.num_channels_list)):
            inverted_res_layer = self._make_single_layer(
                layer_index=i,
                out_channels=num_channels,
                num_blocks=num_blocks,
                kernel_sizes=kernel_sizes,
                expand_ratios=expand_ratios,
                stride=self.stride[i],
                use_se=self.with_se_cfg[i])
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, inverted_res_layer)
            layers.append(inverted_res_layer)

        return layers

    def _make_single_layer(self, layer_index: int, out_channels: List,
                           num_blocks: List, kernel_sizes: List,
                           expand_ratios: List, stride: int, use_se: bool):
        """Stack InvertedResidual blocks to build a layer for MobileNetV2.

        Args:
            out_channels (int): out_channels of block.
            num_blocks (int): number of blocks.
            stride (int): stride of the first block. Default: 1
            expand_ratio (int): Expand the number of channels of the
                hidden layer in InvertedResidual by this ratio. Default: 6.
        """
        _layers = []
        for i in range(max(num_blocks)):
            if i >= 1:
                stride = 1
            if use_se:
                se_cfg = dict(
                    act_cfg=(dict(type='ReLU'), dict(type='HSigmoid')),
                    ratio=4,
                    conv_cfg=self.conv_cfg,
                    use_avgpool=False)
            else:
                se_cfg = None  # type: ignore

            mb_layer = GMLMBBlock(
                in_channels=self.in_channels,
                out_channels=max(out_channels),
                kernel_size=max(kernel_sizes),
                stride=stride,
                expand_ratio=max(expand_ratios),
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                with_cp=self.with_cp,
                se_cfg=se_cfg,
                with_attentive_shortcut=True)

            _layers.append(mb_layer)
            self.in_channels = max(out_channels)

        dynamic_seq = DynamicSequential(*_layers)
        return dynamic_seq

    def register_mutables(self):
        """Mutate the BigNAS-style MobileNetV3."""
        OneShotMutableChannelUnit._register_channel_container(
            self, MutableChannelContainer)

        # mutate the first conv
        mutate_conv_module(
            self.first_conv, mutable_out_channels=self.last_mutable)

        # mutate the built mobilenet layers
        for i, layer in enumerate(self.layers[:-1]):
            num_blocks = self.num_blocks_list[i]
            kernel_sizes = self.kernel_size_list[i]
            expand_ratios = self.expand_ratio_list[i]
            out_channels = self.num_channels_list[i]
            se_channels = self.num_se_channels_list[i]

            mutable_kernel_size = OneShotMutableValue(
                value_list=kernel_sizes, default_value=max(kernel_sizes))
            mutable_expand_value = OneShotMutableValue(
                value_list=expand_ratios, default_value=max(expand_ratios))
            mutable_out_channels = OneShotMutableChannel(
                num_channels=max(out_channels), candidate_choices=out_channels)
            mutable_se_channels = OneShotMutableChannel(
                num_channels=max(se_channels), candidate_choices=se_channels)

            for k in range(max(self.num_blocks_list[i])):
                mutate_mobilenet_layer(layer[k], self.last_mutable,
                                       mutable_out_channels,
                                       mutable_se_channels,
                                       mutable_expand_value,
                                       mutable_kernel_size)
                self.last_mutable = mutable_out_channels

            mutable_depth = OneShotMutableValue(
                value_list=num_blocks, default_value=max(num_blocks))
            layer.register_mutable_attr('depth', mutable_depth)

        mutable_out_channels = OneShotMutableChannel(
            num_channels=self.out_channels,
            candidate_choices=self.last_out_channels_list)
        last_mutable_expand_value = OneShotMutableValue(
            value_list=self.last_expand_ratio_list,
            default_value=max(self.last_expand_ratio_list))
        derived_expand_channels = self.last_mutable * last_mutable_expand_value
        mutate_conv_module(
            self.layers[-1].final_expand_layer,
            mutable_in_channels=self.last_mutable,
            mutable_out_channels=derived_expand_channels)
        mutate_conv_module(
            self.layers[-1].feature_mix_layer,
            mutable_in_channels=derived_expand_channels,
            mutable_out_channels=mutable_out_channels)

        self.last_mutable = mutable_out_channels

    def forward(self, x):

        x = self.first_conv(x)
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i in self.out_indices:
                x = torch.squeeze(x, dim=-1)
                x = torch.squeeze(x, dim=-1)
                outs.append(x)

        return tuple(outs)

    def set_dropout(self, drop_prob: float) -> None:
        # drop_path_ratio is set for last two mobile_layer.
        total_block_nums = sum(len(blocks) for blocks in self.blocks[:-1]) + 1
        visited_block_nums = 0
        for idx, layer in enumerate(self.blocks, start=1):
            assert isinstance(layer, DynamicSequential)
            visited_block_nums += len(layer)
            if idx < self.dropout_stages:
                continue

            for mb_idx, mb_layer in enumerate(layer):
                if isinstance(mb_layer, GMLMBBlock):
                    ratio = (visited_block_nums - len(layer) +
                             mb_idx) / total_block_nums
                    mb_drop_prob = drop_prob * ratio
                    mb_layer.drop_prob = mb_drop_prob

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def init_weights(self) -> None:
        super().init_weights()

        if self.zero_init_residual:
            for name, module in self.named_modules():
                if isinstance(module, GMLMBBlock):
                    if module.with_res_shortcut or \
                            module.with_attentive_shortcut:
                        norm_layer = module.linear_conv.norm
                        constant_init(norm_layer, val=0)
                        logger.debug(
                            f'init {type(norm_layer)} of linear_conv in '
                            f'`{name}` to zero')

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.first_conv.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False
