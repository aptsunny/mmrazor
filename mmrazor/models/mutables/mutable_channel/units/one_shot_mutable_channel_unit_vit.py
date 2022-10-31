# Copyright (c) OpenMMLab. All rights reserved.
from collections import Set
from typing import Dict, Type

import torch.nn as nn
from mmcls.models.utils import PatchEmbed
from torch.nn import LayerNorm

from mmrazor.models.architectures import dynamic_ops
from mmrazor.models.architectures.dynamic_ops.mixins import DynamicChannelMixin
from mmrazor.models.architectures.ops import (MultiheadAttention,
                                              RelativePosition2D)
from mmrazor.models.mutables import DerivedMutable
from mmrazor.models.mutables.mutable_channel import (BaseMutableChannel,
                                                     MutableChannelContainer)
from .mutable_channel_unit import MutableChannelUnit
from .one_shot_mutable_channel_unit import OneShotMutableChannelUnit
from mmrazor.registry import MODELS
from .channel_unit import Channel


@MODELS.register_module()
class OneShotMutableChannelUnit_VIT(OneShotMutableChannelUnit):

    def add_ouptut_related(self, channel: Channel):
        """Add a Channel which is output related."""
        assert channel.is_output_channel
        # assert self.num_channels == channel.num_channels
        if channel not in self.output_related:
            self.output_related.append(channel)

    def add_input_related(self, channel: Channel):
        """Add a Channel which is input related."""
        assert channel.is_output_channel is False
        # assert self.num_channels == channel.num_channels
        if channel not in self.input_related:
            self.input_related.append(channel)

    @classmethod
    def init_from_predefined_model(cls, model: nn.Module):
        """Initialize units using the model with pre-defined dynamicops and
        mutable-channels."""

        def process_container(contanier: MutableChannelContainer,
                              module,
                              module_name,
                              mutable2units,
                              is_output=True):
            for index, mutable in contanier.mutable_channels.items():
                if isinstance(mutable, DerivedMutable):
                    source_mutables: Set = \
                        mutable._trace_source_mutables()
                    source_channel_mutables = [
                        mutable for mutable in source_mutables
                        if isinstance(mutable, BaseMutableChannel)
                    ]
                    assert len(source_channel_mutables) == 1, (
                        'only support one mutable channel '
                        'used in DerivedMutable')
                    mutable = list(source_channel_mutables)[0]

                if mutable not in mutable2units:
                    mutable2units[mutable] = cls.init_from_mutable_channel(
                        mutable)

                unit: MutableChannelUnit = mutable2units[mutable]
                if is_output:
                    unit.add_ouptut_related(
                        Channel(
                            module_name,
                            module,
                            index,
                            is_output_channel=is_output))
                else:
                    unit.add_input_related(
                        Channel(
                            module_name,
                            module,
                            index,
                            is_output_channel=is_output))

        mutable2units: Dict = {}
        for name, module in model.named_modules():
            # [blocker]
            # if isinstance(module, DynamicMultiheadAttention):
            # if isinstance(module, ):
            #     in_container: MutableChannelContainer = \
            #         module.get_mutable_attr(
            #             'in_channels')
            #     out_container: MutableChannelContainer = \
            #         module.get_mutable_attr(
            #             'out_channels')
            #     process_container(in_container, module, name, mutable2units,
            #                       False)
            #     process_container(out_container, module, name, mutable2units,
            #                       True)

            if isinstance(module, (DynamicChannelMixin, MultiheadAttention)):
                in_container: MutableChannelContainer = \
                    module.get_mutable_attr(
                        'in_channels')
                out_container: MutableChannelContainer = \
                    module.get_mutable_attr(
                        'out_channels')
                process_container(in_container, module, name, mutable2units,
                                  False)
                process_container(out_container, module, name, mutable2units,
                                  True)
        units = list(mutable2units.values())
        return units

    @staticmethod
    def _register_channel_container(
            model: nn.Module, container_class: Type[MutableChannelContainer]):
        """register channel container for dynamic ops."""
        """
        for module in model.modules():
            if isinstance(module, dynamic_ops.DynamicChannelMixin):
                if module.get_mutable_attr('in_channels') is None:
                    in_channels = 0
                    if isinstance(module, nn.Conv2d):
                        in_channels = module.in_channels
                    elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                        in_channels = module.num_features
                    elif isinstance(module, nn.Linear):
                        in_channels = module.in_features
                    else:
                        raise NotImplementedError()
                    module.register_mutable_attr('in_channels',
                                                 container_class(in_channels))
                if module.get_mutable_attr('out_channels') is None:
                    out_channels = 0
                    if isinstance(module, nn.Conv2d):
                        out_channels = module.out_channels
                    elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                        out_channels = module.num_features
                    elif isinstance(module, nn.Linear):
                        out_channels = module.out_features
                    else:
                        raise NotImplementedError()
                    module.register_mutable_attr('out_channels',
                                                 container_class(out_channels))

        """
        for module in model.modules():
            # [blocker]
            # if isinstance(module, DynamicMultiheadAttention):
            if isinstance(module, MultiheadAttention):
                in_channels = module.embed_dims
                module.register_mutable_attr('embed_dims',
                                             container_class(in_channels))
                out_channels = module.q_embed_dims
                module.register_mutable_attr('q_embed_dims',
                                             container_class(out_channels))

            if isinstance(module, dynamic_ops.DynamicChannelMixin):
                if module.get_mutable_attr('in_channels') is None:
                    in_channels = 0
                    if isinstance(module, nn.Conv2d):
                        in_channels = module.in_channels
                    elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                        in_channels = module.num_features
                    elif isinstance(module, nn.Linear):
                        in_channels = module.in_features
                    # autoformer
                    elif isinstance(module, PatchEmbed):
                        in_channels = module.embed_dims
                    elif isinstance(module, LayerNorm):
                        in_channels = module.normalized_shape[0]
                    elif isinstance(module, RelativePosition2D):
                        in_channels = module.head_dims
                    else:
                        raise NotImplementedError()
                    module.register_mutable_attr('in_channels',
                                                 container_class(in_channels))

                if module.get_mutable_attr('out_channels') is None:
                    out_channels = 0
                    if isinstance(module, nn.Conv2d):
                        out_channels = module.out_channels
                    elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                        out_channels = module.num_features
                    elif isinstance(module, nn.Linear):
                        out_channels = module.out_features
                    # autoformer
                    elif isinstance(module, PatchEmbed):
                        out_channels = module.embed_dims
                    elif isinstance(module, LayerNorm):
                        out_channels = module.normalized_shape[0]
                    elif isinstance(module, RelativePosition2D):
                        out_channels = module.head_dims
                    else:
                        raise NotImplementedError()
                    module.register_mutable_attr('out_channels',
                                                 container_class(out_channels))