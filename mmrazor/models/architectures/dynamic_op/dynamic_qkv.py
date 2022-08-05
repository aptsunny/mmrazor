# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .base import ChannelDynamicOP
from mmrazor.models.mutables.base_mutable import BaseMutable


class QKV(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True):
        super().__init__(in_features, out_features, bias=bias)
        # super_in_dim and super_out_dim indicate the largest network!
        self.in_features = in_features
        self.out_features = out_features
        
    def forward(self, x):
        sample_weight = self.weight[:, :self.in_features]
        sample_weight = torch.cat([sample_weight[i:self.out_features:3, :] for i in range(3)], dim=0)
        return F.linear(x, sample_weight, self.bias[:self.out_features])


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
        
        max_in_features = self.get_value(in_features, kind='max')
        max_heads = self.get_value(num_heads, kind='max')
        
        max_out_features = max_heads * unit * 3
        
        super(QkvSlice, self).__init__(max_in_features, max_out_features, bias)
        self.in_features = in_features
        self.num_heads = num_heads
        self.unit = unit

    def mutate_num_heads(self, mutable_num_heads):
        ...

    @property
    def mutable_in(self) -> Optional[BaseMutable]:
        return self.mutable_embed_dim 
    
    @property
    def mutable_out(self) -> Optional[BaseMutable]:
        return self.mutable_embed_dim

    def forward_inner(self, x):
        in_features = self.get_value(self.in_features)
        out_features = self.get_value(self.num_heads) * self.unit * 3
        sample_weight = self.weight[:, :in_features]
        sample_weight = torch.cat(
            [sample_weight[i:out_features:3, :] for i in range(3)], dim=0)
        bias = self.bias[:out_features]
        return F.linear(x, sample_weight, bias)

    # def export(self, **kwargs):
    #     """Export LinearSlice to nn.Linear."""
    #     in_features = kwargs.get('in_features',
    #                              self.get_value(self.in_features))
    #     num_heads = kwargs.get('num_heads', self.get_value(self.num_heads))
    #     out_features = self.unit * num_heads * 3
    #     sample_weight = self.weight[:out_features, :in_features]
    #     sample_weight = sample_weight.data
    #     export_module = QKV_Super(
    #         in_features, out_features, bias=(self.bias is not None))
    #     export_module.weight.data.copy_(sample_weight)
    #     if self.bias is not None:
    #         export_module.bias.data.copy_(self.bias.data[:out_features])
    #     return export_module
