# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import torch
import torch.nn as nn
from mmcv.cnn.utils.weight_init import trunc_normal_
from torch import Tensor

from mmrazor.models.mutables.base_mutable import BaseMutable
from ..base import ChannelDynamicOP


class RelativePosition2D(nn.Module):
    """Rethinking and Improving Relative Position Encoding for Vision
    Transformer.

    ICCV 2021. https://arxiv.org/pdf/2107.14222.pdf

    Image RPE (iRPE for short) methods are new relative position encoding
    methods dedicated to 2D images.

    Args:
        head_dims ([int]): embedding dims of relative position.
        max_relative_position ([int]): The max relative position distance.
    """

    def __init__(self, head_dims, max_relative_position=14):
        super().__init__()

        self.head_dims = head_dims
        self.max_relative_position = max_relative_position
        # The first element in embeddings_table_v is the vertical embedding
        # for the class
        self.embeddings_table_v = nn.Parameter(
            torch.randn(max_relative_position * 2 + 2, head_dims))
        self.embeddings_table_h = nn.Parameter(
            torch.randn(max_relative_position * 2 + 2, head_dims))

        trunc_normal_(self.embeddings_table_v, std=.02)
        trunc_normal_(self.embeddings_table_h, std=.02)

    def forward(self, length_q, length_k):
        # remove the first cls token distance computation
        length_q = length_q - 1
        length_k = length_k - 1
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        # compute the row and column distance
        distance_mat_v = (
            range_vec_k[None, :] // int(length_q**0.5) -
            range_vec_q[:, None] // int(length_q**0.5))
        distance_mat_h = (
            range_vec_k[None, :] % int(length_q**0.5) -
            range_vec_q[:, None] % int(length_q**0.5))
        # clip the distance to the range of
        # [-max_relative_position, max_relative_position]
        distance_mat_clipped_v = torch.clamp(distance_mat_v,
                                             -self.max_relative_position,
                                             self.max_relative_position)
        distance_mat_clipped_h = torch.clamp(distance_mat_h,
                                             -self.max_relative_position,
                                             self.max_relative_position)

        # translate the distance from [1, 2 * max_relative_position + 1],
        # 0 is for the cls token
        final_mat_v = distance_mat_clipped_v + self.max_relative_position + 1
        final_mat_h = distance_mat_clipped_h + self.max_relative_position + 1
        # pad the 0 which represent the cls token
        final_mat_v = torch.nn.functional.pad(final_mat_v, (1, 0, 1, 0),
                                              'constant', 0)
        final_mat_h = torch.nn.functional.pad(final_mat_h, (1, 0, 1, 0),
                                              'constant', 0)

        final_mat_v = torch.LongTensor(final_mat_v)
        final_mat_h = torch.LongTensor(final_mat_h)
        # get the embeddings with the corresponding distance
        embeddings = self.embeddings_table_v[
            final_mat_v] + self.embeddings_table_h[final_mat_h]

        return embeddings


class DynamicRelativePosition2D(RelativePosition2D, ChannelDynamicOP):
    """Searchable RelativePosition module.

    Args:
        head_dims (int/Int): Parallel attention heads.
        head_dims ([int/Int]): embedding dims of relative position.
        max_relative_position ([int]): The max relative position distance.
    """
    accepted_mutable_attrs = {'head_dims'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

    @property
    def static_op_factory(self):
        return RelativePosition2D

    @classmethod
    def convert_from(cls, module):
        """Convert a RP to a dynamic RP."""
        dynamic_rp = cls(
            head_dims=module.head_dims,
            max_relative_position=module.max_relative_position)
        return dynamic_rp

    def forward(self, length_q, length_k) -> Tensor:
        return self.forward_mixin(length_q, length_k)
