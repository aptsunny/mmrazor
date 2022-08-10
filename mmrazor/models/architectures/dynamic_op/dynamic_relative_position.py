# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.nn as nn
from mmcv.cnn.utils.weight_init import trunc_normal_
from torch import Tensor

from mmrazor.models.mutables.base_mutable import BaseMutable
from .base import ChannelDynamicOP


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
    accpeted_mutables = {'mutable_head_dims'}

    def __init__(self, head_dims=14, max_relative_position=14):
        super().__init__(
            head_dims=head_dims, max_relative_position=max_relative_position)

        self.mutable_head_dims: Optional[BaseMutable] = None

    def mutate_head_dims(self, mutable_head_dims):
        self.mutable_head_dims = mutable_head_dims

    @property
    def mutable_in(self) -> Optional[BaseMutable]:
        return self.mutable_head_dims

    @property
    def mutable_out(self) -> Optional[BaseMutable]:
        return self.mutable_head_dims

    def _get_dynamic_params(self, length_q,
                            length_k) -> Tuple[Tensor, Optional[Tensor]]:
        if self.mutable_head_dims is None:
            self.current_head_dim = self.head_dims
        else:
            self.current_head_dim = self.mutable_head_dims.current_choice

        self.sample_eb_table_h = self.embeddings_table_h[:, :self.
                                                         current_head_dim]
        self.sample_eb_table_v = self.embeddings_table_v[:, :self.
                                                         current_head_dim]

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
        distance_mat_clipped_v = torch.clamp(distance_mat_v,
                                             -self.max_relative_position,
                                             self.max_relative_position)
        distance_mat_clipped_h = torch.clamp(distance_mat_h,
                                             -self.max_relative_position,
                                             self.max_relative_position)

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

        embeddings = self.sample_eb_table_v[final_mat_v] + \
            self.sample_eb_table_h[final_mat_h]

        return embeddings

    def forward(self, length_q, length_k):
        return self._get_dynamic_params(length_q, length_k)

    def to_static_op(self) -> nn.Module:
        self.check_if_mutables_fixed()

        static_relative_position = RelativePosition2D(self.current_head_dim)
        static_relative_position.embeddings_table_v = \
            nn.Parameter(
                self.embeddings_table_v[:, :self.current_head_dim].clone())
        static_relative_position.embeddings_table_h = \
            nn.Parameter(
                self.embeddings_table_h[:, :self.current_head_dim].clone())

        return static_relative_position
