# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch

from mmrazor.models.architectures.dynamic_op import DynamicMultiheadAttention
from mmrazor.models.mutables import OneShotMutableChannel, OneShotMutableValue


class TestDynamicMHA(TestCase):

    def test_init(self):
        mutable_num_heads = OneShotMutableValue(
            value_list=[2, 4, 8], default_value=8)
        mutable_embed_dims = OneShotMutableChannel(
            num_channels=128,
            candidate_choices=[32, 64, 128],
            candidate_mode='number')

        mutable_head_dims = mutable_embed_dims // mutable_num_heads

        m = DynamicMultiheadAttention(embed_dims=128, num_heads=8)

        m.register_mutable_attr('num_heads', mutable_num_heads)
        m.register_mutable_attr('embed_dims', mutable_embed_dims)

        m.rel_pos_embed_k.register_mutable_attr('head_dims', mutable_head_dims)
        m.rel_pos_embed_v.register_mutable_attr('head_dims', mutable_head_dims)

        self.assertEqual(m.get_mutable_attr('num_heads').current_choice, 8)
        self.assertEqual(m.get_mutable_attr('embed_dims').current_choice, 128)

        x = torch.randn(8, 197, 624)


if __name__ == '__main__':
    unittest.main()
