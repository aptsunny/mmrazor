# Copyright (c) OpenMMLab. All rights reserved.
import copy
import unittest
from unittest import TestCase

from mmrazor.models.architectures.dynamic_op import (DynamicRelativePosition2D,
                                                     RelativePosition2D)
from mmrazor.models.mutables import OneShotMutableChannel, mutable_channel


class TestDynamicRP(TestCase):

    def test_dynamic_relative_position(self) -> None:
        mutable_head_dims = OneShotMutableChannel(
            8, candidate_choices=[2, 4, 6, 8], candidate_mode='number')

        dynamic_rp = DynamicRelativePosition2D(
            head_dims=8, max_relative_position=14)

        mutable_head_dims.current_choice = 6
        dynamic_rp.register_mutable_attr('head_dims', mutable_head_dims)

        assert dynamic_rp.mutable_head_dims.current_choice == 6

        embed = dynamic_rp.forward(14, 14)

        print(embed.shape)


if __name__ == '__main__':
    unittest.main()
