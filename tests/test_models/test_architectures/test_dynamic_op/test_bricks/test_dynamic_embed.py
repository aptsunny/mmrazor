# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

from mmrazor.models.architectures.dynamic_op import DynamicPatchEmbed
from mmrazor.models.mutables import OneShotMutableChannel


class TestPatchEmbed(TestCase):

    def test_patch_embed(self):
        m = DynamicPatchEmbed(img_size=224, in_channels=3, embed_dims=100)

        mutable_embed_dims = OneShotMutableChannel(
            100, candidate_choices=[10, 50, 100], candidate_mode='number')

        mutable_embed_dims.current_choice = 50

        m.register_mutable_attr('embed_dims', mutable_embed_dims)
        c = m.get_mutable_attr('embed_dims').current_choice
        print(c)


if __name__ == '__main__':
    unittest.main()
