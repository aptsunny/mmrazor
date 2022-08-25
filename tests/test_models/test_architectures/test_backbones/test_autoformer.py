# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch

from mmrazor.models.architectures.backbones import Autoformer
from mmrazor.models.mutators.channel_mutator import BigNASChannelMutator
from mmrazor.models.mutators.value_mutator import DynamicValueMutator


class TestAutoformer(TestCase):

    def test_init(self) -> None:
        m = Autoformer()
        i = torch.randn(8, 3, 224, 224)
        o = m(i)

        print(o.shape)

        assert o is not None

    def test_mutator(self):
        m = Autoformer()
        cm = BigNASChannelMutator()
        cm.prepare_from_supernet(m)
        print(cm.search_groups)
        print('=' * 10)
        vm = DynamicValueMutator()
        vm.prepare_from_supernet(m)
        print(vm.search_groups)


if __name__ == '__main__':
    unittest.main()
