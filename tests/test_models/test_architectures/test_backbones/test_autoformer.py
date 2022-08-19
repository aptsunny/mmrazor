# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
from mmcls.models import *  # noqa: F401,F403

from mmrazor.models.architectures import Autoformer


class TestAutoformer(TestCase):

    def test_init(self) -> None:
        m = Autoformer()
        i = torch.randn(8, 3, 224, 224)
        o = m(i)

        assert o is not None


if __name__ == '__main__':
    unittest.main()
