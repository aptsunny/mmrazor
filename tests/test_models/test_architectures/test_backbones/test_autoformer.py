# Copyright (c) OpenMMLab. All rights reserved.
import copy
import sys
import unittest
from unittest import TestCase

import pytest
import torch
from mmcls.models import *  # noqa: F401,F403
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models import *  # noqa: F401,F403
from mmrazor.models.mutables import *  # noqa: F401,F403
from mmrazor.registry import MODELS


class TestAutoformer(TestCase):

    def test_init(self) -> None:
        m = Autoformer()
        i = torch.randn(8, 3, 224, 224)
        o = m(i)


if __name__ == '__main__':
    unittest.main()
