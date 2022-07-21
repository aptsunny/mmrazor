# Copyright (c) OpenMMLab. All rights reserved.
from .mutable_channel import (MutableChannel, OneShotMutableChannel,
                              SlimmableMutableChannel)
from .mutable_manage_mixin import MutableManageMixIn
from .mutable_module import (DiffChoiceRoute, DiffMutableModule, DiffMutableOP,
                             OneShotMutableModule, OneShotMutableOP)
from .mutable_value import MutableValue, OneShotMutableValue

__all__ = [
    'OneShotMutableOP', 'OneShotMutableModule', 'DiffMutableOP',
    'DiffChoiceRoute', 'DiffMutableModule', 'MutableManageMixIn',
    'OneShotMutableChannel', 'SlimmableMutableChannel', 'MutableChannel',
    'MutableValue', 'OneShotMutableValue'
]
