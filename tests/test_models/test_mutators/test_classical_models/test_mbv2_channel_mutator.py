# Copyright (c) OpenMMLab. All rights reserved.
import os
import unittest
from os.path import dirname

import torch
from mmcls.models import *  # noqa: F401,F403
from mmcls.structures import ClsDataSample

from mmrazor import digit_version
from mmrazor.models.mutables import OneShotMutableChannel
from mmrazor.models.mutators import (OneShotChannelMutator,
                                     SlimmableChannelMutator)
from mmrazor.registry import MODELS
from ..utils import load_and_merge_channel_cfgs

MODEL_CFG = dict(
    type='mmcls.ImageClassifier',
    backbone=dict(type='MobileNetV2', widen_factor=1.5),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1920,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))

ONESHOT_MUTABLE_CFG = dict(
    type='OneShotMutableChannel',
    candidate_choices=[1 / 8, 2 / 8, 3 / 8, 4 / 8, 5 / 8, 6 / 8, 7 / 8, 1.0],
    candidate_mode='ratio')
ONESHOT_MUTABLE_CFGS = dict(
    in_features=ONESHOT_MUTABLE_CFG,
    out_features=ONESHOT_MUTABLE_CFG,
    in_channels=ONESHOT_MUTABLE_CFG,
    out_channels=ONESHOT_MUTABLE_CFG,
    num_features=ONESHOT_MUTABLE_CFG)

ONESHOT_MUTATOR_CFG = dict(
    type='OneShotChannelMutator',
    skip_prefixes=['head.fc'],
    tracer_cfg=dict(
        type='BackwardTracer',
        loss_calculator=dict(type='ImageClassifierPseudoLoss')),
    global_mutable_cfgs=ONESHOT_MUTABLE_CFGS)


@unittest.skipIf(
    digit_version(torch.__version__) == digit_version('1.8.1'),
    'PyTorch version 1.8.1 is not supported by the Backward Tracer.')
def test_oneshot_channel_mutator() -> None:
    imgs = torch.randn(16, 3, 224, 224)
    data_samples = [
        ClsDataSample().set_gt_label(torch.randint(0, 1000, (16, )))
    ]

    model = MODELS.build(MODEL_CFG)
    mutator: OneShotChannelMutator = MODELS.build(ONESHOT_MUTATOR_CFG)

    mutator.prepare_from_supernet(model)
    assert hasattr(mutator, 'name2module')

    # test set_min_choices
    mutator.set_min_choices()
    for mutables in mutator.search_groups.values():
        for mutable in mutables:
            # 1 / 8 is the minimum candidate ratio
            assert mutable.current_choice == round(1 / 8 *
                                                   mutable.num_channels)

    # test set_max_channel
    mutator.set_max_choices()
    for mutables in mutator.search_groups.values():
        for mutable in mutables:
            # 1.0 is the maximum candidate ratio
            assert mutable.current_choice == round(1. * mutable.num_channels)

    # test making groups logic
    choice_dict = mutator.sample_choices()
    assert isinstance(choice_dict, dict)
    mutator.set_choices(choice_dict)
    model(imgs, data_samples=data_samples, mode='loss')


def test_slimmable_channel_mutator() -> None:
    imgs = torch.randn(16, 3, 224, 224)
    data_samples = [
        ClsDataSample().set_gt_label(torch.randint(0, 1000, (16, )))
    ]

    root_path = dirname(dirname(dirname(dirname(__file__))))
    channel_cfg_paths = [
        os.path.join(root_path, 'data/MBV2_320M.yaml'),
        os.path.join(root_path, 'data/MBV2_220M.yaml'),
        os.path.join(root_path, 'data/MBV2_530M.yaml')
    ]

    mutator = SlimmableChannelMutator(
        global_mutable_cfgs=dict(
            in_features=dict(type='OneShotMutableChannel'),
            out_features=dict(type='OneShotMutableChannel'),
            in_channels=dict(type='OneShotMutableChannel'),
            out_channels=dict(type='OneShotMutableChannel'),
            num_features=dict(type='OneShotMutableChannel')),
        channel_cfgs=load_and_merge_channel_cfgs(channel_cfg_paths),
        tracer_cfg=dict(
            type='BackwardTracer',
            loss_calculator=dict(type='ImageClassifierPseudoLoss')))

    model = MODELS.build(MODEL_CFG)
    mutator.prepare_from_supernet(model)
    mutator.switch_choices(0)
    for name, module in model.named_modules():
        if isinstance(module, OneShotMutableChannel):
            if len(module.concat_parent_mutables) == 0:
                assert module.current_choice == module.choices[0]
    model(imgs, data_samples=data_samples, mode='loss')

    mutator.switch_choices(1)
    for name, module in model.named_modules():
        if isinstance(module, OneShotMutableChannel):
            if len(module.concat_parent_mutables) == 0:
                assert module.current_choice == module.choices[1]
    model(imgs, data_samples=data_samples, mode='loss')
