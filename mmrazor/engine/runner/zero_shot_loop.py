# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

from mmengine.runner import EpochBasedTrainLoop
from torch.utils.data import DataLoader

from mmrazor.registry import LOOPS, TASK_UTILS

import torch
import time
import numpy as np


@LOOPS.register_module()
class ZeroShotLoop(EpochBasedTrainLoop):
    """Loop for subnet validation in NAS with BN re-calibration.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 validation. Defaults to
            False.
        evaluate_fixed_subnet (bool): Whether to evaluate a fixed subnet only
            or not. Defaults to False.
        calibrate_sample_num (int): The number of images to compute the true
            average of per-batch mean/variance instead of the running average.
            Defaults to 4096.
        estimator_cfg (dict, Optional): Used for building a resource estimator.
            Defaults to dict(type='mmrazor.ResourceEstimator').
    """

    def __init__(
        self,
        runner,
        dataloader: Union[DataLoader, Dict],
        max_epochs: int = 480000,
        population_size: int = 512,
        input_image_size: Optional[int] = 224,
        estimator_cfg: Optional[Dict] = dict(type='mmrazor.TrainFreeEstimator')
    ) -> None:
        super().__init__(runner, dataloader, max_epochs)

        if self.runner.distributed:
            model = self.runner.model.module
        else:
            model = self.runner.model

        # self.gpu = 0
        self.device = 'cpu'
        if next(model.parameters()).is_cuda:
            self.device = 'cuda'
        else:
            raise NotImplementedError('To use cpu to test latency not supported.')

        self.input_image_size = input_image_size

        self.population_size = population_size
        self.model = model

        # initialize estimator
        estimator_cfg = dict() if estimator_cfg is None else estimator_cfg
        if 'type' not in estimator_cfg:
            estimator_cfg['type'] = 'mmrazor.ResourceEstimator'
        self.estimator = TASK_UTILS.build(estimator_cfg)

    def run(self) -> None:
        """Launch searching."""
        # self.runner.call_hook('before_train')

        # if self.predictor_cfg is not None:
        #     self._init_predictor()

        # if self.resume_from:
        #     self._resume()

        self.popu_structure_list = []
        self.popu_zero_shot_score_list = []
        self.popu_latency_list = []

        # while self._epoch < self._max_epochs:
        self.start_timer = time.time()
        for loop_count in range(self._max_epochs):
            # for loop_count in range(self.max_epochs):
            self.run_epoch(loop_count)
            # self._save_searcher_ckpt()

        # self._save_best_fix_subnet()

        # self.runner.call_hook('after_train')

    def run_epoch(self, loop_count) -> None:
        """Iterate one epoch.

        Steps:
            1. Sample some new candidates from the supernet. Then Append them
                to the candidates, Thus make its number equal to the specified
                number.
            2. Validate these candidates(step 1) and update their scores.
            3. Pick the top k candidates based on the scores(step 2), which
                will be used in mutation and crossover.
            4. Implement Mutation and crossover, generate better candidates.
        """

        while len(self.popu_structure_list) > self.population_size:
            min_zero_shot_score = min(self.popu_zero_shot_score_list)
            tmp_idx = self.popu_zero_shot_score_list.index(min_zero_shot_score)
            self.popu_zero_shot_score_list.pop(tmp_idx)
            self.popu_structure_list.pop(tmp_idx)
            self.popu_latency_list.pop(tmp_idx)
        pass

        if loop_count >= 1 and loop_count % 100 == 0:
            max_score = max(self.popu_zero_shot_score_list)
            min_score = min(self.popu_zero_shot_score_list)
            elasp_time = time.time() - self.start_timer
            self.runner.logger.info(
                f'loop_count={loop_count}/{self._max_epochs}, max_score={max_score:4g}, min_score={min_score:4g}, time={elasp_time/3600:4g}h')
            # self.runner.logger.info(f'loop_count={loop_count}/{self.max_epochs}, max_score={max_score:4g}, min_score={min_score:4g}, time={elasp_time/3600:4g}h')

        # ----- generate a random structure ----- #
        random_structure_str = self.sample_candidates()

        the_model = None

        # ----- filter structure by restricted condition ----- #
        # max_layers / budget_model_size / budget_flops / budget_latency=0.0001
        the_nas_core = self.compute_nas_score(random_structure_str, self.device)

        self.popu_structure_list.append(random_structure_str)
        self.popu_zero_shot_score_list.append(the_nas_core)
        self.popu_latency_list.append(np.inf)

        # self._epoch += 1

    def sample_candidates(self):
        # ----- generate a random[] structure ----- #
        candidate = self.model.mutator.sample_choices()
        return candidate

    def compute_nas_score(self, random_structure_str, device):
        # compute network zero-shot proxy score
        self.model.mutator.set_choices(random_structure_str)
        the_nas_core = self.estimator.estimate(model=self.model, device=device,
                                               resolution=self.input_image_size)

        # del the_model
        torch.cuda.empty_cache()
        return the_nas_core
