# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Dict, List, Tuple, Union

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from mmengine.evaluator import Evaluator
from mmengine.runner import TestLoop
from torch._utils import (_flatten_dense_tensors, _take_tensors,
                          _unflatten_dense_tensors)
from torch.utils.data import DataLoader

from mmrazor.models.utils import add_prefix
from mmrazor.registry import LOOPS
from .mixins import CalibrateBNMixin


def _allreduce_coalesced(tensors: torch.Tensor,
                         world_size: int,
                         bucket_size_mb: int = -1) -> None:
    if bucket_size_mb > 0:
        bucket_size_bytes = bucket_size_mb * 1024 * 1024
        buckets = _take_tensors(tensors, bucket_size_bytes)
    else:
        buckets = OrderedDict()
        for tensor in tensors:
            tp = tensor.type()
            if tp not in buckets:
                buckets[tp] = []
            buckets[tp].append(tensor)
        buckets = buckets.values()

    for bucket in buckets:
        flat_tensors = _flatten_dense_tensors(bucket)
        dist.all_reduce(flat_tensors)
        flat_tensors.div_(world_size)
        for tensor, synced in zip(
                bucket, _unflatten_dense_tensors(flat_tensors, bucket)):
            tensor.copy_(synced)

def get_dist_info() -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def allreduce_params(params: List[torch.nn.Parameter],
                     coalesce: bool = True,
                     bucket_size_mb: int = -1) -> None:
    """Allreduce parameters.
    Args:
        params (list[torch.nn.Parameter]): List of parameters or buffers
            of a model.
        coalesce (bool, optional): Whether allreduce parameters as a whole.
            Defaults to True.
        bucket_size_mb (int, optional): Size of bucket, the unit is MB.
            Defaults to -1.
    """
    _, world_size = get_dist_info()
    if world_size == 1:
        return
    params = [param.data for param in params]
    if coalesce:
        _allreduce_coalesced(params, world_size, bucket_size_mb)
    else:
        for tensor in params:
            dist.all_reduce(tensor.div_(world_size))



@LOOPS.register_module()
class AutoSlimTestLoop(TestLoop, CalibrateBNMixin):

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False,
                 calibrated_sample_nums: int = 4096) -> None:
        super().__init__(runner, dataloader, evaluator, fp16)

        if self.runner.distributed:
            model = self.runner.model.module
        else:
            model = self.runner.model

        # just for convenience
        self._model = model
        self.calibrated_sample_nums = calibrated_sample_nums

    def run(self) -> None:
        """Launch validation."""
        self.runner.call_hook('before_test')

        all_metrics = dict()

        self._model.set_max_subnet()
        
        self.calibrate_bn_statistics(self.runner.train_dataloader,
                                     self.calibrated_sample_nums)
        
        metrics = self._evaluate_once()
        all_metrics.update(add_prefix(metrics, 'max_subnet'))

        self._model.set_min_subnet()
        self.calibrate_bn_statistics(self.runner.train_dataloader,
                                     self.calibrated_sample_nums)
        metrics = self._evaluate_once()
        all_metrics.update(add_prefix(metrics, 'min_subnet'))

        self.runner.call_hook('after_test_epoch', metrics=all_metrics)
        self.runner.call_hook('after_test')

    def _evaluate_once(self) -> Dict:
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        return self.evaluator.evaluate(len(self.dataloader.dataset))
