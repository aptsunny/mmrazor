# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple, Union

from mmrazor.registry import TASK_UTILS
from .base_estimator import BaseEstimator
from .zeroshotproxy import (compute_gradnorm_score, compute_naswot_score,
                            compute_NTK_score, compute_syncflow_score,
                            compute_zen_score)


@TASK_UTILS.register_module()
class TrainFreeEstimator(BaseEstimator):
    """Zero Shot Accuracy Function."""

    def __init__(self, metric='Zen', batch_size=64):
        super().__init__()
        self.metric = metric
        self.batch_size = batch_size

    def estimate(self, model, **kwargs):
        if self.metric == 'Zen':
            the_nas_core = compute_zen_score(model, self.batch_size, **kwargs)
        elif self.metric == 'TE-NAS':
            the_nas_core = compute_NTK_score(model, self.batch_size, **kwargs)
        elif self.metric == 'xd':
            the_nas_core = compute_syncflow_score(model, self.batch_size,
                                                  **kwargs)
        elif self.metric == 'GradNorm':
            the_nas_core = compute_gradnorm_score(model, self.batch_size,
                                                  **kwargs)
        elif self.metric == 'NASWOT':
            the_nas_core = compute_naswot_score(model, self.batch_size,
                                                **kwargs)
        else:
            raise NotImplementedError(
                f'compute {self.metric} score not supported.')
        return the_nas_core


"""
import torch
class TrainFreeEstimator_(BaseEstimator):

    def __init__(
        self,
        input_shape: Tuple = (1, 3, 224, 224),
        units: Dict = dict(flops='M', params='M', latency='ms'),
        as_strings: bool = False,
        flops_params_cfg: Optional[dict] = None,
        latency_cfg: Optional[dict] = None,
    ):
        super().__init__(input_shape, units, as_strings)
        if not isinstance(units, dict):
            raise TypeError('units for estimator should be a dict',
                            f'but got `{type(units)}`')
        for unit_key in units:
            if unit_key not in ['flops', 'params', 'latency']:
                raise KeyError(f'Got invalid key `{unit_key}` in units. ',
                               'Should be `flops`, `params` or `latency`.')
        if flops_params_cfg:
            self.flops_params_cfg = flops_params_cfg
        else:
            self.flops_params_cfg = dict()
        self.latency_cfg = latency_cfg if latency_cfg else dict()

    def estimate(self,
                 model: torch.nn.Module,
                 flops_params_cfg: dict = None,
                 latency_cfg: dict = None) -> Dict[str, Union[float, str]]:
        resource_metrics = dict()
        measure_latency = True if latency_cfg else False

        if flops_params_cfg:
            flops_params_cfg = {**self.flops_params_cfg, **flops_params_cfg}
            self._check_flops_params_cfg(flops_params_cfg)
            flops_params_cfg = self._set_default_resource_params(
                flops_params_cfg)
        else:
            flops_params_cfg = self.flops_params_cfg

        if latency_cfg:
            latency_cfg = {**self.latency_cfg, **latency_cfg}
            self._check_latency_cfg(latency_cfg)
            latency_cfg = self._set_default_resource_params(latency_cfg)
        else:
            latency_cfg = self.latency_cfg

        model.eval()
        flops, params = get_model_flops_params(model, **flops_params_cfg)
        if measure_latency:
            latency = get_model_latency(model, **latency_cfg)
        else:
            latency = '0.0 ms' if self.as_strings else 0.0  # type: ignore

        resource_metrics.update({
            'flops': flops,
            'params': params,
            'latency': latency
        })
        return resource_metrics

    def estimate_separation_modules(
            self,
            model: torch.nn.Module,
            flops_params_cfg: dict = None) -> Dict[str, Union[float, str]]:

        if flops_params_cfg:
            flops_params_cfg = {**self.flops_params_cfg, **flops_params_cfg}
            self._check_flops_params_cfg(flops_params_cfg)
            flops_params_cfg = self._set_default_resource_params(
                flops_params_cfg)
        else:
            flops_params_cfg = self.flops_params_cfg
        flops_params_cfg['seperate_return'] = True

        assert len(flops_params_cfg['spec_modules']), (
            'spec_modules can not be empty when calling '
            f'`estimate_separation_modules` of {self.__class__.__name__} ')

        model.eval()
        spec_modules_resources = get_model_flops_params(
            model, **flops_params_cfg)
        return spec_modules_resources

    def _check_flops_params_cfg(self, flops_params_cfg: dict) -> None:

        for key in flops_params_cfg:
            if key not in get_model_flops_params.__code__.co_varnames[
                    1:]:  # type: ignore
                raise KeyError(f'Got invalid key `{key}` in flops_params_cfg.')

    def _check_latency_cfg(self, latency_cfg: dict) -> None:

        for key in latency_cfg:
            if key not in get_model_latency.__code__.co_varnames[
                    1:]:  # type: ignore
                raise KeyError(f'Got invalid key `{key}` in latency_cfg.')

    def _set_default_resource_params(self, cfg: dict) -> dict:

        default_common_settings = ['input_shape', 'units', 'as_strings']
        for key in default_common_settings:
            if key not in cfg:
                cfg[key] = getattr(self, key)
        return cfg
"""
