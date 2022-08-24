# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
from mmengine import BaseDataElement
from mmengine.model import BaseModel
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.mutators import OneShotModuleMutator
from mmrazor.registry import MODELS
from mmrazor.utils import SingleMutatorRandomSubnet, ValidFixMutable
from ..base import BaseAlgorithm, LossResults


@MODELS.register_module()
class Autoformer(BaseAlgorithm):
    """Implementation of `Autoformer <https://arxiv.org/abs/1904.00420>`."""

    # TODO fix ea's name in doc-string.

    def __init__(self,
                 architecture: Union[BaseModel, Dict],
                 mutator: Optional[Union[OneShotModuleMutator, Dict]] = None,
                 fix_subnet: Optional[ValidFixMutable] = None,
                 norm_training: bool = False,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(architecture, data_preprocessor, init_cfg)

        # SPOS has two training mode: supernet training and subnet retraining.
        # fix_subnet is not None, means subnet retraining.
        if fix_subnet:
            # Avoid circular import
            from mmrazor.structures import load_fix_subnet

            # According to fix_subnet, delete the unchosen part of supernet
            load_fix_subnet(self.architecture, fix_subnet)
            self.is_supernet = False
        else:
            assert mutator is not None, \
                'mutator cannot be None when fix_subnet is None.'
            if isinstance(mutator, OneShotModuleMutator):
                self.mutator = mutator
            elif isinstance(mutator, dict):
                self.mutator = MODELS.build(mutator)
            else:
                raise TypeError('mutator should be a `dict` or '
                                f'`OneShotModuleMutator` instance, but got '
                                f'{type(mutator)}')

            # Mutator is an essential component of the NAS algorithm. It
            # provides some APIs commonly used by NAS.
            # Before using it, you must do some preparations according to
            # the supernet.
            self.mutator.prepare_from_supernet(self.architecture)
            self.is_supernet = True

        self.norm_training = norm_training

    def sample_subnet(self) -> SingleMutatorRandomSubnet:
        """Random sample subnet by mutator."""
        return self.mutator.sample_choices()

    def set_subnet(self, subnet: SingleMutatorRandomSubnet):
        """Set the subnet sampled by :meth:sample_subnet."""
        self.mutator.set_choices(subnet)

    def loss(
        self,
        batch_inputs: torch.Tensor,
        data_samples: Optional[List[BaseDataElement]] = None,
    ) -> LossResults:
        """Calculate losses from a batch of inputs and data samples."""
        if self.is_supernet:
            random_subnet = self.sample_subnet()
            self.set_subnet(random_subnet)
            return self.architecture(batch_inputs, data_samples, mode='loss')
        else:
            return self.architecture(batch_inputs, data_samples, mode='loss')

    def train(self, mode=True):
        """Convert the model into eval mode while keep normalization layer
        unfreezed."""

        super().train(mode)
        if self.norm_training and not mode:
            for module in self.architecture.modules():
                if isinstance(module, _BatchNorm):
                    module.training = True
