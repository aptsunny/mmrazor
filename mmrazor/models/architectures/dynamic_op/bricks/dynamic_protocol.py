# Copyright (c) OpenMMLab. All rights reserved.

import sys

if sys.version_info < (3, 8):
    from typing_extensions import Protocol
else:
    from typing import Protocol

import torch.nn as nn

from mmrazor.models.mutables.base_mutable import BaseMutable
from .utils import RelativePosition2D


class DynamicMHAProtocol(Protocol):

    @property
    def relative_position(self) -> bool:
        ...

    @property
    def max_relative_position(self) -> int:
        ...

    @property
    def rel_pos_embed_k(self) -> RelativePosition2D:
        ...

    @property
    def rel_pos_embed_v(self) -> RelativePosition2D:
        ...

    @property
    def w_qs(self) -> nn.Linear:
        ...

    @property
    def w_ks(self) -> nn.Linear:
        ...

    @property
    def w_vs(self) -> nn.Linear:
        ...

    @property
    def embed_dims(self) -> int:
        ...

    @property
    def proj(self) -> nn.Module:
        ...

    @property
    def attn_drop_rate(self) -> float:
        ...


class DynamicRPProtocol(Protocol):

    @property
    def mutable_attrs(self) -> nn.ModuleDict:
        ...

    @property
    def mutable_head_dims(self) -> BaseMutable:
        ...

    @property
    def head_dims(self) -> int:
        ...

    @property
    def max_relative_position(self) -> int:
        ...

    @property
    def embeddings_table_v(self) -> nn.Parameter:
        ...

    @property
    def embeddings_table_h(self) -> nn.Parameter:
        ...
