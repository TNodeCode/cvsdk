# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch.nn as nn
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import OptConfigType, OptMultiConfig


@MODELS.register_module()
class IdentityMapper(BaseModule):
    """Identity mapper that returns the inputs it gets.

    This module is used to convert the output of a backbone to the input of a
    detection head. This module mimics the behavior of the original
    `ChannelMapper` class in MMDetection, but it simply returns the original inputs
    as it expects to be used with the ViT backbone which outputs a tuple of
    outputs from different stages which all have the same feature dimension and spatial dimensions.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        """Forward function."""
        return inputs
