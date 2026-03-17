# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, MultiConfig, OptConfigType


class SpatialAttention(nn.Module):
    """Spatial Attention Module for feature refinement.

    This module applies spatial attention to highlight important spatial
    locations in feature maps. It computes channel-wise average and max
    pooling, concatenates them, applies a convolution, and uses sigmoid
    to generate attention weights.

    Args:
        in_channels (int): Number of input channels.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Defaults to None.
        act_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            activation layer. Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = None,
    ) -> None:
        super().__init__()
        # The concatenated avg_pool and max_pool always have 2 channels
        self.spatial_attention = nn.Sequential(
            ConvModule(
                2,  # Always 2 channels (avg_pool + max_pool)
                in_channels // 8,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                in_channels // 8,
                1,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))

    def forward(self, x: Tensor) -> Tensor:
        """Forward function.

        Args:
            x (Tensor): Input feature map of shape (B, C, H, W).

        Returns:
            Tensor: Refined feature map after applying spatial attention.
        """
        # Compute average and max pooling along channel dimension
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate along channel dimension
        concat = torch.cat([avg_pool, max_pool], dim=1)
        
        # Generate attention map
        attention = self.spatial_attention(concat)
        attention = torch.sigmoid(attention)
        
        # Apply attention
        return x * attention


class ChannelAttention(nn.Module):
    """Channel Attention Module for feature refinement.

    This module applies channel attention as described in the CBAM paper.
    It uses both average-pooled and max-pooled features to compute channel
    attention weights via a shared MLP.

    Args:
        in_channels (int): Number of input channels.
        reduction (int): Reduction ratio for the MLP. Defaults to 16.
    """

    def __init__(
        self,
        in_channels: int,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False))

    def forward(self, x: Tensor) -> Tensor:
        """Forward function.

        Args:
            x (Tensor): Input feature map of shape (B, C, H, W).

        Returns:
            Tensor: Refined feature map after applying channel attention.
        """
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        
        # Combine average and max pooling results
        attention = torch.sigmoid(avg_out + max_out)
        
        return x * attention


class CBAM(nn.Module):
    """CBAM: Convolutional Block Attention Module.

    This module combines channel attention and spatial attention in sequence.
    As described in the CBAM paper, channel attention is applied first,
    followed by spatial attention.

    Args:
        in_channels (int): Number of input channels.
        reduction (int): Reduction ratio for channel attention MLP.
            Defaults to 16.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Defaults to None.
        act_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            activation layer. Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        reduction: int = 16,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = None,
    ) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(
            in_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function.

        Args:
            x (Tensor): Input feature map of shape (B, C, H, W).

        Returns:
            Tensor: Refined feature map after applying CBAM attention.
        """
        # Apply channel attention first, then spatial attention
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


@MODELS.register_module()
class AttentionFPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Defaults to 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Defaults to -1, which means the
            last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Defaults to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Defaults to False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Defaults to False.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Defaults to None.
        act_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            activation layer in ConvModule. Defaults to None.
        upsample_cfg (:obj:`ConfigDict` or dict, optional): Config dict
            for interpolate layer. Defaults to dict(mode='nearest').
        attention_type (str): Type of attention module to use. Options are:
            - 'none': No attention (default).
            - 'spatial': Spatial attention only (as in CBAM).
            - 'channel': Channel attention only (as in CBAM).
            - 'cbam': Both channel and spatial attention (CBAM).
            Defaults to 'none'.
        attention_reduction (int): Reduction ratio for channel attention
            MLP. Only used when attention_type is 'channel' or 'cbam'.
            Defaults to 16.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        start_level: int = 0,
        end_level: int = -1,
        add_extra_convs: Union[bool, str] = False,
        relu_before_extra_convs: bool = False,
        no_norm_on_lateral: bool = False,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = None,
        upsample_cfg: ConfigType = dict(mode='nearest'),
        attention_type: str = 'none',
        attention_reduction: int = 16,
        init_cfg: MultiConfig = dict(
            type='Xavier', layer='Conv2d', distribution='uniform')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        
        # Attention type: 'none', 'spatial', 'channel', or 'cbam'
        assert attention_type in ('none', 'spatial', 'channel', 'cbam'), \
            f"attention_type must be 'none', 'spatial', 'channel', or 'cbam', got {attention_type}"
        self.attention_type = attention_type
        self.attention_reduction = attention_reduction

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.attention_modules = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

            # Add attention module for each stage based on attention_type
            if self.attention_type == 'spatial':
                self.attention_modules.append(
                    SpatialAttention(
                        out_channels,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
            elif self.attention_type == 'channel':
                self.attention_modules.append(
                    ChannelAttention(out_channels, self.attention_reduction))
            elif self.attention_type == 'cbam':
                self.attention_modules.append(
                    CBAM(
                        out_channels,
                        reduction=self.attention_reduction,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        """Forward function.

        Args:
            inputs (tuple[Tensor]): Features from the upstream network, each
                is a 4D-tensor.

        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # Apply attention to each stage before upscaling and addition
            if self.attention_type != 'none':
                # Apply attention to the current level before upscaling
                laterals[i] = self.attention_modules[i](laterals[i])
                # Apply attention to the previous level before addition
                laterals[i - 1] = self.attention_modules[i - 1](laterals[i - 1])

            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
