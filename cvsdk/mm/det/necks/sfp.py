import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch.nn import LayerNorm
from mmdet.registry import MODELS


@MODELS.register_module()
class SFP(BaseModule):
    r"""
    Our custom implementation of Simple Feature Pyramid (SFP).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 use_p2=False,
                 use_act_checkpoint=False,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(SFP, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.use_p2 = use_p2
        self.fp16_enabled = False
        if self.use_p2:
            self.p2 = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(self.in_channels[0], self.in_channels[0]//2, kernel_size=3, padding=1, bias=False),
                LayerNorm(self.in_channels[0]//2),
                nn.GELU(),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(self.in_channels[0]//2, self.in_channels[0]//4, kernel_size=3, padding=1, bias=False),
                LayerNorm(self.in_channels[0]//4),
                nn.GELU(),
                nn.Conv2d(self.in_channels[0]//4, self.out_channels, kernel_size=1, bias=False),
                LayerNorm(self.out_channels),
                nn.GELU(),
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
                LayerNorm(self.out_channels)
            )

        self.p3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels[0], self.in_channels[0]//2, kernel_size=3, padding=1, bias=False),
            LayerNorm(self.in_channels[0]//2),
            nn.GELU(),
            nn.Conv2d(self.in_channels[0]//2, self.out_channels, kernel_size=1, bias=False),
            LayerNorm(self.out_channels),
            nn.GELU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm(self.out_channels)
        )
        self.p4 = nn.Sequential(
            nn.Conv2d(self.in_channels[0], self.out_channels, kernel_size=1, bias=False),
            LayerNorm(self.out_channels),
            nn.GELU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm(self.out_channels)
        )
        self.p5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(self.in_channels[0], self.out_channels, kernel_size=1, bias=False),
            LayerNorm(self.out_channels),
            nn.GELU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm(self.out_channels)
        )
        self.p6 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(self.in_channels[0], self.in_channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            LayerNorm(self.in_channels[0]),
            nn.GELU(),
            nn.Conv2d(self.in_channels[0], self.out_channels, kernel_size=1, bias=False),
            LayerNorm(self.out_channels),
            nn.GELU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm(self.out_channels)
        )
        
        if use_act_checkpoint:
            self.p3 = self.p3
            self.p4 = self.p4
            self.p5 = self.p5
            self.p6 = self.p6
            if self.use_p2:
                self.p2 = self.p2

    def forward(self, inputs):
        """Forward function."""
        x = inputs[0]
        p4 = self.p4(x)
        p3 = self.p3(x)
        p5 = self.p5(x)
        p6 = self.p6(x)
        outs = [p3, p4, p5, p6]
        if self.use_p2:
            outs = [self.p2(x)] + outs
        return tuple(outs)