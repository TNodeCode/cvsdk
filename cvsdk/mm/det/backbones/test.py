import torch
import torch.nn as nn
from mmdet.registry import MODELS


@MODELS.register_module()
class MyBackbone(nn.Module):

    def __init__(self, *args, **kwargs):
        super(MyBackbone, self).__init__()
        self.layer1 = nn.Identity()

    def forward(self, x):  # should return a tuple
        B, C, H, W = x.shape
        return (
          torch.rand((B,4,H//2,W//2)).to(x.device),
          torch.rand((B,8,H//2,W//2)).to(x.device),
          torch.rand((B,16,H//2,W//2)).to(x.device)
        )

