import torch
import torch.nn as nn
from mmdet.registry import MODELS
import transformers
from transformers import DINOv3ViTConfig, DINOv3ViTModel


@MODELS.register_module()
class DINOv3ViTBackbone(nn.Module):

    def __init__(self, *args, **kwargs):
        super(DINOv3ViTBackbone, self).__init__()
        config = DINOv3ViTConfig()
        self.model = DINOv3ViTModel(config)

    def forward(self, x):  # should return a tuple
        B, C, H, W = x.shape
        # embeddings of shape (B, num_patches, hidden_size)
        z = self.model(x)
        return z

