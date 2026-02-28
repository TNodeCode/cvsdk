import torch
import torch.nn as nn
from mmdet.registry import MODELS
import transformers
from transformers import DINOv3ViTConfig, DINOv3ViTModel


@MODELS.register_module()
class DINOv3ViTBackbone(nn.Module):

    def __init__(self, finetuning=False, *args, **kwargs):
        super(DINOv3ViTBackbone, self).__init__()
        self.finetuning = finetuning
        self.config = DINOv3ViTConfig()
        self.model = DINOv3ViTModel(self.config)
        if not self.finetuning:
            self._freeze()


    def _freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False


    def forward(self, x):  # should return a tuple
        """Forward image through model

        Args:
            x (torch.Tensor): image of shape (B, C, H, W)

        Returns:
            torch.tensor: Model output
        """
        z = self.model(x)
        # remove the [CLS] token
        z = z.last_hidden_state[:, 1:, :]
        # batch_size, num_patches, hidden_size
        B, P, D = z.shape
        h = w = int(P ** 0.5)
        z = z.permute(0, 2, 1)
        z = z.reshape(B, D, h, w)
        return z

