import torch
import torch.nn as nn
from mmdet.registry import MODELS
from lightly_train._models.dinov2_vit.dinov2_vit_package import DINOv2ViTPackage


@MODELS.register_module()
class LightlyDINOv2ViTBackbone(nn.Module):

    def __init__(self, model_name: str = "vits14", checkpoint: str | None = None, finetuning=False, *args, **kwargs):
        super(LightlyDINOv2ViTBackbone, self).__init__()
        self.finetuning = finetuning
        self.model = DINOv2ViTPackage.get_model(model_name)
        if checkpoint:
            self.model.load_state_dict(torch.load(checkpoint, weights_only=True))
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
        # Get the patch embeddings (shape BATCH_SIZE, N_PATCHES, HIDDEN_DIM)
        z = self.model.forward_features(x)['x_norm_patchtokens']
        # batch_size, num_patches, hidden_size
        B, P, D = z.shape
        h = w = int(P ** 0.5)
        z = z.permute(0, 2, 1)
        z = z.reshape(B, D, h, w)
        return z

