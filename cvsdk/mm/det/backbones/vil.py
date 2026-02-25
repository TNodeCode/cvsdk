import torch
import torch.nn as nn
from mmdet.registry import MODELS
from cvsdk.mmdet.vision_lstm.vision_lstm2 import VisionLSTM2
import einops


@MODELS.register_module()
class ViL(nn.Module):
    """Vision xLSTM (ViL) backbone."""
    def __init__(self, dim=192, input_shape=(3, 512, 512), patch_size=16, depth=6, drop_path_rate=0.0) -> None:
        """Initialize ViL backbone."""
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.patch_size = patch_size
        self.input_shape = input_shape
        self.drop_path_rate = drop_path_rate
        self.vil = VisionLSTM2(
            dim=dim,  # latent dimension (192 for ViL-T)
            depth=depth,  # how many ViL blocks (1 block consists 2 subblocks of a forward and backward block)
            patch_size=patch_size,  # patch_size (results in 64 patches for 32x32 images)
            input_shape=input_shape,  # RGB images
            drop_path_rate=drop_path_rate,  # stochastic depth parameter (disabled for ViL-T)
            mode="features" # "features" or "classifier"
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward image through backbone.

        Args:
            x (torch.Tensor): Input image tensor

        Returns:
            list[torch.Tensor]: list containing tensor of shape (C, H, W) for each input image
        """
        B, C, H, W = x.shape
        return einops.rearrange(self.vil(x), "b (h w) d -> b d h w", h=H // self.patch_size, w=W // self.patch_size)