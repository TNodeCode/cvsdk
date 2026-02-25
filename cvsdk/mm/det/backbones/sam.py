import torch
import torch.nn as nn
from mmdet.registry import MODELS
from ultralytics.models.sam.model import SAM
from typing import Literal


@MODELS.register_module()
class SAM21ImageEncoder(nn.Module):
    """SAM 2.1 Image Encoder backbone."""

    # this attribute saves which layers belong to which layer for a SAM model. Notation:  inner array:[first_layer, last_layer], outer array: [stage 1, stage 2, stage 3, stage 4]
    stage_layers = {
        "t": [[0,0], [1,2], [3,9], [10,11]],
        "s": [[0,0], [1,2], [3,13], [14,15]],
        "b": [[0,1], [2,4], [5,20], [21,23]],
        "l": [[0,1], [2,7], [8,43], [44,47]]
    }

    def __init__(self, size: Literal["t", "s", "b", "l"] = "t", frozen_stages: int = 0) -> None:
        """Initialize SAM 2.1 Image Encoder backbone.

        Args:
            size (Literal[&quot;t&quot;, &quot;s&quot;, &quot;b&quot;, &quot;l&quot;], optional): SAM size. Defaults to "t".
            frozen_stages (int, optional): Number of frozen stages (between 0 and 4). Defaults to 0.

        SAM output shapes:
            SAM 2.1 t output shapes
            [torch.Size([B, 96, H/4, W/4]),     # block 0
            torch.Size([B, 192, H/8, W/8]),     # block 1-2
            torch.Size([B, 384, H/16, W/16]),   # block 3-9
            torch.Size([B, 768, H/32, W/32])]   # block 10-11

            SAM 2.1 s output shapes
            [torch.Size([B, 96, H/4, W/4]),     # block 0
            torch.Size([B, 192, H/8, W/8]),     # block 1-2
            torch.Size([B, 384, H/16, W/16]),   # block 3-13
            torch.Size([B, 768, H/32, W/32])]   # block 14-15

            SAM 2.1 b output shapes
            [torch.Size([B, 112, H/4, W/4]),    # block 0-1
            torch.Size([B, 224, H/8, W/8]),     # block 2-4
            torch.Size([B, 448, H/16, W/16]),   # block 5-20
            torch.Size([B, 896, H/32, W/32])]   # block 21-23
            ```

            SAM 2.1 l output shapes
            [torch.Size([B, 144, H/4, W/4]),    # block 0-1
            torch.Size([B, 288, H/8, W/8]),     # block 2-7
            torch.Size([B, 576, H/16, W/16]),   # block 8-43
            torch.Size([B, 1152, H/32, W/32])]  # block 44-47
        """
        super().__init__()
        self.encoder = SAM(f"sam2.1_{size}.pt").model.image_encoder.trunk

        # Validate frozen_stages parameter.
        if not (0 <= frozen_stages <= 4):
            raise ValueError("frozen_stages must be between 0 and 4.")

        # freeze patch embedding
        if frozen_stages > 0:
            for param in self.encoder.patch_embed.parameters():
                param.requires_grad = False

        # freeze layers
        for i in range(frozen_stages):
            for l in range(SAM21ImageEncoder.stage_layers[size][i][0], SAM21ImageEncoder.stage_layers[size][i][1] + 1):
                for param in self.encoder.blocks[l].parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward image through backbone.

        Args:
            x (torch.Tensor): Input image tensor

        Returns:
            list[torch.Tensor]: list containing tensor of shape (C, H, W) for each input image
        """
        return self.encoder(x)