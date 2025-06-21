from pathlib import Path

import pytorch_lightning as pl
import torch
import torchvision.datasets as dsets
import torchvision.transforms as T
from torch.utils.data import DataLoader


# ----------------------------------------------------------------------------- #
#                              Multi-crop pipeline                               #
# ----------------------------------------------------------------------------- #
def multicrop_collate(batch):
    """Input  : batch = [(crops_0, _), (crops_1, _), …]  where crops_k is
                     a list like [global1, global2, local1, local2, …]

    Output : [ Tensor shape (B, C, H_g, W_g),   # global crop 1
               Tensor shape (B, C, H_g, W_g),   # global crop 2
               Tensor shape (B, C, H_l, W_l),   # local crop 1
               … ]
    """
    crops_lists, _ = zip(*batch)          # tuple of per-sample lists
    n_crops = len(crops_lists[0])         # all samples have the same count

    stacked = [
        torch.stack([sample_crops[i] for sample_crops in crops_lists], dim=0)
        for i in range(n_crops)
    ]
    return stacked


class MultiCropTransform:
    """Return 2 global crops + N local crops, as in the DINO paper."""

    def __init__(
        self,
        global_size: int = 224,
        local_size: int = 96,
        local_crops: int = 8,
    ):
        # Augmentations follow the original repo (simplified)
        flip_and_color = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
            ]
        )
        normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

        self.global_transform = T.Compose(
            [
                T.RandomResizedCrop(global_size, scale=(0.25, 1.0)),
                flip_and_color,
                T.GaussianBlur(23, sigma=(0.1, 2.0)),
                normalize,
            ]
        )
        self.local_transform = T.Compose(
            [
                T.RandomResizedCrop(local_size, scale=(0.05, 0.25)),
                flip_and_color,
                #T.GaussianBlur(7, sigma=(0.1, 2.0)),
                normalize,
                T.Resize(224)
            ]
        )
        self.local_crops = local_crops

    def __call__(self, img):
        crops: list[torch.Tensor] = []
        crops.append(self.global_transform(img))
        crops.append(self.global_transform(img))
        for _ in range(self.local_crops):
            crops.append(self.local_transform(img))
        return crops


# ----------------------------------------------------------------------------- #
#                             LightningDataModule                               #
# ----------------------------------------------------------------------------- #
class ImageNetMultiCropDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int = 64,
        workers: int = 8,
        local_crops: int = 8,
    ):
        super().__init__()
        self.root = Path(root)
        self.batch_size = batch_size
        self.workers = workers
        self.train_transform = MultiCropTransform(local_crops=local_crops)

    def setup(self, stage=None):
        self.train_set = dsets.ImageFolder(self.root, transform=self.train_transform)

    def train_dataloader(self):
        # Each item is *already* a list of crops; collate just leaves it untouched
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=multicrop_collate,
        )
