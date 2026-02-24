from pydantic import BaseModel
from typing import Literal

class ResNetBackboneConfig(BaseModel):
    type: Literal["ResNet"]
    checkpoint: str
    depth: Literal[18,34,50,101,152]
    frozen_stages: Literal[0,1,2,3,4]
    out_indices: list[int]
    out_channels: list[int]


class EfficientNetBackboneConfig(BaseModel):
    type: Literal["EfficientNetV2"]
    checkpoint: str
    arch: Literal["s","m","l","xl"]
    out_indices: list[int]
    out_channels: list[int]


class SwinTransformerBackboneConfig(BaseModel):
    type: Literal["SwinTransformer"] = "SwinTransformer"
    checkpoint: str = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth"
    pretrain_img_size: int = 512
    num_heads: list[int] = [6, 12, 24, 48]
    out_indices: list[int] = [0, 1, 2, 3]
    attn_drop_rate: float = 0.1
    drop_path_rate: float = 0.3
    drop_rate: float = 0.1
    embed_dims: int = 192


class MyBackboneConfig(BaseModel):
    type: Literal["MyBackbone"]


class TrainingConfig(BaseModel):
    """Model configuration for semantic segmentation
    """
    config_path: str
    model_type: str
    model_name: str
    backbone: ResNetBackboneConfig | EfficientNetBackboneConfig | MyBackboneConfig | SwinTransformerBackboneConfig | None
    dataset_dir: str
    train_dir: str
    val_dir: str
    test_dir: str
    train_seg_dir: str
    val_seg_dir: str
    test_seg_dir: str
    dataset_classes: list[str]
    batch_size: int
    epochs: int
    work_dir: str
    optimizer: str
    lr: float
    weight_decay: float
    momentum: float
    augmentations: list[dict]
