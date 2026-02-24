from pydantic import BaseModel
from typing import Literal


class ResNetBackboneConfig(BaseModel):
    type: Literal["ResNet"]
    checkpoint: str
    depth: Literal[18, 34, 50, 101, 152]
    frozen_stages: Literal[0, 1, 2, 3, 4]
    out_indices: list[int]


class VisionTransformerBackboneConfig(BaseModel):
    type: Literal["VisionTransformer"]
    checkpoint: str
    arch: str
    img_size: int
    patch_size: int


class SwinTransformerBackboneConfig(BaseModel):
    type: Literal["SwinTransformer"]
    checkpoint: str
    arch: str
    img_size: int


class TrainingConfig(BaseModel):
    """Model configuration for MMPretrain
    """
    config_path: str
    model_type: str
    model_name: str
    backbone: ResNetBackboneConfig | VisionTransformerBackboneConfig | SwinTransformerBackboneConfig | None
    dataset_dir: str
    train_dir: str
    val_dir: str
    test_dir: str
    dataset_classes: list[str]
    batch_size: int
    epochs: int
    work_dir: str
    annotations_train: str
    annotations_val: str
    annotations_test: str
    optimizer: str
    lr: float
    weight_decay: float
    momentum: float
    augmentations: list[dict]
