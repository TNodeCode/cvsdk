# MMDetection

This repository provides an interface for training object detection models provided by the MMDetection library.

## Create a configuration file

Create a configuration file `./configs/faster_rcnn_50.yaml` with the following content.

```yaml
# model
model_type: faster_rcnn
model_name: faster-rcnn_r50_fpn_1x_coco

# dataset
dataset_dir: data/coco/
train_dir: train2017
val_dir: val2017
test_dir: test2017
dataset_classes:
  - cat
  - dog
annotations_train: instances_train2017.json
annotations_val: instances_val2017.json
annotations_test: instances_test2017.json

# preprocessing
augmentations:
  - type: RandomFlip
    prob: 0.5
    direction: horizontal
  - type: RandomFlip
    prob: 0.5
    direction: vertical
  - type: RandomFlip
    prob: 0.5
    direction: diagonal
  - type: RandomAffine
    max_rotate_degree: 10.0
    max_translate_ratio: 0.1
    max_shear_degree: 5.0
    scaling_ratio_range: [0.5, 1.5]

# training
batch_size: 2
epochs: 36
work_dir: work_dirs
optimizer: adamw
lr: 1e-4
weight_decay: 0.05
momentum: 0.9
```

`model_name`must be the file name of one of the configuration files found under `repos/onedl-mmdetection/configs`. You can find some of the available models in the table below.

## Available models

| model_type      | model_name                                         |
| --------------- | -------------------------------------------------- |
| faster_rcnn     | faster-rcnn_r50_fpn_1x_coco                        |
|                 | faster-rcnn_r101_fpn_1x_coco                       |
|                 | faster-rcnn_x101-32x4d_fpn_1x_coco                 |
|                 | faster-rcnn_x101-64x4d_fpn_1x_coco                 |
| cascade_rcnn    | cascade-rcnn_r50_fpn_1x_coco                       |
|                 | cascade-rcnn_r101_fpn_1x_coco                      |
|                 | cascade-rcnn_x101-32x4d_fpn_1x_coco                |
|                 | cascade-rcnn_x101-64x4d_fpn_1x_coco                |
| deformable_detr | deformable-detr_r50_16xb2-50e_coco                 |
|                 | deformable-detr-refine_r50_16xb2-50e_coco          |
|                 | deformable-detr-refine-twostage_r50_16xb2-50e_coco |
| yolox           | yolox_nano_8xb8-300e_coco                          |
|                 | yolox_tiny_8xb8-300e_coco                          |
|                 | yolox_s_8xb8-300e_coco                             |
|                 | yolox_m_8xb8-300e_coco                             |
|                 | yolox_l_8xb8-300e_coco                             |
|                 | yolox_x_8xb8-300e_coco                             |

## Train a model

```bash
$ cv mmdet train configs/faster_rcnn_50.yml
```
