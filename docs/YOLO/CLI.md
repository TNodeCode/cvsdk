# YOLO CLI

## Classification

## Object Detection

Dataset structure:

```
dataset_root
├── images
│   ├── train
│       ├── img0001.jpg
│       ├── img0002.jpg
│       ├── ...
│   ├── val
│   └── test
├── labels
│   ├── train
│       ├── img0001.txt
│       ├── img0002.txt
│       ├── ...
│   ├── val
│   └── test
├── classes.txt
└── data.yaml
```

data.yaml:

```yaml
path: <absolute_path_to_dataset>
train: images/train
val: images/val
test: images/test
names:
  - class_1
  - class_2
  - ...
```

Command:

```bash
python ./cvsdk/cli.py yolo train --data-path ./datasets/my_dataset/data.yaml --model-name yolo26x.pt --epochs 50 --batch-size 128 --img-size 640
```