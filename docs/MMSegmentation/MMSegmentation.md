# MMSegmentation CLI

The MMSegmentation CLI provides a command-line interface for training, evaluating, and running inference with semantic segmentation models using the MMSegmentation framework.

## Available Commands

### 1. Train

Train a segmentation model based on a configuration file.

```bash
cv mmseg train <config-file> [load-from]
```

**Arguments:**
- `config-file`: Path to the training configuration file (YAML format)
- `load-from` (optional): Path to a checkpoint file to resume training or use pretrained weights

**Example:**
```bash
cv mmseg train configs/deeplabv3/config.yaml
cv mmseg train configs/deeplabv3/config.yaml work_dirs/deeplabv3/iter_1000.pth
```

### 2. Detect

Perform semantic segmentation on a dataset using a trained model.

```bash
cv mmseg detect --config-file <path> --epoch <num> --work-dir <path> --dataset-dir <path> --image-files <glob> [options]
```

**Options:**
- `--config-file`: Path to the model configuration file (required)
- `--epoch`: Epoch number to use for segmentation (-1 for all epochs, default: -1)
- `--work-dir`: Directory containing model checkpoints (required)
- `--dataset-dir`: Root directory of the dataset (required)
- `--image-files`: Glob pattern for image files (required)
- `--results-file`: Name of the output CSV file (default: "segmentations.csv")
- `--batch-size`: Batch size for inference (default: 2)
- `--score-threshold`: Minimum confidence score (default: 0.5)
- `--device`: Device to use for inference (default: "cuda:0")

**Example:**
```bash
cv mmseg detect \
    --config-file work_dirs/deeplabv3/deeplabv3_r50-d8.py \
    --epoch 12 \
    --work-dir work_dirs/deeplabv3 \
    --dataset-dir /data/segmentation \
    --image-files 'test/images/*.jpg' \
    --results-file segmentation_results.csv \
    --batch-size 4 \
    --device cuda:0
```

### 3. Eval

Evaluate a trained model against a validation or test dataset.

```bash
cv mmseg eval --model_type <type> --model_name <name> --gt-masks-dir <path> --epochs <num> --num-classes <num> --csv_file_pattern <pattern> --results_file <file>
```

**Options:**
- `--model_type`: Type of model (e.g., "deeplabv3", "pspnet") (required)
- `--model_name`: Name of the specific model (required)
- `--gt-masks-dir`: Directory containing ground truth segmentation masks (required)
- `--epochs`: Number of training epochs to evaluate (required)
- `--num-classes`: Number of segmentation classes (required)
- `--csv_file_pattern`: Pattern for CSV files, use $i for epoch number (required)
- `--results_file`: Name of the output results file (required)

**Example:**
```bash
cv mmseg eval \
    --model_type deeplabv3 \
    --model_name deeplabv3_r50-d8 \
    --gt-masks-dir /data/segmentation/masks \
    --epochs 20 \
    --num-classes 21 \
    --csv_file_pattern "segmentation_epoch_$i.csv" \
    --results_file evaluation_results.csv
```

### 4. Board

Start TensorBoard to visualize training metrics from a JSON log file.

```bash
cv mmseg board <json_log_path> [options]
```

**Arguments:**
- `json_log_path`: Path to the JSON log file generated during training

**Options:**
- `--log-dir`: Directory to store TensorBoard logs (default: "./runs")
- `--port`: Port to run TensorBoard on (default: 6006)

**Example:**
```bash
cv mmseg board work_dirs/deeplabv3/20240101_120000/vis_data/scalars.json --log-dir ./tensorboard_logs --port 6006
```

Then open your browser and navigate to `http://localhost:6006` to view the training metrics.

### 5. Extract Backbone

Extract the backbone (feature extractor) from a trained model and save it separately.

```bash
cv mmseg extract-backbone --config-file <path> --output-file <path> [--load-from <path>]
```

**Options:**
- `--config-file`: Path to the model configuration file (required)
- `--output-file`: Path where the backbone weights will be saved (required)
- `--load-from`: Path to checkpoint file to load weights from (optional)

**Example:**
```bash
cv mmseg extract-backbone \
    --config-file work_dirs/deeplabv3/deeplabv3_r50-d8.py \
    --load-from work_dirs/deeplabv3/iter_20000.pth \
    --output-file work_dirs/deeplabv3_backbone.pth
```

## Configuration File Format

The configuration file should be in YAML format and include the following parameters:

```yaml
config_path: "mmsegmentation/configs"
model_type: "deeplabv3"
model_name: "deeplabv3_r50-d8_512x512_20k_voc12aug"
dataset_dir: "/data/segmentation"
train_dir: "train/images"
val_dir: "val/images"
test_dir: "test/images"
train_seg_dir: "train/masks"
val_seg_dir: "val/masks"
test_seg_dir: "test/masks"
dataset_classes: ["background", "class1", "class2"]
batch_size: 4
epochs: 20000
work_dir: "./work_dirs/deeplabv3"
optimizer: "sgd"
lr: 0.01
weight_decay: 0.0005
momentum: 0.9
augmentations:
  - type: "Resize"
    scale: [512, 512]
  - type: "RandomFlip"
    prob: 0.5
```

## Supported Models

The MMSegmentation CLI supports various segmentation models including:

- **FCN**: Fully Convolutional Networks
- **PSPNet**: Pyramid Scene Parsing Network
- **DeepLabV3**: DeepLab v3
- **DeepLabV3+**: DeepLab v3 Plus

Each model type has multiple variants with different backbones (e.g., ResNet-50, ResNet-101).

## Tips

1. **GPU Memory**: If you encounter out-of-memory errors, reduce the `batch_size` parameter.
2. **Training Resume**: Use the `load-from` parameter to resume training from a checkpoint.
3. **Multiple GPUs**: MMSegmentation supports distributed training. Refer to the MMSegmentation documentation for multi-GPU setup.
4. **Custom Backbones**: You can extract and reuse backbones from one model to another using the `extract-backbone` command.
