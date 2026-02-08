# MMPretrain CLI

The MMPretrain CLI provides a comprehensive set of commands for training, evaluating, and using image classification models with the mmpretrain library. This CLI supports various popular model architectures including ResNet, Vision Transformer (ViT), Swin Transformer, and EfficientNet.

## Available Commands

- `train` - Train a classification model based on a configuration file
- `detect` - Perform inference/prediction on images using a trained model
- `eval` - Evaluate a model against a validation or test dataset
- `board` - Start TensorBoard to visualize training metrics
- `extract_backbone` - Remove the classification head and save only the feature extractor (backbone)

## Installation

Make sure you have the mmpretrain library installed:

```bash
pip install mmpretrain
```

## Commands

### Train

Train a classification model using a configuration file.

**Basic Usage:**
```bash
cv mmpretrain train CONFIG_FILE [LOAD_FROM]
```

**Arguments:**
- `CONFIG_FILE` - Path to the configuration file (YAML format)
- `LOAD_FROM` - (Optional) Path to a checkpoint file to resume training or use pretrained weights

**Example:**
```bash
# Train from scratch
cv mmpretrain train configs/resnet50.yaml

# Resume training from checkpoint
cv mmpretrain train configs/resnet50.yaml work_dirs/resnet50/epoch_10.pth
```

**Configuration File Example:**

```yaml
config_path: "mmpretrain/configs"
model_type: "resnet"
model_name: "resnet50"
dataset_dir: "/path/to/dataset"
train_dir: "train"
val_dir: "val"
test_dir: "test"
dataset_classes: ["class1", "class2", "class3"]
batch_size: 32
epochs: 100
work_dir: "./work_dirs/resnet50"
annotations_train: "annotations/train.txt"
annotations_val: "annotations/val.txt"
annotations_test: "annotations/test.txt"
optimizer: "adamw"
lr: 0.001
weight_decay: 0.0001
momentum: 0.9
augmentations:
  - type: "RandomFlip"
    prob: 0.5
  - type: "RandomResizedCrop"
    size: 224
backbone:
  type: "ResNet"
  checkpoint: "https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth"
  depth: 50
  frozen_stages: 0
  out_indices: [3]
```

### Detect (Inference)

Perform inference/prediction on images using a trained model.

**Usage:**
```bash
cv mmpretrain detect \
    --config-file CONFIG_FILE \
    --epoch EPOCH \
    --work-dir WORK_DIR \
    --dataset-dir DATASET_DIR \
    --image-files IMAGE_PATTERN \
    --results-file RESULTS_FILE \
    --batch-size BATCH_SIZE \
    --score-threshold THRESHOLD \
    --device DEVICE
```

**Options:**
- `--config-file` - Path to the configuration file
- `--epoch` - Epoch number to use for predictions (-1 for all epochs)
- `--work-dir` - Directory where checkpoints are stored
- `--dataset-dir` - Root directory of the dataset
- `--image-files` - Glob pattern for image files
- `--results-file` - Output CSV file name (default: predictions.csv)
- `--batch-size` - Batch size for inference (default: 2)
- `--score-threshold` - Minimum confidence score (default: 0.5)
- `--device` - Device to use (default: cuda:0)

**Example:**
```bash
cv mmpretrain detect \
    --config-file work_dirs/resnet50/resnet50.py \
    --epoch 50 \
    --work-dir work_dirs/resnet50 \
    --dataset-dir /path/to/dataset \
    --image-files 'test/*.jpg' \
    --results-file predictions_epoch50.csv \
    --batch-size 8 \
    --score-threshold 0.5 \
    --device cuda:0
```

**Output:**
The command generates a CSV file with the following columns:
- `epoch` - Training epoch used
- `filename` - Image filename
- `pred_label` - Predicted class label (integer)
- `pred_class` - Predicted class name
- `pred_score` - Prediction confidence score

### Eval (Evaluation)

Evaluate model performance across training epochs.

**Usage:**
```bash
cv mmpretrain eval \
    --model_type MODEL_TYPE \
    --model_name MODEL_NAME \
    --annotations ANNOTATIONS_FILE \
    --epochs NUM_EPOCHS \
    --csv_file_pattern CSV_PATTERN \
    --results_file RESULTS_FILE \
    --score-threshold THRESHOLD
```

**Options:**
- `--model_type` - Type of model (e.g., resnet, vit, swin)
- `--model_name` - Name of the model
- `--annotations` - Path to ground truth annotations
- `--epochs` - Number of epochs to evaluate
- `--csv_file_pattern` - Pattern for prediction CSV files (use $i for epoch number)
- `--results_file` - Output file for evaluation results
- `--score-threshold` - Minimum confidence score (default: 0.5)

**Example:**
```bash
cv mmpretrain eval \
    --model_type resnet \
    --model_name resnet50 \
    --annotations /path/to/annotations/test.csv \
    --epochs 100 \
    --csv_file_pattern "predictions_epoch_$i.csv" \
    --results_file evaluation_results.csv \
    --score-threshold 0.5
```

**Output:**
The evaluation command generates a CSV file with metrics for each epoch:
- `epoch` - Epoch number
- `accuracy` - Classification accuracy
- `precision` - Weighted precision
- `recall` - Weighted recall
- `f1` - Weighted F1 score

### Board (TensorBoard)

Parse training logs and visualize metrics in TensorBoard.

**Usage:**
```bash
cv mmpretrain board JSON_LOG_PATH [--log-dir LOG_DIR] [--port PORT]
```

**Arguments:**
- `JSON_LOG_PATH` - Path to the JSON log file generated during training

**Options:**
- `--log-dir` - Directory to store TensorBoard logs (default: ./runs)
- `--port` - Port to run TensorBoard on (default: 6006)

**Example:**
```bash
cv mmpretrain board work_dirs/resnet50/20240101_120000.json \
    --log-dir ./tensorboard_logs \
    --port 6006
```

This will:
1. Parse the training JSON log file
2. Convert metrics to TensorBoard format
3. Start TensorBoard server at http://localhost:6006
4. Keep running until Ctrl+C is pressed

**Available Metrics:**
- Training metrics: loss, learning rate, etc.
- Evaluation metrics: accuracy, top-5 accuracy, etc.

### Extract Backbone

Extract the backbone (feature extractor) from a trained model and save it separately.

**Usage:**
```bash
cv mmpretrain extract_backbone \
    --config-file CONFIG_FILE \
    --output-file OUTPUT_FILE \
    --load-from CHECKPOINT_FILE
```

**Options:**
- `--config-file` - Path to the model configuration file
- `--output-file` - Path to save the extracted backbone weights
- `--load-from` - (Optional) Path to checkpoint file with trained weights

**Example:**
```bash
cv mmpretrain extract_backbone \
    --config-file work_dirs/resnet50/resnet50.py \
    --load-from work_dirs/resnet50/epoch_100.pth \
    --output-file work_dirs/resnet50_backbone.pth
```

This command is useful when you want to:
- Use the trained backbone for transfer learning
- Reduce model size by removing the classification head
- Export the feature extractor for downstream tasks

## Supported Models

The MMPretrain CLI supports various model architectures:

### ResNet Family
- resnet18
- resnet34
- resnet50
- resnet101
- resnet152

### Vision Transformer (ViT)
- vit-small-p16
- vit-base-p16
- vit-large-p16

### Swin Transformer
- swin-tiny
- swin-small
- swin-base
- swin-large

### EfficientNet
- efficientnet-b0
- efficientnet-b1
- efficientnet-b2
- efficientnet-b3

## Workflow Example

Here's a complete workflow for training and evaluating a model:

```bash
# 1. Train the model
cv mmpretrain train configs/resnet50.yaml

# 2. Monitor training with TensorBoard
cv mmpretrain board work_dirs/resnet50/20240101_120000.json --port 6006

# 3. Perform inference on test set
cv mmpretrain detect \
    --config-file work_dirs/resnet50/resnet50.py \
    --epoch 100 \
    --work-dir work_dirs/resnet50 \
    --dataset-dir /path/to/dataset \
    --image-files 'test/*.jpg' \
    --results-file predictions.csv

# 4. Evaluate model performance
cv mmpretrain eval \
    --model_type resnet \
    --model_name resnet50 \
    --annotations /path/to/annotations/test.csv \
    --epochs 100 \
    --csv_file_pattern "predictions_epoch_$i.csv" \
    --results_file evaluation.csv

# 5. Extract backbone for transfer learning
cv mmpretrain extract_backbone \
    --config-file work_dirs/resnet50/resnet50.py \
    --load-from work_dirs/resnet50/epoch_100.pth \
    --output-file resnet50_backbone.pth
```

## Tips and Best Practices

1. **GPU Memory**: If you encounter out-of-memory errors, try reducing the batch size:
   ```bash
   --batch-size 16  # or smaller
   ```

2. **Multiple GPUs**: The training command automatically detects and uses available GPUs. To use specific GPUs:
   ```bash
   CUDA_VISIBLE_DEVICES=0,1 cv mmpretrain train configs/resnet50.yaml
   ```

3. **Checkpointing**: Models are automatically saved every epoch. You can resume training from any checkpoint:
   ```bash
   cv mmpretrain train configs/resnet50.yaml work_dirs/resnet50/epoch_50.pth
   ```

4. **Data Augmentation**: Configure augmentations in your YAML file to improve model generalization:
   ```yaml
   augmentations:
     - type: "RandomFlip"
       prob: 0.5
     - type: "RandomResizedCrop"
       size: 224
     - type: "ColorJitter"
       brightness: 0.4
       contrast: 0.4
       saturation: 0.4
   ```

5. **Evaluation**: Always evaluate on a held-out test set to get unbiased performance estimates.

## Troubleshooting

### Import Errors
If you encounter import errors, make sure mmpretrain is installed:
```bash
pip install mmpretrain
```

### CUDA Errors
If you get CUDA errors, check your PyTorch and CUDA versions are compatible:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Configuration Errors
Verify your configuration file follows the correct format and all paths exist.

## Additional Resources

- [MMPretrain Documentation](https://mmpretrain.readthedocs.io/)
- [Model Zoo](https://github.com/open-mmlab/mmpretrain/tree/main/configs)
- [API Reference](https://mmpretrain.readthedocs.io/en/latest/api.html)
