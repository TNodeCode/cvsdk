import os
import click
from ultralytics import YOLO, RTDETR
import ultralytics.data.dataset as dataset
import ultralytics.data.build as build
import pandas as pd
from pathlib import Path
from glob import glob
import onnxruntime
import cv2
import numpy as np
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from cvsdk.yolo.dataset_weighted import YOLOWeightedDataset
from cvsdk.yolo.clean import DatasetCleaner

from cvsdk.yolo.inspect import inspect as inspect_group
from .transformations import custom_transforms

dataset.YOLODataset = YOLOWeightedDataset
build.YOLODataset = YOLOWeightedDataset


@click.group()
def yolo():
    """CLI for training and managing a YOLO model on a custom dataset."""
    pass


yolo.add_command(inspect_group)

@yolo.command()
@click.option('--data', type=click.Path(exists=True), default='./data/my-dataset', help='Path to training data')
@click.option('--model', type=str, required=True, help='YOLO model to train, e.g., "yolov8n.pt"')
@click.option('--epochs', type=int, default=10, help='Number of epochs for training')
@click.option('--batch-size', type=int, default=2, help='Batch size')
@click.option('--img-size', type=int, default=640, help='Image size for training')
@click.option('--workers', type=int, default=0, help='Number of workers for data loading (0 = no workers)')
@click.option('--resume', type=bool, default=False, help='Whether to resume training')
@click.option('--save-dir', type=click.Path(exists=False), default=None, help='Path to save dir')
def train(data, model, epochs, batch_size, img_size, workers, resume, save_dir):
    """Train the YOLO model."""
    model = YOLO(model) if "yolo" in model else RTDETR(model)
    cfg_path = os.path.join(os.path.dirname(data), "config.yaml")
    cfg_path = cfg_path if os.path.exists(cfg_path) else None
    # Training the model
    model.train(
        data=data,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        resume=resume,
        workers=workers,
        cfg=cfg_path,
        patience=5,
        save_dir=save_dir,
        exist_ok=False,
        augmentations=custom_transforms,  # Use custom transforms
    )

@yolo.command()
@click.option('--data', type=click.Path(exists=True), required=True, help='Path to validation data')
@click.option('--model', type=click.Path(exists=True), required=True, help='Path to trained YOLO model')
def val(data, model):
    """Validate the YOLO model."""
    model = YOLO(model)
    # Run validation
    results = model.val(data=data)
    print(f"Validation Results: {results}")

@yolo.command()
@click.option('--images', type=str, required=True, help='Directory where to search for images to run inference on')
@click.option('--model', type=click.Path(exists=True), required=True, help='Path to trained YOLO model')
@click.option('--output-csv', type=click.Path(), help="Path to save detections as CSV")
def inference(images, model, output_csv):
    """Run inference on images."""
    model = YOLO(model)
    folder = Path(images)
    image_extensions = {".jpg", ".jpeg", ".png"}
    img_paths = [p for p in  folder.rglob("*") if p.suffix.lower() in image_extensions]
    detections = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Running inference...", total=len(img_paths))

        for img_path in img_paths:
            results = model(img_path)
            img = cv2.imread(str(img_path))
            img_height, img_width = img.shape[:2] if img is not None else (0, 0)

            for r in results:
                for box in r.boxes:
                    xmin, ymin, xmax, ymax = box.xyxy[0]
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    detections.append({
                        'x0': int(xmin),
                        'y0': int(ymin),
                        'x1': int(xmax),
                        'y1': int(ymax),
                        'image': str(img_path).replace(images, ""),
                        'label': cls,
                        'score': conf,
                        'img_width': img_width,
                        'img_height': img_height
                    })

            progress.update(task, advance=1)

    if output_csv:
        pd.DataFrame(detections).to_csv(output_csv, index=False)
        print(f"Saved detections to {output_csv}")

@yolo.command()
@click.option('--model', type=click.Path(exists=True), required=True, help='Path to trained YOLO model')
@click.option('--img-size', type=int, default=640, help='Image size for training')
@click.option('--export-format', type=click.Choice(['onnx', 'coreml', 'tensorrt', 'tflite']), required=True, help='Export format')
def export(model, img_size, export_format):
    """Export the YOLO model to a specified format."""
    model = YOLO(model)
    # Export the model to the specified format
    model.export(format=export_format, imgsz=(img_size, img_size), simplify=True)
    print(f"Model {model} exported to {format} format.")

@yolo.command()
@click.option('--onnx-model-path', type=click.Path(exists=True), required=True, help='Path to YOLO ONNX model')
@click.option('--image-path', type=click.Path(exists=True), required=True, help='Path to an image')
def onnx_inference(onnx_model_path, image_path):
    """Run inference with a YOLO ONNX model on a single image."""
    session = onnxruntime.InferenceSession(onnx_model_path)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (640, 640))
    img = img.transpose(2, 0, 1)  # Change shape to (C, H, W)
    img = img[np.newaxis, :].astype(np.float32) / 255.0

    inputs = {session.get_inputs()[0].name: img}
    output = session.run(None, inputs)[0]

    for det in output:
        xmin, ymin, xmax, ymax, conf, cls = det
        if conf > 0.5:  # Only print detections with confidence above threshold
            print(f"Detected class {cls} at [{xmin}, {ymin}, {xmax}, {ymax}] with confidence {conf}")

@yolo.command()
@click.option('--dataset', type=click.Path(exists=True), default='./datasets/yolo_gsta', help='Path to YOLO dataset root')
@click.option('--output-dir', type=click.Path(), default='./datasets/yolo_unlabelled', help='Output directory for mismatched files')
@click.option('--move/--no-move', 'should_move', default=True, help='Whether to actually move files (default: move)')
def clean(dataset, output_dir, should_move):
    """
    Find and optionally move mismatched files in a YOLO dataset.
    
    This command handles two types of mismatches:
    - Images without labels (or with empty/background labels)
    - Labels without corresponding images (orphan labels)
    
    By default, files are moved to the output directory. Use --no-move to just preview.
    """
    dataset = Path(dataset)
    output_dir = Path(output_dir)
    
    print(f"Dataset directory: {dataset}")
    print(f"Output directory: {output_dir}")
    print(f"Move files: {should_move}")
    print("-" * 50)
    
    # Initialize the cleaner
    cleaner = DatasetCleaner(dataset, output_dir)
    
    print(f"Valid class IDs: {sorted(cleaner.valid_class_ids)}")
    print()
    
    # Find unlabelled images and images with invalid coordinates
    print("Scanning for images without labels and with invalid coordinates...")
    unlabelled, invalid_coords = cleaner.find_unlabelled_images()
    
    print(f"Found {len(unlabelled)} images without labels (or background only)")
    print(f"Found {len(invalid_coords)} images with invalid coordinates")
    
    # Display some examples
    if unlabelled:
        print("\nFirst 10 unlabelled/background images:")
        for img_path, rel_path in unlabelled[:10]:
            print(f"  - {rel_path}")
        if len(unlabelled) > 10:
            print(f"  ... and {len(unlabelled) - 10} more")
    
    if invalid_coords:
        print("\nFirst 10 images with invalid coordinates:")
        for img_path, rel_path in invalid_coords[:10]:
            print(f"  - {rel_path}")
        if len(invalid_coords) > 10:
            print(f"  ... and {len(invalid_coords) - 10} more")
    
    # Find orphan labels (labels without corresponding images)
    print("\nScanning for orphan labels (labels without corresponding images)...")
    orphan_labels = cleaner.find_orphan_labels()
    
    print(f"Found {len(orphan_labels)} orphan label files")
    
    if orphan_labels:
        print("\nFirst 10 orphan labels:")
        for label_path, rel_path in orphan_labels[:10]:
            print(f"  - {rel_path}")
        if len(orphan_labels) > 10:
            print(f"  ... and {len(orphan_labels) - 10} more")
    
    # Combine both lists for moving
    all_to_move = unlabelled + invalid_coords
    
    # Move files if should_move is True
    if should_move:
        print("\nMoving unlabelled images and images with invalid coordinates...")
        moved = cleaner.move_images_to_unlabelled(all_to_move)
        print(f"Successfully moved {moved} images to {output_dir}")
        
        # Move orphan labels
        if orphan_labels:
            print("\nMoving orphan labels...")
            moved_labels = cleaner.move_orphan_labels(orphan_labels)
            print(f"Successfully moved {moved_labels} orphan labels to {output_dir}")
        
        print("\nDone! Mismatched files have been moved.")
    else:
        print("\n--no-move specified. No files were moved.")
        print("Run without --no-move to move the mismatched files.")
    
    print("\nSummary:")
    print(f"  - Images without labels: {len(unlabelled)}")
    print(f"  - Images with invalid coordinates: {len(invalid_coords)}")
    print(f"  - Orphan labels (no corresponding image): {len(orphan_labels)}")
    print(f"  - Total files to move: {len(all_to_move) + len(orphan_labels)}")


if __name__ == '__main__':
    yolo()
