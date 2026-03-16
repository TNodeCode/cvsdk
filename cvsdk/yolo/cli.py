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
from cvsdk.yolo.dataset_weighted import YOLOWeightedDataset

from cvsdk.yolo.inspect import inspect as inspect_group

dataset.YOLODataset = YOLOWeightedDataset
build.YOLODataset = YOLOWeightedDataset


@click.group()
def yolo():
    """CLI for training and managing a YOLO model on a custom dataset."""
    pass


yolo.add_command(inspect_group)

@yolo.command()
@click.option('--data-path', type=click.Path(exists=True), default='./data/my-dataset', help='Path to training data')
@click.option('--model-name', type=str, required=True, help='YOLO model to train, e.g., "yolov8n.pt"')
@click.option('--epochs', type=int, default=10, help='Number of epochs for training')
@click.option('--batch-size', type=int, default=2, help='Batch size')
@click.option('--img-size', type=int, default=640, help='Image size for training')
@click.option('--workers', type=int, default=0, help='Number of workers for data loading (0 = no workers)')
@click.option('--resume', type=click.Path(exists=True), help='Path to checkpoint to resume training')
def train(data_path, model_name, epochs, batch_size, img_size, workers, resume):
    """Train the YOLO model."""
    print(f"Training {model_name} on {data_path} for {epochs} epochs at image size {img_size}.")
    model = YOLO(model_name) if "yolo" in model_name else RTDETR(model_name)
    # Training the model
    model.train(data=data_path, epochs=epochs, batch=batch_size, imgsz=img_size, resume=resume, workers=workers)
    print("Training completed.")

@yolo.command()
@click.option('--data-path', type=click.Path(exists=True), required=True, help='Path to validation data')
@click.option('--model-path', type=click.Path(exists=True), required=True, help='Path to trained YOLO model')
def val(data_path, model_path):
    """Validate the YOLO model."""
    model = YOLO(model_path)
    # Run validation
    results = model.val(data=data_path)
    print(f"Validation Results: {results}")

@yolo.command()
@click.option('--images', type=str, required=True, help='Directory where to search for images to run inference on')
@click.option('--model-path', type=click.Path(exists=True), required=True, help='Path to trained YOLO model')
@click.option('--output-csv', type=click.Path(), help="Path to save detections as CSV")
def inference(images, model_path, output_csv):
    """Run inference on images."""
    model = YOLO(model_path)
    folder = Path(images)
    image_extensions = {".jpg", ".jpeg", ".png"}
    img_paths = [p for p in  folder.rglob("*") if p.suffix.lower() in image_extensions]
    detections = []

    for img_path in img_paths:
        results = model(img_path)
        for r in results:
            for box in r.boxes:
                xmin, ymin, xmax, ymax = box.xyxy[0]
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                detections.append({
                    'xmin': int(xmin), 'ymin': int(ymin), 'xmax': int(xmax), 'ymax': int(ymax),
                    'cls': cls, 'score': conf
                })

    if output_csv:
        pd.DataFrame(detections).to_csv(output_csv, index=False)
        print(f"Saved detections to {output_csv}")

@yolo.command()
@click.option('--model-path', type=click.Path(exists=True), required=True, help='Path to trained YOLO model')
@click.option('--img-size', type=int, default=640, help='Image size for training')
@click.option('--export-format', type=click.Choice(['onnx', 'coreml', 'tensorrt', 'tflite']), required=True, help='Export format')
def export(model_path, img_size, export_format):
    """Export the YOLO model to a specified format."""
    model = YOLO(model_path)
    # Export the model to the specified format
    model.export(format=export_format, imgsz=(img_size, img_size), simplify=True)
    print(f"Model {model_path} exported to {format} format.")

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

if __name__ == '__main__':
    cli()
