"""CLI for RF Segment models - training, evaluation, and export."""
import os
import click
from rfdetr import RFDETRSegNano, RFDETRSegSmall, RFDETRSegMedium, RFDETRSegLarge, RFDETRSegXLarge, RFDETRSeg2XLarge


# Valid model sizes for RF Segment
MODEL_SIZES = ['nano', 'small', 'base', 'medium', 'large', 'xlarge', '2xlarge']

# Mapping of size strings to RF Segment model classes
SEG_MODEL_CLASSES = {
    'nano': RFDETRSegNano,
    'small': RFDETRSegSmall,
    'medium': RFDETRSegMedium,
    'large': RFDETRSegLarge,
    'xlarge': RFDETRSegXLarge,
    '2xlarge': RFDETRSeg2XLarge,
}


@click.group()
def seg():
    """CLI for RF Segment models - training, evaluation, and export."""
    pass


@seg.command()
@click.option('--dataset-dir', required=True, type=click.Path(exists=True), help='Path to the dataset directory.')
@click.option('--size', default='base', type=click.Choice(MODEL_SIZES), help='Model size: nano, small, base, medium, large, xlarge, 2xlarge (default: base).')
@click.option('--epochs', default=10, type=int, help='Number of training epochs (default: 10).')
@click.option('--batch-size', default=4, type=int, help='Batch size for training (default: 4).')
@click.option('--grad-accum-steps', default=4, type=int, help='Gradient accumulation steps (default: 4).')
@click.option('--lr', default=1e-4, type=float, help='Learning rate (default: 1e-4).')
@click.option('--output-dir', required=True, type=click.Path(), help='Path to the output directory for saving checkpoints and logs.')
@click.option('--resume', default=None, type=click.Path(exists=True), help='Path to a checkpoint to resume training from.')
@click.option('--early-stopping', is_flag=True, default=True, help='Enable early stopping (default: True).')
def train(dataset_dir, size, epochs, batch_size, grad_accum_steps, lr, output_dir, resume, early_stopping):
    """
    Train an RF Segment model on a dataset.
    
    This command initializes an RF Segment model and starts training with the specified parameters.
    """
    click.echo(f"Starting RF Segment training...")
    click.echo(f"Model size: {size}")
    click.echo(f"Dataset directory: {dataset_dir}")
    click.echo(f"Epochs: {epochs}")
    click.echo(f"Batch size: {batch_size}")
    click.echo(f"Gradient accumulation steps: {grad_accum_steps}")
    click.echo(f"Learning rate: {lr}")
    click.echo(f"Output directory: {output_dir}")
    click.echo(f"Resume from: {resume if resume else 'None'}")
    click.echo(f"Early stopping: {early_stopping}")
    
    model_class = SEG_MODEL_CLASSES[size]
    model = model_class()
    
    model.train(
        dataset_dir=dataset_dir,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        lr=lr,
        output_dir=output_dir,
        resume=resume,
        early_stopping=early_stopping,
    )
    
    click.echo("Training completed successfully!")


@seg.command()
@click.option('--checkpoint', required=True, type=click.Path(exists=True), help='Path to the model checkpoint.')
@click.option('--size', default='base', type=click.Choice(MODEL_SIZES), help='Model size: nano, small, base, medium, large, xlarge, 2xlarge (default: base).')
@click.option('--image', required=True, type=click.Path(exists=True), help='Path to the image file.')
@click.option('--output', default='segmentations.csv', type=click.Path(), help='Output CSV file path (default: segmentations.csv).')
@click.option('--score-threshold', default=0.5, type=float, help='Minimum confidence score for segmentation (default: 0.5).')
def detect(checkpoint, size, image, output, score_threshold):
    """
    Run segmentation on a single image using a trained RF Segment model.
    
    This command loads a model from a checkpoint and runs inference on the specified image.
    """
    click.echo(f"Running RF Segment detection...")
    click.echo(f"Model size: {size}")
    click.echo(f"Checkpoint: {checkpoint}")
    click.echo(f"Image: {image}")
    click.echo(f"Score threshold: {score_threshold}")
    click.echo(f"Output file: {output}")
    
    model_class = SEG_MODEL_CLASSES[size]
    model = model_class(pretrain_weights=checkpoint)
    
    detections = model.predict(image)
    
    # Save detections to CSV
    import pandas as pd
    if len(detections) > 0:
        df = pd.DataFrame({
            'image': [os.path.basename(image)] * len(detections),
            'label': [d.label for d in detections],
            'score': [d.score for d in detections],
            'x': [d.x for d in detections],
            'y': [d.y for d in detections],
            'width': [d.width for d in detections],
            'height': [d.height for d in detections],
        })
    else:
        df = pd.DataFrame(columns=['image', 'label', 'score', 'x', 'y', 'width', 'height'])
    
    df.to_csv(output, index=False)
    click.echo(f"Segmentations saved to {output}")
    click.echo(f"Found {len(detections)} segmentations")


@seg.command()
@click.option('--checkpoint', required=True, type=click.Path(exists=True), help='Path to the model checkpoint.')
@click.option('--size', default='base', type=click.Choice(MODEL_SIZES), help='Model size: nano, small, base, medium, large, xlarge, 2xlarge (default: base).')
@click.option('--dataset-dir', required=True, type=click.Path(exists=True), help='Path to the dataset directory.')
@click.option('--output', default='segmentation_evaluation_results.csv', type=click.Path(), help='Output CSV file path (default: segmentation_evaluation_results.csv).')
@click.option('--score-threshold', default=0.5, type=float, help='Minimum confidence score for segmentation (default: 0.5).')
def eval(checkpoint, size, dataset_dir, output, score_threshold):
    """
    Evaluate an RF Segment model on a dataset.
    
    This command loads a model from a checkpoint and evaluates it on the specified dataset,
    saving the results to a CSV file.
    """
    click.echo(f"Running RF Segment evaluation...")
    click.echo(f"Model size: {size}")
    click.echo(f"Checkpoint: {checkpoint}")
    click.echo(f"Dataset directory: {dataset_dir}")
    click.echo(f"Score threshold: {score_threshold}")
    click.echo(f"Output file: {output}")
    
    model_class = SEG_MODEL_CLASSES[size]
    model = model_class(pretrain_weights=checkpoint)
    
    # Get all images in the dataset directory
    supported_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = [
        os.path.join(dataset_dir, fname)
        for fname in os.listdir(dataset_dir)
        if fname.lower().endswith(supported_ext)
    ]
    
    if not image_files:
        click.echo("No images found in the specified dataset directory.")
        return
    
    click.echo(f"Found {len(image_files)} images to evaluate")
    
    all_detections = []
    for image_path in image_files:
        detections = model.predict(image_path)
        for det in detections:
            all_detections.append({
                'image': os.path.basename(image_path),
                'label': det.label,
                'score': det.score,
                'x': det.x,
                'y': det.y,
                'width': det.width,
                'height': det.height,
            })
    
    # Save all detections to CSV
    import pandas as pd
    if all_detections:
        df = pd.DataFrame(all_detections)
    else:
        df = pd.DataFrame(columns=['image', 'label', 'score', 'x', 'y', 'width', 'height'])
    
    df.to_csv(output, index=False)
    click.echo(f"Evaluation results saved to {output}")
    click.echo(f"Total segmentations: {len(all_detections)}")


@seg.command()
@click.option('--checkpoint', required=True, type=click.Path(exists=True), help='Path to the model checkpoint.')
@click.option('--size', default='base', type=click.Choice(MODEL_SIZES), help='Model size: nano, small, base, medium, large, xlarge, 2xlarge (default: base).')
@click.option('--output', required=True, type=click.Path(), help='Output ONNX file path.')
@click.option('--input-size', default=640, type=int, help='Input size for the model (default: 640).')
def export(checkpoint, size, output, input_size):
    """
    Export an RF Segment model to ONNX format.
    
    This command loads a model from a checkpoint and exports it to ONNX format
    for deployment in inference environments.
    """
    click.echo(f"Exporting RF Segment model to ONNX...")
    click.echo(f"Model size: {size}")
    click.echo(f"Checkpoint: {checkpoint}")
    click.echo(f"Output file: {output}")
    click.echo(f"Input size: {input_size}")
    
    model_class = SEG_MODEL_CLASSES[size]
    model = model_class(pretrain_weights=checkpoint)
    
    # Export to ONNX
    model.export_onnx(output_path=output, input_size=input_size)
    
    click.echo(f"Model exported successfully to {output}")
