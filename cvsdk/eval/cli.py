"""CLI for evaluation commands."""
import click
import pandas as pd
from pathlib import Path

from cvsdk.eval.detection.confusion_matrix import DetectionConfusionMatrix
from cvsdk.eval.detection.precision_confidence import PrecisionConfidenceCurve
from cvsdk.eval.detection.recall_confidence import RecallConfidenceCurve
from cvsdk.eval.detection.precision_recall import PrecisionRecallCurve
from cvsdk.eval.detection.f1_confidence import F1ConfidenceCurve


@click.group()
def eval():
    """CLI for evaluation commands."""
    pass


@eval.command()
@click.option('--ground-truth', required=True, type=click.Path(exists=True), 
              help='Path to ground truth CSV/Parquet file')
@click.option('--detections', required=True, type=click.Path(exists=True), 
              help='Path to detections CSV/Parquet file')
@click.option('--output-dir', required=True, type=click.Path(), 
              help='Directory to save the confusion matrix plot')
@click.option('--iou-threshold', default=0.5, type=float, 
              help='IoU threshold for matching (default: 0.5)')
@click.option('--score-threshold', default=0.0, type=float, 
              help='Minimum confidence score for detections (default: 0.0)')
@click.option('--format', 'file_format', default='csv', type=click.Choice(['csv', 'parquet']),
              help='File format for input files (default: csv)')
def det(ground_truth, detections, output_dir, iou_threshold, score_threshold, file_format):
    """Compute confusion matrix for object detection and save the plot.
    
    Takes ground truth and detections DataFrames and produces a confusion matrix plot.
    
    Example:
        cvsdk evaldet --ground-truth labels.csv --detections preds.csv --output-dir ./results
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load ground truth
    if file_format == 'csv':
        gt_df = pd.read_csv(ground_truth)
    else:
        gt_df = pd.read_parquet(ground_truth)
    
    # Load detections
    if file_format == 'csv':
        det_df = pd.read_csv(detections)
    else:
        det_df = pd.read_parquet(detections)
    
    # Convert image paths to basenames for consistent matching
    gt_df['image'] = gt_df['image'].apply(lambda x: Path(x).name if isinstance(x, (str, Path)) else x)
    det_df['image'] = det_df['image'].apply(lambda x: Path(x).name if isinstance(x, (str, Path)) else x)
    
    click.echo(f"Loaded {len(gt_df)} ground truth boxes from {ground_truth}")
    click.echo(f"Loaded {len(det_df)} detections from {detections}")
    click.echo(f"IoU threshold: {iou_threshold}")
    click.echo(f"Score threshold: {score_threshold}")
    
    # Compute confusion matrix
    calculator = DetectionConfusionMatrix(
        ground_truth_df=gt_df,
        detections_df=det_df,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold
    )
    
    result = calculator.compute()
    
    click.echo(f"\nResults:")
    click.echo(f"  True Positives: {result.total_tp}")
    click.echo(f"  False Positives: {result.total_fp}")
    click.echo(f"  False Negatives: {result.total_fn}")
    click.echo(f"  Precision: {result.precision:.4f}")
    click.echo(f"  Recall: {result.recall:.4f}")
    click.echo(f"  F1 Score: {result.f1_score:.4f}")
    
    # Save confusion matrix plot
    calculator.plot(str(output_path / 'confusion_matrix.png'))
    click.echo(f"Confusion matrix plot saved to: {output_path / 'confusion_matrix.png'}")

    # Precision-Confidence curve
    PrecisionConfidenceCurve(gt_df, det_df, iou_threshold=iou_threshold).plot(
        str(output_path / 'precision_confidence.png')
    )
    click.echo(f"Precision-Confidence plot saved to: {output_path / 'precision_confidence.png'}")

    # Recall-Confidence curve
    RecallConfidenceCurve(gt_df, det_df, iou_threshold=iou_threshold).plot(
        str(output_path / 'recall_confidence.png')
    )
    click.echo(f"Recall-Confidence plot saved to: {output_path / 'recall_confidence.png'}")

    # Precision-Recall curve
    PrecisionRecallCurve(gt_df, det_df, iou_threshold=iou_threshold).plot(
        str(output_path / 'precision_recall.png')
    )
    click.echo(f"Precision-Recall plot saved to: {output_path / 'precision_recall.png'}")

    # F1-Confidence curve
    F1ConfidenceCurve(gt_df, det_df, iou_threshold=iou_threshold).plot(
        str(output_path / 'f1_confidence.png')
    )
    click.echo(f"F1-Confidence plot saved to: {output_path / 'f1_confidence.png'}")

    # Also save metrics as CSV
    per_class_metrics = calculator.get_per_class_metrics()
    metrics_data = []
    for class_name, metrics in per_class_metrics.items():
        metrics_data.append({
            'class': class_name,
            'tp': metrics['tp'],
            'fp': metrics['fp'],
            'fn': metrics['fn'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score']
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_path = output_path / 'metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    click.echo(f"Per-class metrics saved to: {metrics_path}")
