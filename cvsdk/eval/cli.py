"""CLI for evaluation commands."""
import click
import pandas as pd
from pathlib import Path

from cvsdk.eval.detection.confusion_matrix import DetectionConfusionMatrix
from cvsdk.eval.detection.precision_confidence import PrecisionConfidenceCurve
from cvsdk.eval.detection.recall_confidence import RecallConfidenceCurve
from cvsdk.eval.detection.precision_recall import PrecisionRecallCurve
from cvsdk.eval.detection.f1_confidence import F1ConfidenceCurve

from cvsdk.eval.classification._probability import ProbabilitySpec
from cvsdk.eval.classification.confusion_matrix import ClassificationConfusionMatrix
from cvsdk.eval.classification.precision_confidence import ClassificationPrecisionConfidenceCurve
from cvsdk.eval.classification.recall_confidence import ClassificationRecallConfidenceCurve
from cvsdk.eval.classification.f1_confidence import ClassificationF1ConfidenceCurve
from cvsdk.eval.classification.roc_curve import ROCCurve
from cvsdk.eval.classification.precision_recall import ClassificationPrecisionRecallCurve
from cvsdk.eval.classification.reliability_diagram import ReliabilityDiagram
from cvsdk.eval.classification.probability_kde import ProbabilityKDE
from cvsdk.eval.classification.topk_accuracy import TopKAccuracyCurve


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


@eval.command()
@click.option('--ground-truth', required=True, type=click.Path(exists=True),
              help='Path to ground truth CSV/Parquet file')
@click.option('--predictions', required=True, type=click.Path(exists=True),
              help='Path to predictions CSV/Parquet file')
@click.option('--output-dir', required=True, type=click.Path(),
              help='Directory to save evaluation outputs')
@click.option('--gt-label-col', required=True, type=str,
              help='Column name for ground truth labels')
@click.option('--pred-label-col', required=True, type=str,
              help='Column name for predicted labels')
@click.option('--prob-col', default=None, type=str,
              help='Column name for a single predicted probability (max confidence)')
@click.option('--prob-prefix', default=None, type=str,
              help='Prefix for per-class probability columns (e.g., "prob_" for prob_cat, prob_dog)')
@click.option('--prob-columns', default=None, type=str,
              help='Explicit per-class probability columns as class:col pairs, comma-separated '
                   '(e.g., "cat:prob_cat,dog:prob_dog")')
@click.option('--format', 'file_format', default='csv', type=click.Choice(['csv', 'parquet']),
              help='File format for input files (default: csv)')
@click.option('--merge-key', default=None, type=str,
              help='Column name to merge ground truth and predictions on. '
                   'If not set, files are assumed to have the same row order.')
@click.option('--n-bins', default=10, type=int,
              help='Number of bins for reliability diagram (default: 10)')
@click.option('--max-k', default=5, type=int,
              help='Maximum k for top-k accuracy curves (default: 5)')
def cls(ground_truth, predictions, output_dir, gt_label_col, pred_label_col,
        prob_col, prob_prefix, prob_columns, file_format, merge_key, n_bins, max_k):
    """Compute classification evaluation metrics and generate plots.

    Takes ground truth and predictions DataFrames and produces confusion matrix,
    ROC curves, PR curves, confidence curves, reliability diagrams, KDE plots,
    and top-k accuracy curves.

    Example:
        cvsdk eval cls --ground-truth labels.csv --predictions preds.csv \
            --output-dir ./results --gt-label-col label --pred-label-col pred \
            --prob-prefix prob_
    """
    # Validate mutually exclusive probability options
    prob_options = [prob_col, prob_prefix, prob_columns]
    specified = sum(o is not None for o in prob_options)
    if specified > 1:
        raise click.UsageError(
            "Only one of --prob-col, --prob-prefix, --prob-columns may be specified."
        )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load files
    if file_format == 'csv':
        gt_df = pd.read_csv(ground_truth)
        pred_df = pd.read_csv(predictions)
    else:
        gt_df = pd.read_parquet(ground_truth)
        pred_df = pd.read_parquet(predictions)

    click.echo(f"Loaded {len(gt_df)} ground truth rows from {ground_truth}")
    click.echo(f"Loaded {len(pred_df)} predictions from {predictions}")

    # Align rows
    if merge_key is not None:
        merged = gt_df.merge(pred_df, on=merge_key, how='inner')
        click.echo(f"Merged on '{merge_key}': {len(merged)} rows")
    else:
        if len(gt_df) != len(pred_df):
            raise click.UsageError(
                f"Row count mismatch: GT has {len(gt_df)}, predictions has {len(pred_df)}. "
                "Use --merge-key to align rows."
            )
        merged = gt_df.copy()
        for col in pred_df.columns:
            if col not in merged.columns:
                merged[col] = pred_df[col].values
            else:
                merged[f"pred_{col}"] = pred_df[col].values

    # Determine class names
    y_true = merged[gt_label_col].values
    y_pred = merged[pred_label_col].values
    all_classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    class_names = [str(c) for c in all_classes]

    click.echo(f"Classes: {class_names}")

    # Build ProbabilitySpec — validate against merged since that's what we'll extract from
    if prob_col is not None:
        spec = ProbabilitySpec.from_single_column(prob_col)
    elif prob_prefix is not None:
        spec = ProbabilitySpec.from_prefix(merged, prob_prefix, all_classes)
    elif prob_columns is not None:
        col_mapping = {}
        for pair in prob_columns.split(','):
            cls_name, col_name = pair.strip().split(':')
            col_mapping[cls_name.strip()] = col_name.strip()
        spec = ProbabilitySpec.from_column_list(merged, col_mapping)
    else:
        spec = ProbabilitySpec.none()

    # ---- Always compute: Confusion Matrix ----
    calculator = ClassificationConfusionMatrix(
        y_true=y_true, y_pred=y_pred, class_names=class_names
    )
    result = calculator.compute()

    click.echo(f"\nResults:")
    click.echo(f"  Accuracy: {result.accuracy:.4f}")
    click.echo(f"  Precision: {result.overall_precision:.4f}")
    click.echo(f"  Recall: {result.overall_recall:.4f}")
    click.echo(f"  F1 Score: {result.overall_f1:.4f}")

    calculator.plot(str(output_path / 'confusion_matrix.png'))
    click.echo(f"Confusion matrix plot saved to: {output_path / 'confusion_matrix.png'}")

    # Save metrics CSV
    metrics_df = calculator.get_metrics_dataframe()
    metrics_path = output_path / 'metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    click.echo(f"Per-class metrics saved to: {metrics_path}")

    # ---- Probability-dependent metrics ----
    if not spec.has_probabilities:
        skipped = [
            "Precision-Confidence", "Recall-Confidence", "F1-Confidence",
            "ROC curves", "Precision-Recall curve", "Reliability diagram",
            "Probability KDE", "Top-k accuracy",
        ]
        click.echo("\nNo probabilities provided. Skipping:")
        for name in skipped:
            click.echo(f"  - {name}")
        return

    # Extract confidence array (always from merged)
    confidence = spec.get_confidence(merged)

    # Precision-Confidence curve
    ClassificationPrecisionConfidenceCurve(
        y_true, y_pred, confidence, class_names
    ).plot(str(output_path / 'precision_confidence.png'))
    click.echo(f"Precision-Confidence plot saved to: {output_path / 'precision_confidence.png'}")

    # Recall-Confidence curve
    ClassificationRecallConfidenceCurve(
        y_true, y_pred, confidence, class_names
    ).plot(str(output_path / 'recall_confidence.png'))
    click.echo(f"Recall-Confidence plot saved to: {output_path / 'recall_confidence.png'}")

    # F1-Confidence curve
    ClassificationF1ConfidenceCurve(
        y_true, y_pred, confidence, class_names
    ).plot(str(output_path / 'f1_confidence.png'))
    click.echo(f"F1-Confidence plot saved to: {output_path / 'f1_confidence.png'}")

    if not spec.has_per_class:
        skipped_per_class = [
            "ROC curves", "Precision-Recall curve", "Reliability diagram",
            "Probability KDE", "Top-k accuracy",
        ]
        click.echo("\nNo per-class probabilities. Skipping:")
        for name in skipped_per_class:
            click.echo(f"  - {name}")
        return

    # Extract probability matrix (always from merged)
    y_prob = spec.get_probability_matrix(merged)

    # ROC curves
    ROCCurve(y_true, y_prob, class_names).plot(
        str(output_path / 'roc_curves.png')
    )
    click.echo(f"ROC curves plot saved to: {output_path / 'roc_curves.png'}")

    # Precision-Recall curve
    ClassificationPrecisionRecallCurve(y_true, y_prob, class_names).plot(
        str(output_path / 'precision_recall.png')
    )
    click.echo(f"Precision-Recall plot saved to: {output_path / 'precision_recall.png'}")

    # Reliability diagram
    ReliabilityDiagram(y_true, y_prob, class_names, n_bins=n_bins).plot(
        str(output_path / 'reliability_diagram.png')
    )
    click.echo(f"Reliability diagram saved to: {output_path / 'reliability_diagram.png'}")

    # Probability KDE
    ProbabilityKDE(y_true, y_pred, y_prob, class_names).plot(
        str(output_path / 'probability_kde.png')
    )
    click.echo(f"Probability KDE saved to: {output_path / 'probability_kde.png'}")

    # Top-k accuracy
    TopKAccuracyCurve(y_true, y_prob, class_names, max_k=max_k).plot(
        str(output_path / 'topk_accuracy.png')
    )
    click.echo(f"Top-k accuracy plot saved to: {output_path / 'topk_accuracy.png'}")
