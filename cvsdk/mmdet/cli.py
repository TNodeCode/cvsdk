import signal
import subprocess
import sys

import click
import pandas as pd

from cvsdk.mmdet.board import log_metrics_to_tensorboard, parse_json_log_file
from cvsdk.mmdet.detect import detect as _detect
from cvsdk.mmdet.eval import evaluate as evaluate
from cvsdk.mmdet.utils import MMDetModels
from cvsdk.model.eval.boxes import DetectionsEvaluator

import os
import sys
sys.path.append(os.path.join(os.getcwd(), "mmdetection"))

@click.group()
def mmdet():
    """CLI for training and managing a MMDet model on a custom dataset."""

@mmdet.command()
@click.argument("config-file", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("load-from", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False))
def train(config_file: str, load_from: str | None):
    """Train the MMDet model."""
    MMDetModels.train(config_file=config_file, load_from=load_from)

@mmdet.command()
@click.option('--config-file', type=str, required=True, help='Name of the model')
@click.option('--epoch', type=int, default=-1, help='Epoch which detections should be made for')
@click.option('--work-dir', type=str, required=True, help='Root path of the checkpoint files')
@click.option('--dataset-dir', type=str, required=True, help='Root path of dataset')
@click.option('--image-files', type=str, required=True, help='Glob path for images')
@click.option('--results-file', type=str, default='detections.csv', help='Name of the resulting CSV file')
@click.option('--batch-size', type=int, default=2, help='Batch size for training (greater than 0, default: 2)')
@click.option('--score-threshold', type=float, default=0.5, help='Minimum confidence score for bounding box detection')
@click.option('--device', type=str, default='cuda:0', help='Device to use for detection (default: cuda:0)')
def detect(config_file, epoch, work_dir, dataset_dir, image_files, results_file, batch_size, score_threshold, device):
    _detect(
        config_file=config_file,
        epoch=epoch,
        results_file=results_file,
        work_dir=work_dir,
        dataset_dir=dataset_dir,
        image_files=image_files.replace("'", ""),
        batch_size=batch_size,
        score_threshold=score_threshold,
        device=device
    )


@mmdet.command()
@click.option('--gt', type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True, help='Path to the annotations')
@click.option('--det', type=str, required=True, help='Path to the detections file')
@click.option('--out', type=str, required=True, help='Name of the resulting CSV file')
@click.option('--score-threshold', type=float, default=0.5, help='Minimum confidence score for bounding box detection')
def eval(
    gt: str,
    det: str,
    out: str,
    score_threshold: float,
):
    evaluate(
        gt_file=gt,
        detections_file=det,
        results_file=out,
        score_threshold=score_threshold,
    )


@mmdet.command()
@click.argument('gt', type=click.Path(exists=True, dir_okay=False))
@click.argument('det', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_path', type=click.Path(dir_okay=False))
@click.option('--iou-threshold', '-t', default=0.5, show_default=True,
              help='Minimum IoU for a detection to be considered a True Positive.')
@click.option('--epoch', '-e', type=int, default=None,
              help='Epoch number to filter both GT and detections.')
@click.option('--class-name', '-c', default=None,
              help='Class name to filter both GT and detections.')
def evaluate_cli(gt: str,
                 det: str,
                 output_path: str,
                 iou_threshold: float,
                 epoch: int | None,
                 class_name: str | None):
    """Evaluate object detection results against ground truth.

    GT and detections CSVs must include columns:
      epoch, filename, class_index, class_name, xmin, ymin, xmax, ymax
    Detections CSV must also include a 'score' column.

    The output CSV will contain all original detection columns plus an 'evaluation'
    column with values 'TP', 'FP', or 'FN'.
    """
    # Read inputs
    gt_df = pd.read_csv(gt)
    det_df = pd.read_csv(det)

    # Perform evaluation
    eval_df = DetectionsEvaluator.evaluate(
        detections_df=det_df,
        ground_truth_df=gt_df,
        iou_threshold=iou_threshold,
        epoch=epoch,
        class_name=class_name
    )

    # Save results
    eval_df.to_csv(output_path, index=False)
    click.echo(f"Evaluation complete; results saved to {output_path}")


@mmdet.command()
@click.option("--config-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--output-file", required=True, type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--load-from", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False))
def extract_backbone(config_file: str, output_file: str, load_from: str | None):
    MMDetModels.extract_backbone(
        config_file=config_file,
        load_from=load_from,
        output_file=output_file
    )


@mmdet.command()
@click.option("--source-config-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--target-config-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--output-file", required=True, type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--load-source-from", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False))
def copy_backbone(source_config_file: str, target_config_file: str, output_file: str, load_source_from: str | None):
    MMDetModels.copy_backbone(
        source_config_file=source_config_file,
        target_config_file=target_config_file,
        load_source_from=load_source_from,
        output_file=output_file
    )


@mmdet.command()
@click.argument("json_log_path", type=click.Path(exists=True))
@click.option("--log-dir", type=str, default="./runs", help="Directory to store TensorBoard logs")
@click.option("--port", type=int, default=6006, help="Port to run TensorBoard on")
def board(json_log_path, log_dir, port):
    """Parse a training JSON log file and start TensorBoard until Ctrl+C."""
    click.echo(f"Processing log file: {json_log_path}")
    batch_df, eval_df = parse_json_log_file(json_log_path)

    click.echo(f"Logging metrics to TensorBoard at: {log_dir}")
    log_metrics_to_tensorboard(batch_df, eval_df, log_dir=log_dir)

    try:
        click.echo(f"Starting TensorBoard at http://localhost:{port} (Press Ctrl+C to stop)")
        process = subprocess.Popen(["tensorboard", "--logdir", log_dir, "--port", str(port)])

        # Wait for Ctrl+C
        process.wait()
    except KeyboardInterrupt:
        click.echo("\nStopping TensorBoard...")
        process.send_signal(signal.SIGINT)
        process.wait()
