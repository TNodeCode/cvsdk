import signal
import subprocess
import sys
import os

import click

from cvsdk.mm.pretrain.board import log_metrics_to_tensorboard, parse_json_log_file
from cvsdk.mm.pretrain.detect import detect as _detect
from cvsdk.mm.pretrain.eval import evaluate as evaluate
from cvsdk.mm.pretrain.utils import MMPretrainModels

sys.path.append(os.path.join(os.getcwd(), "mmpretrain"))


@click.group()
def mmpretrain():
    """CLI for training and managing a MMPretrain model on a custom dataset."""


@mmpretrain.command()
@click.argument("config-file", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument(
    "load-from", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
def train(config_file: str, load_from: str | None):
    """Train the MMPretrain model."""
    MMPretrainModels.train(config_file=config_file, load_from=load_from)


@mmpretrain.command()
@click.option("--config-file", type=str, required=True, help="Path to configuration file")
@click.option("--epoch", type=int, default=-1, help="Epoch which predictions should be made for (-1 for all)")
@click.option("--work-dir", type=str, required=True, help="Root path of the checkpoint files")
@click.option("--dataset-dir", type=str, required=True, help="Root path of dataset")
@click.option("--image-files", type=str, required=True, help="Glob path for images")
@click.option(
    "--results-file", type=str, default="predictions.csv", help="Name of the resulting CSV file"
)
@click.option(
    "--batch-size", type=int, default=2, help="Batch size for inference (greater than 0, default: 2)"
)
@click.option(
    "--score-threshold",
    type=float,
    default=0.5,
    help="Minimum confidence score for predictions",
)
@click.option(
    "--device", type=str, default="cuda:0", help="Device to use for inference (default: cuda:0)"
)
def detect(
    config_file,
    epoch,
    work_dir,
    dataset_dir,
    image_files,
    results_file,
    batch_size,
    score_threshold,
    device,
):
    """Perform inference on images using a trained model."""
    _detect(
        config_file=config_file,
        epoch=epoch,
        results_file=results_file,
        work_dir=work_dir,
        dataset_dir=dataset_dir,
        image_files=image_files.replace("'", ""),
        batch_size=batch_size,
        score_threshold=score_threshold,
        device=device,
    )


@mmpretrain.command()
@click.option("--model_type", type=str, required=True, help="Type of model to use")
@click.option("--model_name", type=str, required=True, help="Name of the model")
@click.option(
    "--annotations",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    help="Path to the ground truth annotations",
)
@click.option(
    "--epochs", type=int, required=True, help="Number of training epochs (greater than 0)"
)
@click.option(
    "--csv_file_pattern",
    type=str,
    required=True,
    help="Pattern for the CSV files ($i will be replaced by epoch number)",
)
@click.option("--results_file", type=str, required=True, help="Name of the resulting CSV file")
@click.option(
    "--score-threshold",
    type=float,
    default=0.5,
    help="Minimum confidence score for predictions",
)
def eval(
    model_type: str,
    model_name: str,
    annotations: str,
    epochs: int,
    csv_file_pattern: str,
    results_file: str,
    score_threshold: float,
):
    """Evaluate model performance across training epochs."""
    evaluate(
        gt_file_path=annotations,
        model_type=model_type,
        model_name=model_name,
        csv_file_pattern=csv_file_pattern,
        results_file=results_file,
        max_epochs=epochs,
        score_threshold=score_threshold,
    )


@mmpretrain.command()
@click.option(
    "--config-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
@click.option(
    "--output-file", required=True, type=click.Path(exists=False, file_okay=True, dir_okay=False)
)
@click.option(
    "--load-from", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
def extract_backbone(config_file: str, output_file: str, load_from: str | None):
    """Extract the backbone from a trained model and save it."""
    MMPretrainModels.extract_backbone(
        config_file=config_file, load_from=load_from, output_file=output_file
    )


@mmpretrain.command()
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
