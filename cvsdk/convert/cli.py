"""CLI for converting between dataset formats."""
from pathlib import Path
import click
from cvsdk.model.loaders.coco import CocoLoader
from cvsdk.model.loaders.yolo import YOLOLoader


@click.group()
def convert() -> None:
    """Convert between dataset formats (YOLO <-> COCO)."""
    pass


@convert.command()
@click.argument("yolo_root", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("output_path", type=click.Path(file_okay=True, dir_okay=False))
@click.option("--task-type", default="detection", type=click.Choice(["classification", "detection", "segmentation", "panoptic", "tracking"]), help="Task type for the dataset")
def yolo_to_coco(yolo_root: str, output_path: str, task_type: str) -> None:
    """Convert a YOLO dataset to COCO format.
    
    Args:
        yolo_root: Path to the YOLO dataset root directory.
        output_path: Path to the output COCO JSON file.
        task_type: Task type of the dataset.
    """
    dataset = YOLOLoader.import_dataset(yolo_root, task_type=task_type)
    CocoLoader.export_dataset(dataset, output_path=output_path)
    click.echo(f"Converted YOLO dataset from {yolo_root} to COCO format at {output_path}")


@convert.command()
@click.argument("coco-json", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("output-dir", type=click.Path(file_okay=False, dir_okay=True))
@click.option("--task-type", default="detection", type=click.Choice(["classification", "detection", "segmentation", "panoptic", "tracking"]), help="Task type for the dataset")
def coco_to_yolo(coco_json: str, output_dir: str, task_type: str) -> None:
    """Convert a COCO dataset to YOLO format.
    
    Args:
        coco_json: Path to the COCO JSON file.
        output_dir: Path to the output YOLO dataset root directory.
        task_type: Task type of the dataset.
    """
    dataset = CocoLoader.import_dataset(coco_json, task_type=task_type)
    YOLOLoader.export_dataset(dataset, output_dir=output_dir)
    click.echo(f"Converted COCO dataset from {coco_json} to YOLO format at {output_dir}")
