"""CLI for converting between dataset formats."""
from pathlib import Path
import click
from cvsdk.model.loaders.coco import CocoLoader
from cvsdk.model.loaders.yolo import YOLOLoader
from cvsdk.model.loaders.dataframe import DataframeLoader


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


@convert.command()
@click.argument("yolo_root", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("output_path", type=click.Path(file_okay=True, dir_okay=False))
@click.option("--task-type", default="detection", type=click.Choice(["classification", "detection", "segmentation", "panoptic", "tracking"]), help="Task type for the dataset")
@click.option("--file-format", default=None, type=click.Choice(["csv", "parquet"]), help="Output file format (default: inferred from output_path)")
def yolo_to_df(yolo_root: str, output_path: str, task_type: str, file_format: str) -> None:
    """Convert a YOLO dataset to DataFrame format (CSV or Parquet).
    
    Args:
        yolo_root: Path to the YOLO dataset root directory.
        output_path: Path to the output CSV or Parquet file.
        task_type: Task type of the dataset.
        file_format: Output file format ('csv' or 'parquet').
    """
    dataset = YOLOLoader.import_dataset(yolo_root, task_type=task_type)
    DataframeLoader.export_dataset(dataset, output_path=output_path, file_format=file_format)
    click.echo(f"Converted YOLO dataset from {yolo_root} to DataFrame format at {output_path}")


@convert.command()
@click.argument("input_path", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("output_dir", type=click.Path(file_okay=False, dir_okay=True))
@click.option("--task-type", default="detection", type=click.Choice(["classification", "detection", "segmentation", "panoptic", "tracking"]), help="Task type for the dataset")
@click.option("--file-format", default=None, type=click.Choice(["csv", "parquet"]), help="Input file format (default: inferred from input_path)")
def df_to_yolo(input_path: str, output_dir: str, task_type: str, file_format: str) -> None:
    """Convert a DataFrame dataset (CSV or Parquet) to YOLO format.
    
    Args:
        input_path: Path to the input CSV or Parquet file.
        output_dir: Path to the output YOLO dataset root directory.
        task_type: Task type of the dataset.
        file_format: Input file format ('csv' or 'parquet').
    """
    dataset = DataframeLoader.import_dataset(input_path, task_type=task_type, file_format=file_format)
    YOLOLoader.export_dataset(dataset, output_dir=output_dir)
    click.echo(f"Converted DataFrame dataset from {input_path} to YOLO format at {output_dir}")


@convert.command()
@click.argument("coco-json", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("output_path", type=click.Path(file_okay=True, dir_okay=False))
@click.option("--task-type", default="detection", type=click.Choice(["classification", "detection", "segmentation", "panoptic", "tracking"]), help="Task type for the dataset")
@click.option("--file-format", default=None, type=click.Choice(["csv", "parquet"]), help="Output file format (default: inferred from output_path)")
def coco_to_df(coco_json: str, output_path: str, task_type: str, file_format: str) -> None:
    """Convert a COCO dataset to DataFrame format (CSV or Parquet).
    
    Args:
        coco_json: Path to the COCO JSON file.
        output_path: Path to the output CSV or Parquet file.
        task_type: Task type of the dataset.
        file_format: Output file format ('csv' or 'parquet').
    """
    dataset = CocoLoader.import_dataset(coco_json, task_type=task_type)
    DataframeLoader.export_dataset(dataset, output_path=output_path, file_format=file_format)
    click.echo(f"Converted COCO dataset from {coco_json} to DataFrame format at {output_path}")


@convert.command()
@click.argument("input_path", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("output_path", type=click.Path(file_okay=True, dir_okay=False))
@click.option("--task-type", default="detection", type=click.Choice(["classification", "detection", "segmentation", "panoptic", "tracking"]), help="Task type for the dataset")
@click.option("--file-format", default=None, type=click.Choice(["csv", "parquet"]), help="Input file format (default: inferred from input_path)")
def df_to_coco(input_path: str, output_path: str, task_type: str, file_format: str) -> None:
    """Convert a DataFrame dataset (CSV or Parquet) to COCO format.
    
    Args:
        input_path: Path to the input CSV or Parquet file.
        output_path: Path to the output COCO JSON file.
        task_type: Task type of the dataset.
        file_format: Input file format ('csv' or 'parquet').
    """
    dataset = DataframeLoader.import_dataset(input_path, task_type=task_type, file_format=file_format)
    CocoLoader.export_dataset(dataset, output_path=output_path)
    click.echo(f"Converted DataFrame dataset from {input_path} to COCO format at {output_path}")


@convert.command()
@click.option("--coco", "coco_paths", multiple=True, default=[], help="COCO dataset paths to merge")
@click.option("--yolo", "yolo_paths", multiple=True, default=[], help="YOLO dataset root paths to merge")
@click.option("--df", "df_paths", multiple=True, default=[], help="DataFrame (CSV/Parquet) paths to merge")
@click.option("--task-type", default="detection", type=click.Choice(["classification", "detection", "segmentation", "panoptic", "tracking"]), help="Task type for the datasets")
@click.option("--format", "output_format", default="coco", type=click.Choice(["coco", "yolo", "df"]), help="Output format")
@click.option("--output", "output_path", required=True, type=click.Path(file_okay=True, dir_okay=True), help="Output path")
@click.option("--file-format", default=None, type=click.Choice(["csv", "parquet"]), help="DataFrame file format (for --df output or input)")
def merge(coco_paths: tuple, yolo_paths: tuple, df_paths: tuple, task_type: str, output_format: str, output_path: str, file_format: str) -> None:
    """Merge multiple datasets from different formats into a single dataset.
    
    Example:
        convert merge --coco path1 path2 --yolo path3 path4 --df path5 --format coco --output merged.json
    
    Args:
        coco_paths: Paths to COCO JSON files.
        yolo_paths: Paths to YOLO dataset root directories.
        df_paths: Paths to DataFrame files (CSV or Parquet).
        task_type: Task type for all datasets.
        output_format: Output format ('coco', 'yolo', or 'df').
        output_path: Output path.
        file_format: File format for DataFrame input/output ('csv' or 'parquet').
    """
    from cvsdk.model import Dataset, Image
    
    # Check if at least one dataset is provided
    if not coco_paths and not yolo_paths and not df_paths:
        click.echo("Error: At least one dataset must be provided (--coco, --yolo, or --df)", err=True)
        return
    
    # Collect all images and categories from all datasets
    all_images = []
    all_categories = {}
    image_id_offset = 0
    
    # Import COCO datasets
    for coco_path in coco_paths:
        dataset = CocoLoader.import_dataset(coco_path, task_type=task_type)
        # Remap image IDs to avoid conflicts
        for img in dataset.images:
            img.id += image_id_offset
        all_images.extend(dataset.images)
        # Merge categories (by id)
        for cat_id, cat_name in dataset.categories.items():
            if cat_id not in all_categories:
                all_categories[cat_id] = cat_name
        image_id_offset = max(img.id for img in all_images) + 1
    
    # Import YOLO datasets
    for yolo_path in yolo_paths:
        dataset = YOLOLoader.import_dataset(yolo_path, task_type=task_type)
        # Remap image IDs to avoid conflicts
        for img in dataset.images:
            img.id += image_id_offset
        all_images.extend(dataset.images)
        # Merge categories (by id)
        for cat_id, cat_name in dataset.categories.items():
            if cat_id not in all_categories:
                all_categories[cat_id] = cat_name
        image_id_offset = max(img.id for img in all_images) + 1
    
    # Import DataFrame datasets
    for df_path in df_paths:
        dataset = DataframeLoader.import_dataset(df_path, task_type=task_type, file_format=file_format)
        # Remap image IDs to avoid conflicts
        for img in dataset.images:
            img.id += image_id_offset
        all_images.extend(dataset.images)
        # Merge categories (by id)
        for cat_id, cat_name in dataset.categories.items():
            if cat_id not in all_categories:
                all_categories[cat_id] = cat_name
        image_id_offset = max(img.id for img in all_images) + 1
    
    # Create merged dataset
    merged_dataset = Dataset(
        images=all_images,
        categories=all_categories,
        task_type=task_type
    )
    
    # Export to the desired format
    if output_format == "coco":
        CocoLoader.export_dataset(merged_dataset, output_path=output_path)
        click.echo(f"Merged {len(coco_paths) + len(yolo_paths) + len(df_paths)} datasets to COCO format at {output_path}")
    elif output_format == "yolo":
        YOLOLoader.export_dataset(merged_dataset, output_path=output_path)
        click.echo(f"Merged {len(coco_paths) + len(yolo_paths) + len(df_paths)} datasets to YOLO format at {output_path}")
    elif output_format == "df":
        DataframeLoader.export_dataset(merged_dataset, output_path=output_path, file_format=file_format)
        click.echo(f"Merged {len(coco_paths) + len(yolo_paths) + len(df_paths)} datasets to DataFrame format at {output_path}")
