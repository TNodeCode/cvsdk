"""Inspect sub-group for the YOLO CLI – dataset statistics."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
import pandas as pd
from rich.console import Console
from rich.table import Table

import cvsdk.model.dataset_accessor  # noqa: F401 – registers df.dataset accessor
from cvsdk.model import Dataset
from cvsdk.model.loaders.coco import CocoLoader
from cvsdk.model.loaders.dataframe import DataframeLoader
from cvsdk.model.loaders.yolo import YOLOLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dataset_to_df(dataset: Dataset) -> pd.DataFrame:
    """Convert a :class:`~cvsdk.model.Dataset` to a pandas DataFrame.

    A ``split`` column is added using the dataset's ``split_map`` when
    available; otherwise the value is set to ``'unknown'``.

    Args:
        dataset: Dataset to convert.

    Returns:
        pd.DataFrame: Flat DataFrame representation with a ``split`` column.
    """
    split_map: dict = dataset.split_map or {}

    if dataset.task_type == "classification":
        rows = []
        for image in dataset.images:
            split = split_map.get(image.id, "unknown")
            for label in image.labels:
                rows.append(
                    {
                        "image": image.file_name,
                        "class_id": label,
                        "class": dataset.categories.get(label, str(label)),
                        "width": image.width,
                        "height": image.height,
                        "split": split,
                    }
                )
        df = pd.DataFrame(rows)

    elif dataset.task_type == "detection":
        rows = []
        for image in dataset.images:
            split = split_map.get(image.id, "unknown")
            for box in image.bounding_boxes:
                rows.append(
                    {
                        "image": image.file_name,
                        "x_min": box.xmin,
                        "y_min": box.ymin,
                        "x_max": box.xmin + box.width,
                        "y_max": box.ymin + box.height,
                        "class_id": box.category_id,
                        "width": image.width,
                        "height": image.height,
                        "split": split,
                    }
                )
        df = pd.DataFrame(rows)

    elif dataset.task_type == "segmentation":
        rows = []
        for image in dataset.images:
            split = split_map.get(image.id, "unknown")
            for mask in image.segmentation_masks:
                rows.append(
                    {
                        "image": image.file_name,
                        "class_id": mask.category_id,
                        "width": image.width,
                        "height": image.height,
                        "split": split,
                    }
                )
        df = pd.DataFrame(rows)

    else:
        raise click.ClickException(
            f"Task type '{dataset.task_type}' is not supported by the inspect command. "
            "Use 'classification', 'detection' or 'segmentation'."
        )

    if not df.empty and "image" in df.columns:
        df["image"] = df["image"].astype("category")

    return df


def _load_raw_df(data_path: str, file_format: Optional[str]) -> pd.DataFrame:
    """Load a CSV or Parquet file into a pandas DataFrame.

    Args:
        data_path: Path to the file.
        file_format: Explicit format (``'csv'`` or ``'parquet'``).  When
            ``None`` the format is inferred from the file extension.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    path = Path(data_path)
    fmt = file_format or path.suffix.lstrip(".").lower()
    if fmt == "csv":
        return pd.read_csv(path)
    if fmt == "parquet":
        return pd.read_parquet(path)
    raise click.ClickException(
        f"Cannot infer file format from '{path.suffix}'. "
        "Use --file-format csv or --file-format parquet."
    )


# ---------------------------------------------------------------------------
# Rich table printers
# ---------------------------------------------------------------------------

def _print_class_counts(counts: pd.DataFrame, task_type: str) -> None:
    """Print class sample counts as a rich table.

    Args:
        counts: DataFrame with class names as the index and splits as columns.
        task_type: Task type string used in the table title.
    """
    console = Console()
    table = Table(
        title=f"Class Sample Counts ({task_type})",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Class", style="bold")
    for col in counts.columns:
        table.add_column(str(col).capitalize(), justify="right")
    table.add_column("Total", justify="right", style="bold green")

    for class_name, row in counts.iterrows():
        total = int(row.sum())
        table.add_row(
            str(class_name),
            *[str(int(v)) for v in row.values],
            str(total),
        )

    console.print(table)


def _print_objects_per_class(stats: pd.DataFrame, task_type: str) -> None:
    """Print mean/variance of objects per class per image as a rich table.

    Args:
        stats: DataFrame with ``class``, optionally ``split``,
            ``mean_objects`` and ``var_objects`` columns.
        task_type: Task type string used in the table title.
    """
    console = Console()
    table = Table(
        title=f"Objects per Class per Image ({task_type})",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Class", style="bold")
    has_split = "split" in stats.columns
    if has_split:
        table.add_column("Split", justify="center")
    table.add_column("Mean Objects", justify="right")
    table.add_column("Variance", justify="right")

    for _, row in stats.iterrows():
        cells = [str(row["class"])]
        if has_split:
            cells.append(str(row["split"]).capitalize())
        cells.extend([f"{row['mean_objects']:.2f}", f"{row['var_objects']:.2f}"])
        table.add_row(*cells)

    console.print(table)


# ---------------------------------------------------------------------------
# CLI group and commands
# ---------------------------------------------------------------------------

@click.group()
def inspect() -> None:
    """Commands for inspecting datasets."""
    pass


@inspect.command()
@click.option(
    "--data-path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the dataset (directory for yolo, JSON file for coco, CSV/Parquet for dataframe).",
)
@click.option(
    "--format",
    "fmt",
    required=True,
    type=click.Choice(["yolo", "coco", "dataframe"]),
    help="Annotation format of the dataset.",
)
@click.option(
    "--task-type",
    default="detection",
    show_default=True,
    type=click.Choice(["classification", "detection", "segmentation"]),
    help="Computer-vision task type.",
)
@click.option(
    "--train-dir",
    default=None,
    help="Custom training split directory name (YOLO format only).",
)
@click.option(
    "--val-dir",
    default=None,
    help="Custom validation split directory name (YOLO format only).",
)
@click.option(
    "--test-dir",
    default=None,
    help="Custom test split directory name (YOLO format only).",
)
@click.option(
    "--file-format",
    default=None,
    type=click.Choice(["csv", "parquet"]),
    help="Explicit file format for dataframe datasets (inferred from extension when omitted).",
)
def dataset(
    data_path: str,
    fmt: str,
    task_type: str,
    train_dir: Optional[str],
    val_dir: Optional[str],
    test_dir: Optional[str],
    file_format: Optional[str],
) -> None:
    """Inspect a dataset: compute class distribution across train/val/test splits.

    Supports classification, detection and segmentation datasets in YOLO, COCO
    or DataFrame (CSV/Parquet) annotation format.  YOLO and COCO datasets are
    automatically converted to a pandas DataFrame before analysis.

    For detection datasets the mean and variance of object counts per class per
    image are also reported.

    Examples::

        cv yolo inspect dataset --data-path ./data --format yolo --task-type detection
        cv yolo inspect dataset --data-path annotations.json --format coco --task-type segmentation
        cv yolo inspect dataset --data-path labels.csv --format dataframe --task-type classification
    """
    class_names: Optional[dict] = None

    if fmt == "yolo":
        ds = YOLOLoader.import_dataset(
            data_path,
            task_type=task_type,
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
        )
        df = _dataset_to_df(ds)
        class_names = ds.categories

    elif fmt == "coco":
        ds = CocoLoader.import_dataset(data_path, task_type=task_type)
        df = _dataset_to_df(ds)
        class_names = ds.categories

    else:  # dataframe
        df = _load_raw_df(data_path, file_format)

    # Resolve class column name (classification DataFrames use 'class', others use 'class_id')
    class_col = "class" if task_type == "classification" and "class" in df.columns else "class_id"

    counts = df.dataset.class_counts(class_col=class_col, class_names=class_names)
    _print_class_counts(counts, task_type)

    if task_type == "detection":
        stats = df.dataset.objects_per_class_per_image(
            class_col=class_col, class_names=class_names
        )
        _print_objects_per_class(stats, task_type)
