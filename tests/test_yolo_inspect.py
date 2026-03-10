"""Tests for the yolo inspect dataset command and the DatasetAccessor."""
import pytest
import pandas as pd
from click.testing import CliRunner

import cvsdk.model.dataset_accessor  # noqa: F401 – registers df.dataset accessor
from cvsdk.model import Dataset, Image, BoundingBox, SegmentationMask
from cvsdk.yolo.inspect import _dataset_to_df, _print_class_counts, _print_objects_per_class
from cvsdk.yolo.cli import yolo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def detection_dataset():
    """Detection dataset with two splits (train/val) and two classes."""
    return Dataset(
        images=[
            Image(
                id=1,
                file_name="train/img1.jpg",
                width=512,
                height=512,
                bounding_boxes=[
                    BoundingBox(xmin=10, ymin=10, width=50, height=50, category_id=0),
                    BoundingBox(xmin=100, ymin=100, width=60, height=60, category_id=1),
                    BoundingBox(xmin=200, ymin=200, width=40, height=40, category_id=0),
                ],
            ),
            Image(
                id=2,
                file_name="train/img2.jpg",
                width=512,
                height=512,
                bounding_boxes=[
                    BoundingBox(xmin=10, ymin=10, width=50, height=50, category_id=1),
                ],
            ),
            Image(
                id=3,
                file_name="val/img3.jpg",
                width=512,
                height=512,
                bounding_boxes=[
                    BoundingBox(xmin=10, ymin=10, width=50, height=50, category_id=0),
                ],
            ),
        ],
        categories={0: "cat", 1: "dog"},
        task_type="detection",
        split_map={1: "train", 2: "train", 3: "val"},
    )


@pytest.fixture
def classification_dataset():
    """Classification dataset with train/val splits and two classes."""
    return Dataset(
        images=[
            Image(id=1, file_name="train/cat/img1.jpg", width=224, height=224, labels=[0]),
            Image(id=2, file_name="train/cat/img2.jpg", width=224, height=224, labels=[0]),
            Image(id=3, file_name="train/dog/img3.jpg", width=224, height=224, labels=[1]),
            Image(id=4, file_name="val/cat/img4.jpg", width=224, height=224, labels=[0]),
        ],
        categories={0: "cat", 1: "dog"},
        task_type="classification",
        split_map={1: "train", 2: "train", 3: "train", 4: "val"},
    )


@pytest.fixture
def segmentation_dataset():
    """Segmentation dataset with a single split.

    ``model_construct`` is used for ``SegmentationMask`` to bypass a
    pre-existing parity check in the field validator that rejects masks with
    an odd number of sub-polygons (e.g. a single polygon).
    """
    return Dataset(
        images=[
            Image(
                id=1,
                file_name="train/img1.jpg",
                width=512,
                height=512,
                segmentation_masks=[
                    SegmentationMask.model_construct(
                        segmentation=[[10, 10, 60, 10, 60, 60, 10, 60]], category_id=0
                    ),
                    SegmentationMask.model_construct(
                        segmentation=[[100, 100, 150, 100, 150, 150, 100, 150]], category_id=1
                    ),
                ],
            ),
            Image(
                id=2,
                file_name="val/img2.jpg",
                width=512,
                height=512,
                segmentation_masks=[
                    SegmentationMask.model_construct(
                        segmentation=[[20, 20, 80, 20, 80, 80, 20, 80]], category_id=0
                    ),
                ],
            ),
        ],
        categories={0: "cat", 1: "dog"},
        task_type="segmentation",
        split_map={1: "train", 2: "val"},
    )


# ---------------------------------------------------------------------------
# Tests for _dataset_to_df
# ---------------------------------------------------------------------------

class TestDatasetToDf:
    def test_detection_columns(self, detection_dataset):
        df = _dataset_to_df(detection_dataset)
        for col in ("image", "x_min", "y_min", "x_max", "y_max", "class_id", "split"):
            assert col in df.columns

    def test_detection_row_count(self, detection_dataset):
        df = _dataset_to_df(detection_dataset)
        assert len(df) == 5  # 3 boxes in img1 + 1 in img2 + 1 in img3

    def test_detection_split_values(self, detection_dataset):
        df = _dataset_to_df(detection_dataset)
        assert set(df["split"].unique()) == {"train", "val"}

    def test_classification_columns(self, classification_dataset):
        df = _dataset_to_df(classification_dataset)
        for col in ("image", "class_id", "class", "split"):
            assert col in df.columns

    def test_classification_row_count(self, classification_dataset):
        df = _dataset_to_df(classification_dataset)
        assert len(df) == 4

    def test_segmentation_columns(self, segmentation_dataset):
        df = _dataset_to_df(segmentation_dataset)
        for col in ("image", "class_id", "split"):
            assert col in df.columns

    def test_segmentation_row_count(self, segmentation_dataset):
        df = _dataset_to_df(segmentation_dataset)
        assert len(df) == 3  # 2 masks in img1 + 1 in img2

    def test_missing_split_map(self):
        ds = Dataset(
            images=[
                Image(id=1, file_name="img.jpg", width=100, height=100, labels=[0]),
            ],
            categories={0: "cat"},
            task_type="classification",
            split_map=None,
        )
        df = _dataset_to_df(ds)
        assert df["split"].iloc[0] == "unknown"


# ---------------------------------------------------------------------------
# Tests for DatasetAccessor.class_counts
# ---------------------------------------------------------------------------

class TestClassCounts:
    def test_detection_class_counts_with_split(self, detection_dataset):
        df = _dataset_to_df(detection_dataset)
        counts = df.dataset.class_counts(class_col="class_id", class_names=detection_dataset.categories)
        # Two classes: cat (id=0) and dog (id=1)
        assert "cat" in counts.index
        assert "dog" in counts.index

    def test_detection_class_counts_splits_are_columns(self, detection_dataset):
        df = _dataset_to_df(detection_dataset)
        counts = df.dataset.class_counts(class_col="class_id", class_names=detection_dataset.categories)
        assert "train" in counts.columns
        assert "val" in counts.columns

    def test_detection_class_counts_values(self, detection_dataset):
        df = _dataset_to_df(detection_dataset)
        counts = df.dataset.class_counts(class_col="class_id", class_names=detection_dataset.categories)
        # cat appears 2 times in train (img1) + 1 in val (img3) = 3 total
        assert counts.loc["cat", "train"] == 2
        assert counts.loc["cat", "val"] == 1
        # dog appears 1 time in train (img1) + 1 in train (img2) = 2 in train
        assert counts.loc["dog", "train"] == 2

    def test_classification_class_counts(self, classification_dataset):
        df = _dataset_to_df(classification_dataset)
        counts = df.dataset.class_counts(class_col="class")
        assert counts.index.name == "class"

    def test_no_split_column(self, detection_dataset):
        df = _dataset_to_df(detection_dataset).drop(columns=["split"])
        counts = df.dataset.class_counts(class_col="class_id", class_names=detection_dataset.categories)
        assert "train" not in counts.columns
        assert "val" not in counts.columns
        assert "count" in counts.columns


# ---------------------------------------------------------------------------
# Tests for DatasetAccessor.objects_per_class_per_image
# ---------------------------------------------------------------------------

class TestObjectsPerClassPerImage:
    def test_returns_expected_columns(self, detection_dataset):
        df = _dataset_to_df(detection_dataset)
        stats = df.dataset.objects_per_class_per_image(
            class_col="class_id", class_names=detection_dataset.categories
        )
        for col in ("class", "mean_objects", "var_objects"):
            assert col in stats.columns

    def test_mean_objects_cat_train(self, detection_dataset):
        df = _dataset_to_df(detection_dataset)
        stats = df.dataset.objects_per_class_per_image(
            class_col="class_id", class_names=detection_dataset.categories
        )
        cat_train = stats[(stats["class"] == "cat") & (stats["split"] == "train")]
        # img1 has 2 cat boxes (only image with cat in train) → mean = 2.0
        assert cat_train["mean_objects"].iloc[0] == 2.0

    def test_variance_single_image(self, detection_dataset):
        df = _dataset_to_df(detection_dataset)
        stats = df.dataset.objects_per_class_per_image(
            class_col="class_id", class_names=detection_dataset.categories
        )
        # val split only has one image with cat → variance should be 0 (single observation)
        cat_val = stats[(stats["class"] == "cat") & (stats["split"] == "val")]
        assert cat_val["var_objects"].iloc[0] == 0.0


# ---------------------------------------------------------------------------
# Tests for CLI command
# ---------------------------------------------------------------------------

class TestInspectDatasetCLI:
    def test_inspect_group_registered(self):
        runner = CliRunner()
        result = runner.invoke(yolo, ["inspect", "--help"])
        assert result.exit_code == 0
        assert "inspect" in result.output.lower() or "dataset" in result.output.lower()

    def test_dataset_command_registered(self):
        runner = CliRunner()
        result = runner.invoke(yolo, ["inspect", "dataset", "--help"])
        assert result.exit_code == 0
        assert "--format" in result.output
        assert "--task-type" in result.output

    def test_dataset_command_with_dataframe_csv(self, tmp_path, detection_dataset):
        df = _dataset_to_df(detection_dataset)
        csv_path = tmp_path / "detections.csv"
        df.to_csv(csv_path, index=False)

        runner = CliRunner()
        result = runner.invoke(
            yolo,
            [
                "inspect",
                "dataset",
                "--data-path",
                str(csv_path),
                "--format",
                "dataframe",
                "--task-type",
                "detection",
                "--file-format",
                "csv",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_dataset_command_classification_csv(self, tmp_path, classification_dataset):
        df = _dataset_to_df(classification_dataset)
        csv_path = tmp_path / "classification.csv"
        df.to_csv(csv_path, index=False)

        runner = CliRunner()
        result = runner.invoke(
            yolo,
            [
                "inspect",
                "dataset",
                "--data-path",
                str(csv_path),
                "--format",
                "dataframe",
                "--task-type",
                "classification",
                "--file-format",
                "csv",
            ],
        )
        assert result.exit_code == 0, result.output
