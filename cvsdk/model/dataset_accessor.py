"""Pandas DataFrame accessor for computer vision dataset statistics."""
import pandas as pd
from typing import Optional


@pd.api.extensions.register_dataframe_accessor("dataset")
class DatasetAccessor:
    """Pandas DataFrame accessor for computer vision dataset statistics.

    Provides methods for computing class distribution statistics on datasets
    stored in DataFrame format.

    Usage::

        import cvsdk.model.dataset_accessor  # noqa: F401 – registers the accessor
        df.dataset.class_counts()
        df.dataset.objects_per_class_per_image()
    """

    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        self._obj = pandas_obj

    def class_counts(
        self,
        class_col: str = "class_id",
        split_col: Optional[str] = "split",
        class_names: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Count the number of samples per class, optionally grouped by split.

        Each row of the input DataFrame represents one sample (one image for
        classification/segmentation datasets, or one bounding-box annotation
        for detection datasets).

        Args:
            class_col: Column name for the class identifier.  For classification
                datasets this is ``'class'``; for detection/segmentation it is
                ``'class_id'``.
            split_col: Column name for the dataset split (``train``/``val``/
                ``test``).  When ``None`` or the column is absent all rows are
                treated as a single group.
            class_names: Optional mapping from integer class id to class name.
                When provided a human-readable ``class_name`` column is added
                before grouping.

        Returns:
            pd.DataFrame: Pivot table with splits as columns and class names as
            the index; values are sample counts.
        """
        df = self._obj.copy()

        if class_names is not None:
            df[class_col] = df[class_col].map(class_names).fillna(df[class_col].astype(str))

        if split_col and split_col in df.columns:
            counts = (
                df.groupby([class_col, split_col])
                .size()
                .reset_index(name="count")
            )
            result = (
                counts.pivot(index=class_col, columns=split_col, values="count")
                .fillna(0)
                .astype(int)
            )
        else:
            counts = df.groupby(class_col).size().reset_index(name="count")
            result = counts.set_index(class_col)

        result.index.name = "class"
        result.columns.name = None
        return result

    def objects_per_class_per_image(
        self,
        class_col: str = "class_id",
        image_col: str = "image",
        split_col: Optional[str] = "split",
        class_names: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Compute mean and variance of object counts per class per image.

        This is meaningful for object detection datasets where each row
        represents a single bounding-box annotation.  The method first counts
        how many boxes of each class appear in each image, then aggregates those
        per-image counts across all images.

        Args:
            class_col: Column name for the class identifier.
            image_col: Column name for the image path/identifier.
            split_col: Column name for the dataset split.
            class_names: Optional mapping from integer class id to class name.

        Returns:
            pd.DataFrame: DataFrame with ``class``, optionally ``split``,
            ``mean_objects`` and ``var_objects`` columns.
        """
        df = self._obj.copy()

        if class_names is not None:
            df[class_col] = df[class_col].map(class_names).fillna(df[class_col].astype(str))

        group_keys = [image_col, class_col]
        if split_col and split_col in df.columns:
            group_keys.append(split_col)

        per_image_counts = (
            df.groupby(group_keys, observed=True).size().reset_index(name="count")
        )

        agg_keys = [class_col]
        if split_col and split_col in df.columns:
            agg_keys.append(split_col)

        result = (
            per_image_counts.groupby(agg_keys)["count"]
            .agg(mean_objects="mean", var_objects="var")
            .reset_index()
        )

        result["mean_objects"] = result["mean_objects"].round(2)
        result["var_objects"] = result["var_objects"].round(2).fillna(0.0)
        result.rename(columns={class_col: "class"}, inplace=True)
        return result
