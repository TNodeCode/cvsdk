import re
from typing import Any, Dict, List, Optional, Type
import pandas as pd
from cvsdk.model import Dataset, Image, BoundingBox, SegmentationMask, PanopticSegment
from pydantic import BaseModel


class PandasLoader:
    """Utility class for exporting a Dataset Pydantic model to a pandas DataFrame and importing from a DataFrame back to the Pydantic Dataset model."""

    @staticmethod
    def export_dataset(dataset: Dataset) -> pd.DataFrame:
        """Convert a Dataset model into a pandas DataFrame.

        Each image is one row; bounding boxes, segmentation masks,
        and panoptic segments are flattened into separate columns.
        """
        rows: list[dict[str, Any]] = []

        for image in dataset.images:
            row: dict[str, Any] = {
                'id': image.id,
                'file_name': image.file_name,
                'width': image.width,
                'height': image.height,
                'labels': image.labels,
                'stack': image.stack,
            }

            # Flatten bounding boxes
            for i, box in enumerate(image.bounding_boxes, start=1):
                prefix = f'bbox_{i}'
                row[f'{prefix}_xmin'] = box.xmin
                row[f'{prefix}_ymin'] = box.ymin
                row[f'{prefix}_width'] = box.width
                row[f'{prefix}_height'] = box.height
                row[f'{prefix}_category_id'] = box.category_id
                row[f'{prefix}_id'] = box.id

            # Flatten segmentation masks
            for j, mask in enumerate(image.segmentation_masks, start=1):
                prefix = f'segmentation_{j}'
                # store as list of floats
                row[prefix] = mask.segmentation
                row[f'{prefix}_category_id'] = mask.category_id
                row[f'{prefix}_id'] = mask.id

            # Flatten panoptic segments
            for k, seg in enumerate(image.panoptic_segments, start=1):
                prefix = f'panoptic_{k}'
                row[f'{prefix}_segment_id'] = seg.segment_id
                row[f'{prefix}_category_id'] = seg.category_id
                row[f'{prefix}_mask'] = seg.mask

            rows.append(row)

        df = pd.DataFrame(rows)
        return df

    @staticmethod
    def import_dataset(
        df: pd.DataFrame,
        categories: dict[int, str],
        task_type: str,
        image_model: Image,
        bbox_model: BoundingBox,
        seg_model: SegmentationMask,
        panoptic_model: PanopticSegment,
    ) -> BaseModel:
        """Convert a pandas DataFrame back into a Dataset Pydantic model.

        Parameters:
            df: DataFrame produced by `export_dataset`.
            categories: category mapping used for the Dataset.
            task_type: one of 'detection', 'segmentation', 'panoptic', 'classification'.
            image_model: the Pydantic Image model class.
            bbox_model: the Pydantic BoundingBox model class.
            seg_model: the Pydantic SegmentationMask model class.
            panoptic_model: the Pydantic PanopticSegment model class.

        Returns:
            A Dataset model instance.
        """
        images = []

        # Detect dynamic columns
        bbox_pattern = re.compile(r'bbox_(\d+)_(xmin|ymin|width|height|category_id|id)')
        seg_pattern = re.compile(r'segmentation_(\d+)(_category_id|_id)?$')
        pan_pattern = re.compile(r'panoptic_(\d+)_(segment_id|category_id|mask)')

        for _, row in df.iterrows():
            # Base image fields
            img_kwargs: dict[str, Any] = {
                'id': int(row['id']),
                'file_name': row['file_name'],
                'width': int(row['width']),
                'height': int(row['height']),
                'labels': list(row.get('labels') or []),
                'stack': row.get('stack', 'main_stack'),
            }

            # Collect bounding boxes
            bboxes = {}
            for col in df.columns:
                m = bbox_pattern.match(col)
                if m and not pd.isna(row[col]):
                    idx, field = m.groups()
                    bboxes.setdefault(idx, {})[field] = row[col]

            bbox_list = []
            for idx in sorted(bboxes, key=int):
                kwargs = bboxes[idx]
                # Rename fields if necessary
                bbox_list.append(bbox_model(**kwargs))

            # Collect segmentation masks
            segs = {}
            for col in df.columns:
                m = seg_pattern.match(col)
                if m and not pd.isna(row[col]):
                    idx, suffix = m.groups()
                    suffix = suffix or ''
                    key = suffix.lstrip('_') or 'segmentation'
                    segs.setdefault(idx, {})[key] = row[col]

            seg_list = []
            for idx in sorted(segs, key=int):
                mask_kwargs = segs[idx]
                seg_list.append(seg_model(**mask_kwargs))

            # Collect panoptic segments
            pan_segs = {}
            for col in df.columns:
                m = pan_pattern.match(col)
                if m and not pd.isna(row[col]):
                    idx, field = m.groups()
                    pan_segs.setdefault(idx, {})[field] = row[col]

            pan_list = []
            for idx in sorted(pan_segs, key=int):
                pan_list.append(panoptic_model(**pan_segs[idx]))

            img_kwargs['bounding_boxes'] = bbox_list
            img_kwargs['segmentation_masks'] = seg_list
            img_kwargs['panoptic_segments'] = pan_list

            images.append(image_model(**img_kwargs))

        # Build and return the Dataset model
        return Dataset(images=images, categories=categories, task_type=task_type)
