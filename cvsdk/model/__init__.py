from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Dict, Any, Self
import json


class BoundingBox(BaseModel):
    """Represents a bounding box in an image."""
    xmin: float = Field(..., ge=0)
    ymin: float = Field(..., ge=0)
    width: float = Field(..., ge=0)
    height: float = Field(..., ge=0)
    category_id: int
    id: int | None = None

    @field_validator('xmin', 'ymin', 'width', 'height')
    def check_is_integer(cls, v: int) -> int:
        """Check if value is integer.

        Args:
            v (int): value

        Raises:
            ValueError: raised when value is not an integer

        Returns:
            int: value
        """
        if not float(v).is_integer():
            raise ValueError("Bounding box coordinates must be integers")
        return v


class SegmentationMask(BaseModel):
    """Represents a segmentation mask as a list of polygons."""
    segmentation: list[list[float]]
    category_id: int
    id: int | None = None

    @field_validator('segmentation')
    def check_polygon_coords(cls, polygon: list[float]) -> list[float]:
        """Check polygon coords.

        Args:
            polygon (list[float]): polygon coordinates

        Returns:
            list[float]: polygon coordinates
        """
        if len(polygon) % 2 != 0:
            raise ValueError("Polygon coordinates must be in pairs of (x, y)")
        if not all(float(x).is_integer() for x in polygon):
            raise ValueError("Segmentation polygon coordinates must be integers")
        return polygon


class PanopticSegment(BaseModel):
    """Represents a panoptic segmentation mask for an image."""
    segment_id: int
    category_id: int
    mask: str  # Path to the segmentation mask file


class Image(BaseModel):
    """Represents an image in the dataset."""
    id: int
    file_name: str
    width: int
    height: int
    labels: List[int] = []  # Image classification labels (category IDs)
    stack: str = "main_stack"
    bounding_boxes: List[BoundingBox] = []
    segmentation_masks: List[SegmentationMask] = []
    panoptic_segments: List[PanopticSegment] = []

    @model_validator(mode='after')
    @classmethod
    def check_dimensions_and_coords(cls, values: Self) -> Self:
        """Check box and segmentation mask coordinates against image dimensions.

        Args:
            values (Self): Image object

        Returns:
            Self: itself
        """
        width = values.width
        height = values.height

        # Check image dimensions

        if width <= 0 or height <= 0:
            raise ValueError(f"Image dimensions must be greater than zero (got {width}x{height})")

        # Check bounding box limits
        for box in values.bounding_boxes:
            if box.xmin + box.width > width or box.ymin + box.height > height:
                raise ValueError(f"Bounding box {box} exceeds image dimensions {width}x{height}")

        # Check segmentation polygon limits
        for mask in values.segmentation_masks:
            for polygon in mask.segmentation:
                xs = polygon[::2]
                ys = polygon[1::2]
                if any(x >= width or x < 0 for x in xs) or any(y >= height or y < 0 for y in ys):
                    raise ValueError(f"Segmentation polygon {polygon} is out of bounds for image {width}x{height}")
        return values


class Dataset(BaseModel):
    """Represents an entire dataset."""
    images: list[Image]
    categories: dict[int, str]
    task_type: str  # "detection", "segmentation", "panoptic", "classification"

    @model_validator(mode='after')
    @classmethod
    def validate_dataset(cls, values: Self) -> Self:
        """Validate dataset.

        Args:
            values (Self): Dataset object

        Returns:
            Self: itself
        """
        images = values.images
        categories = values.categories

        image_ids = set()
        for image in images:
            if image.id in image_ids:
                raise ValueError(f"Duplicate image id found: {image.id}")
            image_ids.add(image.id)

            for label in image.labels:
                if label not in categories:
                    raise ValueError(f"Invalid image classification label {label} in image {image.id}")

            for box in image.bounding_boxes:
                if box.category_id not in categories:
                    raise ValueError(f"Invalid category_id {box.category_id} in bounding box in image {image.id}")

            for mask in image.segmentation_masks:
                if mask.category_id not in categories:
                    raise ValueError(f"Invalid category_id {mask.category_id} in segmentation mask in image {image.id}")

            for segment in image.panoptic_segments:
                if segment.category_id not in categories:
                    raise ValueError(f"Invalid category_id {segment.category_id} in panoptic segment in image {image.id}")

        return values


    def get_stacks(self) -> set[str]:
        """Get all stacks.

        Returns:
            set[str]: All stack names
        """
        return set([image.stack for image in self.images if image.stack is not None])