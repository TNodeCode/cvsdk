import os
from typing import List, Optional
from pathlib import Path
from PIL import Image as PILImage
from cvsdk.model import Image, BoundingBox, SegmentationMask, PanopticSegment, Dataset


def _get_image_dimensions(img_path: Path) -> tuple:
    """Get image dimensions using PIL.
    
    Args:
        img_path: Path to the image file.
        
    Returns:
        Tuple of (width, height).
    """
    try:
        with PILImage.open(img_path) as img:
            return img.size  # (width, height)
    except Exception:
        return (0, 0)


class YOLOLoader:
    """YOLO dataset import and export.
    """
    @staticmethod
    def import_dataset(
        yolo_root: str,
        task_type: str,
        train_dir: Optional[str] = None,
        val_dir: Optional[str] = None,
        test_dir: Optional[str] = None
    ) -> Dataset:
        """
        Import a YOLO-format dataset into the Dataset Pydantic model.

        Args:
            yolo_root (str): Path to YOLO dataset root.
            task_type (str): Task type: 'classification', 'detection', 'segmentation', 'panoptic', 'tracking'.
            train_dir (Optional[str]): Custom directory name for training split. Defaults to 'train'.
            val_dir (Optional[str]): Custom directory name for validation split. Defaults to 'val'.
            test_dir (Optional[str]): Custom directory name for test split. Defaults to 'test'.

        Returns:
            Dataset: Parsed dataset object.
        """
        images = []
        categories = {}
        image_id = 0
        split_map = {}  # Maps image_id to split name (train/val/test)

        # Use custom directory names if provided, otherwise defaults
        split_dirs = {
            'train': train_dir if train_dir else 'train',
            'val': val_dir if val_dir else 'val',
            'test': test_dir if test_dir else 'test'
        }

        if task_type == "classification":
            # Classification: directory structure is root/train/class_name, root/val/class_name, etc.
            # No "images" or "labels" directories - class directories are directly under split directories
            for split, split_dir in split_dirs.items():
                split_path = Path(yolo_root) / split_dir
                if not split_path.exists():
                    continue
                
                # Each subdirectory represents a class
                for class_dir in split_path.iterdir():
                    if not class_dir.is_dir():
                        continue
                    
                    class_name = class_dir.name
                    # Use class index based on existing categories or assign new one
                    class_id = len(categories)
                    categories[class_id] = class_name
                    
                    # Get all image files in the class directory
                    for img_path in class_dir.glob("**/*.*"):
                        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']:
                            file_name = str(img_path.relative_to(yolo_root))
                            width, height = _get_image_dimensions(img_path)
                            
                            image = Image(
                                id=image_id,
                                file_name=file_name,
                                width=width,
                                height=height,
                            )
                            image.labels.append(class_id)
                            
                            images.append(image)
                            split_map[image_id] = split
                            image_id += 1
        else:
            # Detection, segmentation, panoptic, tracking: standard YOLO structure with images/labels
            image_dir = Path(yolo_root) / "images"
            label_dir = Path(yolo_root) / "labels"
            mask_dir = Path(yolo_root) / "masks" if task_type == "panoptic" else None

            for split, split_dir in split_dirs.items():
                split_image_dir = image_dir / split_dir
                if not split_image_dir.exists():
                    continue
                    
                for img_path in split_image_dir.glob("**/*.*"):
                    file_name = str(img_path.relative_to(yolo_root))
                    # Get the relative path from the split image directory to preserve subdirectories
                    rel_path = img_path.relative_to(split_image_dir)
                    label_path = label_dir / split_dir / rel_path.with_suffix(".txt")
                    width, height = _get_image_dimensions(img_path)

                    try:
                        image = Image(
                            id=image_id,
                            file_name=file_name,
                            width=width,
                            height=height,
                        )
                    except:
                        continue

                    if label_path.exists():
                        with open(label_path) as f:
                            for line in f:
                                parts = list(map(float, line.strip().split()))
                                class_id = int(parts[0])
                                categories[class_id] = categories.get(class_id, f"class_{class_id}")

                                if task_type == "detection":
                                    cx, cy, w, h = parts[1:]
                                    # Convert normalized YOLO coordinates to pixel coordinates
                                    xmin = int((cx - w / 2) * width)
                                    ymin = int((cy - h / 2) * height)
                                    box_width = int(w * width)
                                    box_height = int(h * height)
                                    # Clamp coordinates to not exceed image dimensions
                                    xmin = max(0, min(xmin, width - 1))
                                    ymin = max(0, min(ymin, height - 1))
                                    box_width = max(1, min(box_width, width - xmin))
                                    box_height = max(1, min(box_height, height - ymin))
                                    image.bounding_boxes.append(
                                        BoundingBox(
                                            xmin=xmin,
                                            ymin=ymin,
                                            width=box_width,
                                            height=box_height,
                                            category_id=class_id
                                        )
                                    )

                                elif task_type == "segmentation":
                                    cx, cy, w, h, *seg = parts[1:]
                                    # Convert normalized YOLO coordinates to pixel coordinates
                                    xmin = int((cx - w / 2) * width)
                                    ymin = int((cy - h / 2) * height)
                                    box_width = int(w * width)
                                    box_height = int(h * height)
                                    # Clamp coordinates to not exceed image dimensions
                                    xmin = max(0, min(xmin, width - 1))
                                    ymin = max(0, min(ymin, height - 1))
                                    box_width = max(1, min(box_width, width - xmin))
                                    box_height = max(1, min(box_height, height - ymin))
                                    image.bounding_boxes.append(
                                        BoundingBox(
                                            xmin=xmin,
                                            ymin=ymin,
                                            width=box_width,
                                            height=box_height,
                                            category_id=class_id
                                        )
                                    )
                                    # Convert normalized polygon coordinates to pixel coordinates
                                    # YOLO-seg format: cx, cy, w, h, x1, y1, x2, y2, ...
                                    pixel_seg = []
                                    for coord in seg:
                                        # Alternate between x (multiply by width) and y (multiply by height)
                                        idx = len(pixel_seg)
                                        if idx % 2 == 0:
                                            pixel_seg.append(int(coord * width))
                                        else:
                                            pixel_seg.append(int(coord * height))
                                    polygons = [pixel_seg]
                                    image.segmentation_masks.append(
                                        SegmentationMask(
                                            segmentation=polygons,
                                            category_id=class_id
                                        )
                                    )

                                elif task_type == "panoptic":
                                    # For panoptic, polygon segmentation and mask paths are in separate files
                                    # Could be something like: `class_id mask_path`
                                    mask_path = parts[1]
                                    image.panoptic_segments.append(
                                        PanopticSegment(
                                            segment_id=len(image.panoptic_segments),
                                            category_id=class_id,
                                            mask=mask_path
                                        )
                                    )

                                elif task_type == "tracking":
                                    # YOLO tracking: class_id track_id cx cy w h
                                    track_id = int(parts[1])
                                    cx, cy, w, h = parts[2:]
                                    # Convert normalized YOLO coordinates to pixel coordinates
                                    xmin = int((cx - w / 2) * width)
                                    ymin = int((cy - h / 2) * height)
                                    box_width = int(w * width)
                                    box_height = int(h * height)
                                    # Clamp coordinates to not exceed image dimensions
                                    xmin = max(0, min(xmin, width - 1))
                                    ymin = max(0, min(ymin, height - 1))
                                    box_width = max(1, min(box_width, width - xmin))
                                    box_height = max(1, min(box_height, height - ymin))
                                    image.bounding_boxes.append(
                                        BoundingBox(
                                            xmin=xmin,
                                            ymin=ymin,
                                            width=box_width,
                                            height=box_height,
                                            category_id=class_id,
                                            id=track_id
                                        )
                                    )

                    images.append(image)
                    split_map[image_id] = split
                    image_id += 1

        return Dataset(images=images, categories=categories, task_type=task_type, split_map=split_map if split_map else None)

    @staticmethod
    def export_dataset(dataset: Dataset, output_dir: str):
        """
        Export the dataset into YOLO-format files.

        Args:
            dataset (Dataset): Dataset to export.
            output_dir (str): Target root directory for YOLO format.
        """
        output_dir = Path(output_dir)
        label_dir = output_dir / "labels/train"
        image_dir = output_dir / "images/train"
        mask_dir = output_dir / "masks/train"

        label_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        if dataset.task_type == "panoptic":
            mask_dir.mkdir(parents=True, exist_ok=True)

        for image in dataset.images:
            img_name = Path(image.file_name).name
            label_file = label_dir / (Path(img_name).stem + ".txt")

            lines = []

            if dataset.task_type == "classification":
                for class_id in image.labels:
                    lines.append(f"{class_id}")

            elif dataset.task_type == "detection":
                for box in image.bounding_boxes:
                    cx = (box.xmin + box.width / 2) / image.width
                    cy = (box.ymin + box.height / 2) / image.height
                    w = box.width / image.width
                    h = box.height / image.height
                    lines.append(f"{box.category_id} {cx} {cy} {w} {h}")

            elif dataset.task_type == "segmentation":
                for mask in image.segmentation_masks:
                    # Flatten the polygon (normalized coordinates)
                    flat_seg = " ".join(str(p / image.width if i % 2 == 0 else p / image.height for i, p in enumerate(mask.segmentation[0])))
                    # Find the corresponding bbox
                    bbox = next(
                        (b for b in image.bounding_boxes if b.category_id == mask.category_id),
                        None
                    )
                    if bbox:
                        cx = (bbox.xmin + bbox.width / 2) / image.width
                        cy = (bbox.ymin + bbox.height / 2) / image.height
                        w = bbox.width / image.width
                        h = bbox.height / image.height
                        lines.append(f"{mask.category_id} {cx} {cy} {w} {h} {flat_seg}")

            elif dataset.task_type == "panoptic":
                for segment in image.panoptic_segments:
                    lines.append(f"{segment.category_id} {segment.mask}")

            elif dataset.task_type == "tracking":
                for box in image.bounding_boxes:
                    cx = (box.xmin + box.width / 2) / image.width
                    cy = (box.ymin + box.height / 2) / image.height
                    w = box.width / image.width
                    h = box.height / image.height
                    lines.append(f"{box.category_id} {box.id} {cx} {cy} {w} {h}")

            with open(label_file, "w") as f:
                f.write("\n".join(lines))