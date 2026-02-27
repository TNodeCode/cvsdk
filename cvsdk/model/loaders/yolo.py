import os
from typing import List
from pathlib import Path
from cvsdk.model import Image, BoundingBox, SegmentationMask, PanopticSegment, Dataset


class YOLOLoader:
    """YOLO dataset import and export.
    """
    @staticmethod
    def import_dataset(yolo_root: str, task_type: str) -> Dataset:
        """
        Import a YOLO-format dataset into the Dataset Pydantic model.

        Args:
            yolo_root (str): Path to YOLO dataset root.
            task_type (str): Task type: 'classification', 'detection', 'segmentation', 'panoptic', 'tracking'.

        Returns:
            Dataset: Parsed dataset object.
        """
        images = []
        categories = {}
        image_id = 0

        image_dir = Path(yolo_root) / "images"
        label_dir = Path(yolo_root) / "labels"
        mask_dir = Path(yolo_root) / "masks" if task_type == "panoptic" else None

        for split in ['train', 'val', 'test']:
            for img_path in (image_dir / split).glob("*.*"):
                file_name = str(img_path.relative_to(yolo_root))
                label_path = label_dir / split / (img_path.stem + ".txt")

                image = Image(
                    id=image_id,
                    file_name=file_name,
                    width=0,  # You could use PIL.Image.open(img_path).size here
                    height=0,
                )

                if label_path.exists():
                    with open(label_path) as f:
                        for line in f:
                            parts = list(map(float, line.strip().split()))
                            class_id = int(parts[0])
                            categories[class_id] = categories.get(class_id, f"class_{class_id}")

                            if task_type == "classification":
                                image.labels.append(class_id)

                            elif task_type == "detection":
                                cx, cy, w, h = parts[1:]
                                xmin = cx - w / 2
                                ymin = cy - h / 2
                                image.bounding_boxes.append(
                                    BoundingBox(
                                        xmin=xmin,
                                        ymin=ymin,
                                        width=w,
                                        height=h,
                                        category_id=class_id
                                    )
                                )

                            elif task_type == "segmentation":
                                cx, cy, w, h, *seg = parts[1:]
                                image.bounding_boxes.append(
                                    BoundingBox(
                                        xmin=cx - w / 2,
                                        ymin=cy - h / 2,
                                        width=w,
                                        height=h,
                                        category_id=class_id
                                    )
                                )
                                polygons = [seg]  # In YOLO-seg each object has one polygon
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
                                xmin = cx - w / 2
                                ymin = cy - h / 2
                                image.bounding_boxes.append(
                                    BoundingBox(
                                        xmin=xmin,
                                        ymin=ymin,
                                        width=w,
                                        height=h,
                                        category_id=class_id,
                                        id=track_id
                                    )
                                )

                images.append(image)
                image_id += 1

        return Dataset(images=images, categories=categories, task_type=task_type)

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