from cvsdk.model import Dataset, Image, BoundingBox, SegmentationMask, PanopticSegment
import json
import structlog


class CocoLoader:
    """Imports a COCO JSON file and converts it into the Dataset model."""

    @staticmethod
    def import_dataset(json_path: str, task_type: str) -> Dataset:
        """Import a COCO Dataset

        Args:
            json_path (str): path to the COCO json annotation file
            task_type (str): task type

        Returns:
            Dataset: _description_
        """
        with open(json_path, "r") as f:
            coco_data = json.load(f)

        categories = {c["id"]: c["name"] for c in coco_data["categories"]}

        images = {img["id"]: Image(
            id=img["id"],
            file_name=img["file_name"],
            width=img["width"],
            height=img["height"],
            bounding_boxes=[],
            segmentation_masks=[],
            panoptic_segments=[],
            labels=[]
        ) for img in coco_data["images"]}

        if task_type == "detection" or task_type == "segmentation":
            for ann in coco_data["annotations"]:
                if "bbox" in ann:
                    try:
                        bbox = BoundingBox(
                            xmin=ann["bbox"][0],
                            ymin=ann["bbox"][1],
                            width=ann["bbox"][2],
                            height=ann["bbox"][3],
                            category_id=ann["category_id"],
                            id=ann["id"]
                        )
                        images[ann["image_id"]].bounding_boxes.append(bbox)
                    except Exception as e:
                        structlog.get_logger().warning(f"Failed to parse bounding box: {e}")

                if "segmentation" in ann and ann["segmentation"]:
                    mask = SegmentationMask(
                        segmentation=ann["segmentation"],
                        category_id=ann["category_id"],
                        id=ann["id"]
                    )
                    images[ann["image_id"]].segmentation_masks.append(mask)

        elif task_type == "panoptic":
            for ann in coco_data["annotations"]:
                mask = PanopticSegment(
                    segment_id=ann["id"],
                    category_id=ann["category_id"],
                    mask=ann["segmentation"]  # Assuming this contains the path to a mask
                )
                images[ann["image_id"]].panoptic_segments.append(mask)

        elif task_type == "classification":
            for ann in coco_data["annotations"]:
                images[ann["image_id"]].labels.append(ann["category_id"])

        return Dataset(images=list(images.values()), categories=categories, task_type=task_type)

    @staticmethod
    def to_coco_dict(dataset: Dataset) -> dict:
        """Export dataset to COCO dictionary.

        Args:
            dataset (Dataset): dataset

        Returns:
            dict: COCO style dictionary
        """
        coco_dict = {
            "images": [
                {"id": img.id, "file_name": img.file_name, "width": img.width, "height": img.height}
                for img in dataset.images
            ],
            "annotations": [],
            "categories": [{"id": cat_id, "name": name} for cat_id, name in dataset.categories.items()]
        }

        annotation_id = 1
        for img in dataset.images:
            if dataset.task_type == "detection" or dataset.task_type == "segmentation":
                for bbox in img.bounding_boxes:
                    coco_dict["annotations"].append({
                        "id": annotation_id,
                        "image_id": img.id,
                        "category_id": bbox.category_id,
                        "bbox": [bbox.xmin, bbox.ymin, bbox.width, bbox.height],
                        "area": bbox.width * bbox.height,
                        "iscrowd": 0
                    })
                    annotation_id += 1

                for mask in img.segmentation_masks:
                    coco_dict["annotations"].append({
                        "id": annotation_id,
                        "image_id": img.id,
                        "category_id": mask.category_id,
                        "segmentation": mask.segmentation,
                        "area": sum([sum(mask.segmentation[i]) for i in range(len(mask.segmentation))]),
                        "iscrowd": 0
                    })
                    annotation_id += 1

            elif dataset.task_type == "panoptic":
                for mask in img.panoptic_segments:
                    coco_dict["annotations"].append({
                        "id": annotation_id,
                        "image_id": img.id,
                        "category_id": mask.category_id,
                        "segmentation": mask.mask  # Assuming this contains the path to a mask
                    })
                    annotation_id += 1

            elif dataset.task_type == "classification":
                for label in img.labels:
                    coco_dict["annotations"].append({
                        "id": annotation_id,
                        "image_id": img.id,
                        "category_id": label
                    })
                    annotation_id += 1
        return coco_dict


    @staticmethod
    def export_dataset(dataset: Dataset, output_path: str) -> None:
        """Export dataset to COCO file.

        Args:
            dataset (Dataset): the dataset that should be exported
            output_path (str): the path to the coco file
        """
        coco_dict = CocoLoader.to_coco_dict(dataset);
        with open(output_path, "w") as f:
            json.dump(coco_dict, f, indent=4)

