import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
from cvsdk.model import Dataset, Image, BoundingBox, SegmentationMask, PanopticSegment


class DataframeLoader:
    """DataFrame dataset import and export.
    
    Supports loading from CSV and Parquet files.
    
    DataFrame format:
    - Classification: one row per image with 'image' (categorical), 'class' columns
    - Detection: one row per bounding box with 'image' (categorical), 'x_min', 'y_min', 'x_max', 'y_max', 'class_id', 'width', 'height' columns
    - Segmentation: one row per mask with 'image' (categorical), 'mask' (list of coords), 'class_id', 'width', 'height' columns
    """

    @staticmethod
    def import_dataset(
        file_path: str,
        task_type: str,
        file_format: Optional[str] = None
    ) -> Dataset:
        """Import a DataFrame-format dataset into the Dataset Pydantic model.

        Args:
            file_path (str): Path to the CSV or Parquet file.
            task_type (str): Task type: 'classification', 'detection', 'segmentation', 'panoptic', 'tracking'.
            file_format (str): Optional file format override ('csv' or 'parquet'). 
                              If not provided, inferred from file extension.

        Returns:
            Dataset: Parsed dataset object.
        """
        file_path = Path(file_path)
        
        # Determine file format
        if file_format is None:
            suffix = file_path.suffix.lower()
            if suffix == '.csv':
                file_format = 'csv'
            elif suffix == '.parquet':
                file_format = 'parquet'
            else:
                raise ValueError(f"Unsupported file format: {suffix}. Use 'csv' or 'parquet'.")
        
        # Load the dataframe
        if file_format == 'csv':
            df = pd.read_csv(file_path)
        elif file_format == 'parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        # Convert categorical 'image' column to string if needed
        if 'image' in df.columns and hasattr(df['image'], 'cat'):
            df['image'] = df['image'].astype(str)
        
        # Build categories from unique class labels
        categories = {}
        
        if task_type == "classification":
            # For classification, one row per image
            images = []
            image_id = 0
            
            # Get unique images and their class labels
            for _, row in df.iterrows():
                image_path = row['image']
                class_label = row['class']
                
                # Add category if not exists
                if class_label not in categories:
                    categories[len(categories)] = str(class_label)
                
                class_id = [k for k, v in categories.items() if str(v) == str(class_label)][0]
                
                image = Image(
                    id=image_id,
                    file_name=image_path,
                    width=row.get('width', 0),
                    height=row.get('height', 0),
                    labels=[class_id]
                )
                images.append(image)
                image_id += 1
            
            return Dataset(images=images, categories=categories, task_type=task_type)
        
        elif task_type == "detection":
            # For detection, one row per bounding box
            images_dict = {}
            
            for _, row in df.iterrows():
                image_path = row['image']
                x_min = row['x_min']
                y_min = row['y_min']
                x_max = row['x_max']
                y_max = row['y_max']
                class_id = row['class_id']
                width = row.get('width', 0)
                height = row.get('height', 0)
                
                # Add category if not exists
                if class_id not in categories:
                    categories[class_id] = f"class_{class_id}"
                
                # Get or create image
                if image_path not in images_dict:
                    images_dict[image_path] = Image(
                        id=len(images_dict),
                        file_name=image_path,
                        width=width,
                        height=height
                    )
                
                # Add bounding box
                bbox = BoundingBox(
                    xmin=x_min,
                    ymin=y_min,
                    width=x_max - x_min,
                    height=y_max - y_min,
                    category_id=class_id
                )
                images_dict[image_path].bounding_boxes.append(bbox)
            
            return Dataset(images=list(images_dict.values()), categories=categories, task_type=task_type)
        
        elif task_type == "segmentation":
            # For segmentation, one row per mask
            images_dict = {}
            
            for _, row in df.iterrows():
                image_path = row['image']
                mask_coords = row['mask']
                class_id = row['class_id']
                width = row.get('width', 0)
                height = row.get('height', 0)
                
                # Add category if not exists
                if class_id not in categories:
                    categories[class_id] = f"class_{class_id}"
                
                # Get or create image
                if image_path not in images_dict:
                    images_dict[image_path] = Image(
                        id=len(images_dict),
                        file_name=image_path,
                        width=width,
                        height=height
                    )
                
                # Parse mask coordinates
                if isinstance(mask_coords, str):
                    # Parse string representation of list
                    mask_coords = eval(mask_coords)
                elif isinstance(mask_coords, np.ndarray):
                    mask_coords = mask_coords.tolist()
                
                # Convert to list of lists format for SegmentationMask
                polygons = [mask_coords]
                
                # Add segmentation mask
                seg_mask = SegmentationMask(
                    segmentation=polygons,
                    category_id=class_id
                )
                images_dict[image_path].segmentation_masks.append(seg_mask)
            
            return Dataset(images=list(images_dict.values()), categories=categories, task_type=task_type)
        
        elif task_type == "panoptic":
            # For panoptic, similar to segmentation but uses PanopticSegment
            images_dict = {}
            
            for _, row in df.iterrows():
                image_path = row['image']
                mask_path = row['mask']
                class_id = row['class_id']
                width = row.get('width', 0)
                height = row.get('height', 0)
                
                # Add category if not exists
                if class_id not in categories:
                    categories[class_id] = f"class_{class_id}"
                
                # Get or create image
                if image_path not in images_dict:
                    images_dict[image_path] = Image(
                        id=len(images_dict),
                        file_name=image_path,
                        width=width,
                        height=height
                    )
                
                # Add panoptic segment
                panoptic_seg = PanopticSegment(
                    segment_id=len(images_dict[image_path].panoptic_segments),
                    category_id=class_id,
                    mask=mask_path
                )
                images_dict[image_path].panoptic_segments.append(panoptic_seg)
            
            return Dataset(images=list(images_dict.values()), categories=categories, task_type=task_type)
        
        elif task_type == "tracking":
            # For tracking, similar to detection but with track_id
            images_dict = {}
            
            for _, row in df.iterrows():
                image_path = row['image']
                x_min = row['x_min']
                y_min = row['y_min']
                x_max = row['x_max']
                y_max = row['y_max']
                class_id = row['class_id']
                track_id = row.get('track_id', 0)
                width = row.get('width', 0)
                height = row.get('height', 0)
                
                # Add category if not exists
                if class_id not in categories:
                    categories[class_id] = f"class_{class_id}"
                
                # Get or create image
                if image_path not in images_dict:
                    images_dict[image_path] = Image(
                        id=len(images_dict),
                        file_name=image_path,
                        width=width,
                        height=height
                    )
                
                # Add bounding box with track_id
                bbox = BoundingBox(
                    xmin=x_min,
                    ymin=y_min,
                    width=x_max - x_min,
                    height=y_max - y_min,
                    category_id=class_id,
                    id=track_id
                )
                images_dict[image_path].bounding_boxes.append(bbox)
            
            return Dataset(images=list(images_dict.values()), categories=categories, task_type=task_type)
        
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    @staticmethod
    def export_dataset(
        dataset: Dataset,
        output_path: str,
        file_format: Optional[str] = None
    ) -> None:
        """Export the dataset into a DataFrame format (CSV or Parquet).

        Args:
            dataset (Dataset): Dataset to export.
            output_path (str): Target file path.
            file_format (str): Optional file format override ('csv' or 'parquet').
                              If not provided, inferred from file extension.
        """
        output_path = Path(output_path)
        
        # Determine file format
        if file_format is None:
            suffix = output_path.suffix.lower()
            if suffix == '.csv':
                file_format = 'csv'
            elif suffix == '.parquet':
                file_format = 'parquet'
            else:
                raise ValueError(f"Unsupported file format: {suffix}. Use 'csv' or 'parquet'.")
        
        if dataset.task_type == "classification":
            # One row per image with class label
            rows = []
            for image in dataset.images:
                for label in image.labels:
                    rows.append({
                        'image': image.file_name,
                        'class': dataset.categories.get(label, f"class_{label}"),
                        'width': image.width,
                        'height': image.height
                    })
            df = pd.DataFrame(rows)
            # Convert image column to categorical for storage
            df['image'] = df['image'].astype('category')
        
        elif dataset.task_type == "detection":
            # One row per bounding box
            rows = []
            for image in dataset.images:
                for box in image.bounding_boxes:
                    rows.append({
                        'image': image.file_name,
                        'x_min': box.xmin,
                        'y_min': box.ymin,
                        'x_max': box.xmin + box.width,
                        'y_max': box.ymin + box.height,
                        'class_id': box.category_id,
                        'width': image.width,
                        'height': image.height
                    })
            df = pd.DataFrame(rows)
            # Convert image column to categorical for storage
            df['image'] = df['image'].astype('category')
        
        elif dataset.task_type == "segmentation":
            # One row per segmentation mask
            rows = []
            for image in dataset.images:
                for mask in image.segmentation_masks:
                    # Flatten the polygon coordinates
                    if mask.segmentation and len(mask.segmentation) > 0:
                        # Take the first polygon
                        flat_mask = mask.segmentation[0]
                        rows.append({
                            'image': image.file_name,
                            'mask': flat_mask,
                            'class_id': mask.category_id,
                            'width': image.width,
                            'height': image.height
                        })
            df = pd.DataFrame(rows)
            # Convert image column to categorical for storage
            df['image'] = df['image'].astype('category')
        
        elif dataset.task_type == "panoptic":
            # One row per panoptic segment
            rows = []
            for image in dataset.images:
                for segment in image.panoptic_segments:
                    rows.append({
                        'image': image.file_name,
                        'mask': segment.mask,
                        'class_id': segment.category_id,
                        'width': image.width,
                        'height': image.height
                    })
            df = pd.DataFrame(rows)
            # Convert image column to categorical for storage
            df['image'] = df['image'].astype('category')
        
        elif dataset.task_type == "tracking":
            # One row per bounding box with track_id
            rows = []
            for image in dataset.images:
                for box in image.bounding_boxes:
                    rows.append({
                        'image': image.file_name,
                        'x_min': box.xmin,
                        'y_min': box.ymin,
                        'x_max': box.xmin + box.width,
                        'y_max': box.ymin + box.height,
                        'class_id': box.category_id,
                        'track_id': box.id,
                        'width': image.width,
                        'height': image.height
                    })
            df = pd.DataFrame(rows)
            # Convert image column to categorical for storage
            df['image'] = df['image'].astype('category')
        
        else:
            raise ValueError(f"Unsupported task type: {dataset.task_type}")
        
        # Save to file
        if file_format == 'csv':
            df.to_csv(output_path, index=False)
        elif file_format == 'parquet':
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
