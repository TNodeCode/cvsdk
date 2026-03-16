import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from cvsdk.model import Dataset, Image, BoundingBox, SegmentationMask, PanopticSegment


class DataframeLoader:
    """DataFrame dataset import and export.
    
    Supports loading from CSV and Parquet files.
    
    DataFrame format for detection datasets:
    - One row per bounding box with columns: 'x0', 'y0', 'x1', 'y1', 'image', 'label', 'width', 'height'
    - 'x0', 'y0': top-left corner coordinates
    - 'x1', 'y1': bottom-right corner coordinates
    - 'image': path to the image file
    - 'label': integer label encoding
    - 'width', 'height': image dimensions
    
    Legacy column names (x_min, y_min, x_max, y_max, class_id) are also supported for backward compatibility.
    
    DataFrame format for classification:
    - One row per image with 'image' (categorical), 'class' columns
    
    DataFrame format for segmentation:
    - One row per mask with 'image' (categorical), 'mask' (list of coords), 'class_id', 'width', 'height' columns
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
            # Support both new column names (x0, y0, x1, y1, label) and legacy (x_min, y_min, x_max, y_max, class_id)
            images_dict = {}
            
            # Determine which column names to use
            has_new_columns = 'x0' in df.columns and 'y0' in df.columns and 'x1' in df.columns and 'y1' in df.columns and 'label' in df.columns
            has_legacy_columns = 'x_min' in df.columns and 'y_min' in df.columns and 'x_max' in df.columns and 'y_max' in df.columns and 'class_id' in df.columns
            
            for _, row in df.iterrows():
                image_path = row['image']
                
                if has_new_columns:
                    x0 = row['x0']
                    y0 = row['y0']
                    x1 = row['x1']
                    y1 = row['y1']
                    class_id = row['label']
                elif has_legacy_columns:
                    x0 = row['x_min']
                    y0 = row['y_min']
                    x1 = row['x_max']
                    y1 = row['y_max']
                    class_id = row['class_id']
                else:
                    raise ValueError("DataFrame must contain either new columns (x0, y0, x1, y1, label) or legacy columns (x_min, y_min, x_max, y_max, class_id)")
                
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
                    xmin=x0,
                    ymin=y0,
                    width=x1 - x0,
                    height=y1 - y0,
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
            # Support both new column names (x0, y0, x1, y1, label) and legacy (x_min, y_min, x_max, y_max, class_id)
            images_dict = {}
            
            # Determine which column names to use
            has_new_columns = 'x0' in df.columns and 'y0' in df.columns and 'x1' in df.columns and 'y1' in df.columns and 'label' in df.columns
            has_legacy_columns = 'x_min' in df.columns and 'y_min' in df.columns and 'x_max' in df.columns and 'y_max' in df.columns and 'class_id' in df.columns
            
            for _, row in df.iterrows():
                image_path = row['image']
                
                if has_new_columns:
                    x0 = row['x0']
                    y0 = row['y0']
                    x1 = row['x1']
                    y1 = row['y1']
                    class_id = row['label']
                elif has_legacy_columns:
                    x0 = row['x_min']
                    y0 = row['y_min']
                    x1 = row['x_max']
                    y1 = row['y_max']
                    class_id = row['class_id']
                else:
                    raise ValueError("DataFrame must contain either new columns (x0, y0, x1, y1, label) or legacy columns (x_min, y_min, x_max, y_max, class_id)")
                
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
                    xmin=x0,
                    ymin=y0,
                    width=x1 - x0,
                    height=y1 - y0,
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
        file_format: Optional[str] = None,
        split_map: Optional[Dict[int, str]] = None
    ) -> None:
        """Export the dataset into a DataFrame format (CSV or Parquet).

        Args:
            dataset (Dataset): Dataset to export.
            output_path (str): Target file path or directory path.
            file_format (str): Optional file format override ('csv' or 'parquet').
                              If not provided, inferred from file extension.
            split_map (Optional[Dict[int, str]]): Optional mapping of image_id to split name.
                                                  If provided and output_path is a directory,
                                                  exports separate files per split (df_train.csv, df_val.csv, df_test.csv).
        """
        output_path = Path(output_path)
        
        # Determine file format
        if file_format is None:
            # Check if output_path is a directory (either exists or has no file extension)
            # This handles both existing directories and new directory paths
            is_directory = output_path.is_dir() or (not output_path.suffix and not output_path.name.endswith('.'))
            
            if is_directory:
                file_format = 'csv'
            else:
                suffix = output_path.suffix.lower()
                if suffix == '.csv':
                    file_format = 'csv'
                elif suffix == '.parquet':
                    file_format = 'parquet'
                else:
                    raise ValueError(f"Unsupported file format: {suffix}. Use 'csv' or 'parquet'.")
        
        # Check if we should export splits to a directory
        # This happens when split_map is provided and output_path is a directory (or will be)
        # We determine this by checking if it's not a file with an extension
        is_directory_output = not output_path.suffix or output_path.is_dir()
        
        should_export_splits = (
            split_map is not None and
            len(split_map) > 0 and
            is_directory_output
        )
        
        if should_export_splits:
            # Create the output directory if it doesn't exist
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Group images by split
            split_images: Dict[str, List[Image]] = {
                'train': [],
                'val': [],
                'test': []
            }
            
            for image in dataset.images:
                split = split_map.get(image.id, 'train')  # Default to 'train' if not in split_map
                if split not in split_images:
                    split_images[split] = []
                split_images[split].append(image)
            
            # Export each split
            for split_name, images in split_images.items():
                if not images:
                    continue
                    
                # Create a temporary dataset for this split
                split_dataset = Dataset(
                    images=images,
                    categories=dataset.categories,
                    task_type=dataset.task_type
                )
                
                # Determine output file path
                split_filename = f"df_{split_name}.{file_format}"
                split_output_path = output_path / split_filename
                
                # Export this split
                DataframeLoader._export_single_dataset(split_dataset, split_output_path, file_format)
        else:
            # Original behavior: export single file
            DataframeLoader._export_single_dataset(dataset, output_path, file_format)
    
    @staticmethod
    def _export_single_dataset(
        dataset: Dataset,
        output_path: Path,
        file_format: str
    ) -> None:
        """Export a single dataset to a file.
        
        Args:
            dataset (Dataset): Dataset to export.
            output_path (Path): Target file path.
            file_format (str): File format ('csv' or 'parquet').
        """
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
            # One row per bounding box with new column names: x0, y0, x1, y1, label
            rows = []
            for image in dataset.images:
                for box in image.bounding_boxes:
                    rows.append({
                        'x0': box.xmin,
                        'y0': box.ymin,
                        'x1': box.xmin + box.width,
                        'y1': box.ymin + box.height,
                        'image': image.file_name,
                        'label': box.category_id,
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
            # One row per bounding box with track_id and new column names: x0, y0, x1, y1, label
            rows = []
            for image in dataset.images:
                for box in image.bounding_boxes:
                    rows.append({
                        'x0': box.xmin,
                        'y0': box.ymin,
                        'x1': box.xmin + box.width,
                        'y1': box.ymin + box.height,
                        'image': image.file_name,
                        'label': box.category_id,
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
