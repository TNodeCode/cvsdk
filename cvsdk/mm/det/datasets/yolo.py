# Copyright (c) OpenMMLab. All rights reserved.
"""YOLO dataset for object detection.

This dataset class supports the YOLO format for object detection:
- Label files: One .txt file per image in labels/ directory
- Format per line: class_id x_center y_center width height (normalized 0-1)
- Directory structure:
    root/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── data.yaml (defines classes)
"""

import os
import os.path as osp
from typing import Dict, List, Optional, Union

from mmengine.fileio import get_local_path, list_from_file, load

from mmdet.registry import DATASETS
from mmdet.datasets.base_det_dataset import BaseDetDataset


@DATASETS.register_module()
class YOLODataset(BaseDetDataset):
    """Dataset for YOLO format object detection.

    Args:
        img_suffix (str): Suffix of images. Default: '.jpg'.
        label_suffix (str): Suffix of label files. Default: '.txt'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        return_classes (bool): Whether to return class information
            for open vocabulary-based algorithms. Defaults to False.
        caption_prompt (dict, optional): Prompt for captioning.
            Defaults to None.
    """

    def __init__(self,
                 img_suffix: str = '.jpg',
                 label_suffix: str = '.txt',
                 backend_args: Optional[dict] = None,
                 return_classes: bool = False,
                 caption_prompt: Optional[dict] = None,
                 **kwargs) -> None:
        self.img_suffix = img_suffix
        self.label_suffix = label_suffix
        super().__init__(
            backend_args=backend_args,
            return_classes=return_classes,
            caption_prompt=caption_prompt,
            **kwargs)

    def _scan_images_recursive(self, img_dir: str) -> List[tuple]:
        """Recursively scan for image files in directory.
        
        Args:
            img_dir: Directory to scan
            
        Returns:
            List of tuples (relative_path, filename) for each image
        """
        img_files = []
        
        if not osp.exists(img_dir):
            return img_files
            
        for root, dirs, files in os.walk(img_dir):
            for f in files:
                if f.endswith(self.img_suffix):
                    # Get relative path from img_dir
                    rel_path = osp.relpath(osp.join(root, f), img_dir)
                    img_files.append((rel_path, f))
        
        return img_files

    def load_data_list(self) -> List[dict]:
        """Load annotations from YOLO format annotation files.

        YOLO directory structure (standard):
            root/
            ├── images/
            │   └── train/  (or val/, test/)
            │       ├── image1.jpg
            │       └── image2.jpg
            └── labels/
                └── train/  (or val/, test/)
                    ├── image1.txt
                    └── image2.txt

        Or with class subdirectories:
            root/
            ├── images/
            │   └── train/
            │       ├── class1/
            │       │   └── image1.jpg
            │       └── class2/
            │           └── image2.jpg
            └── labels/
                └── train/
                    ├── class1/
                    │   └── image1.txt
                    └── class2/
                        └── image2.txt

        Returns:
            List[dict]: A list of annotation.
        """
        assert self._metainfo.get('classes', None) is not None, \
            '`classes` in `YOLODataset` can not be None.'

        self.cat2label = {
            cat: i
            for i, cat in enumerate(self._metainfo['classes'])
        }

        data_list = []

        # Get directories from data_prefix
        img_dir = self.data_prefix.get('img', '')
        label_dir = self.data_prefix.get('label', '')
        
        # Make path absolute relative to data_root
        if osp.isabs(img_dir):
            img_dir_for_scan = img_dir
        else:
            img_dir_for_scan = osp.join(self.data_root, img_dir) if self.data_root else img_dir
        
        # Recursively scan for image files
        img_files = self._scan_images_recursive(img_dir_for_scan)
        
        # Sort for reproducibility
        img_files.sort()
        
        for rel_path, img_file in img_files:
            # Get base name without extension
            img_id = osp.splitext(img_file)[0]
            
            # Construct paths - use relative path for img_path
            img_path = osp.join(self.data_prefix.get('img', ''), rel_path)
            
            # For label path, we need to find the corresponding label file
            # It could be in parallel directory structure or in class subdirectories
            # Try to find label by replacing images/ with labels/ and keeping subdir structure
            label_rel_path = rel_path.replace('images/', 'labels/')
            label_rel_path = osp.splitext(label_rel_path)[0] + self.label_suffix
            label_path = osp.join(self.data_prefix.get('label', ''), label_rel_path)
            
            raw_img_info = {
                'img_id': img_id,
                'file_name': rel_path,
                'img_path': img_path,
                'label_path': label_path
            }
            
            parsed_data_info = self.parse_data_info(raw_img_info)
            data_list.append(parsed_data_info)

        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information.

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_path = raw_data_info['img_path']
        label_path = raw_data_info['label_path']
        img_id = raw_data_info['img_id']

        data_info = {}
        data_info['img_path'] = img_path
        data_info['img_id'] = img_id
        
        # Initialize with empty instances
        instances = []
        
        # Try to load label file - handle both local and remote paths
        try:
            with get_local_path(label_path, backend_args=self.backend_args) as local_path:
                with open(local_path, 'r') as f:
                    lines = f.readlines()
                    
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                    except ValueError:
                        continue
                    
                    # Skip if class_id is out of range
                    if class_id not in self.cat2label:
                        continue
                    
                    # Store normalized bbox for now
                    # Will convert to absolute coordinates later in the pipeline
                    instance = {
                        'bbox': [x_center, y_center, w, h],  # normalized center format
                        'bbox_label': self.cat2label[class_id],
                        'ignore_flag': 0
                    }
                    
                    # Handle segmentation if available (YOLO segmentation format)
                    if len(parts) >= 6:
                        # Segmentation points follow the bbox
                        seg_points = []
                        for i in range(5, len(parts), 2):
                            if i + 1 < len(parts):
                                seg_x = float(parts[i])
                                seg_y = float(parts[i + 1])
                                seg_points.append([seg_x, seg_y])
                        if len(seg_points) >= 3:
                            instance['mask'] = [seg_points]
                    
                    instances.append(instance)
        except (FileNotFoundError, OSError):
            # No label file found - this is normal for test data
            pass
        
        data_info['instances'] = instances

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True

        return data_info

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        valid_data_infos = []
        for data_info in self.data_list:
            # Filter empty gt
            if filter_empty_gt and len(data_info.get('instances', [])) == 0:
                continue
            
            # Filter by min size (if image dimensions are available)
            width = data_info.get('width', 0)
            height = data_info.get('height', 0)
            if min_size > 0 and (width == 0 or height == 0):
                # If dimensions not available, keep the sample
                # Dimensions will be loaded during training
                valid_data_infos.append(data_info)
            elif min_size > 0 and min(width, height) >= min_size:
                valid_data_infos.append(data_info)
            elif min_size == 0:
                valid_data_infos.append(data_info)

        return valid_data_infos
