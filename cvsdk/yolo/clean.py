"""
Service for cleaning YOLO datasets by finding and moving mismatched files.

This module handles two types of mismatches:
1. Images without labels (or with empty/background labels)
2. Labels without corresponding images (orphan labels)
"""

import os
import shutil
import yaml
from pathlib import Path
from typing import Tuple, List, Set


class DatasetCleaner:
    """Service class for cleaning YOLO datasets."""
    
    def __init__(self, dataset_dir: Path, output_dir: Path = None):
        """
        Initialize the DatasetCleaner.
        
        Args:
            dataset_dir: Path to the YOLO dataset root
            output_dir: Output directory for mismatched files (default: None)
        """
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir) if output_dir else None
        self.valid_class_ids = self._get_valid_class_ids()
    
    def _get_valid_class_ids(self) -> Set[int]:
        """
        Read the data.yaml file to get valid class IDs.
        
        Returns:
            Set of valid class IDs (integers)
        """
        data_yaml_path = self.dataset_dir / "data.yaml"
        if not data_yaml_path.exists():
            # Fallback to default if data.yaml doesn't exist
            return set(range(5))  # Classes 0-4
        
        try:
            with open(data_yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            if 'names' in data:
                # Get all class IDs from the names dictionary
                class_ids = set()
                for class_id in data['names'].keys():
                    try:
                        class_ids.add(int(class_id))
                    except (ValueError, TypeError):
                        pass
                return class_ids
        except Exception:
            pass
        
        # Fallback to default
        return set(range(5))
    
    def _get_image_extensions(self) -> Set[str]:
        """Return common image extensions used in YOLO datasets."""
        return {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    
    def _has_valid_label(self, label_path: Path) -> Tuple[bool, bool, bool]:
        """
        Check if a label file exists and is not empty, and if it has valid class IDs and coordinates.
        
        Supports both detection format (5 values) and segmentation format (variable number of coordinates).
        
        Detection format: <class_id> <x_center> <y_center> <width> <height>
        Segmentation format: <class_id> <x1> <y1> <x2> <y2> <x3> <y3> ... (polygon coordinates)
        
        Args:
            label_path: Path to the label file
            
        Returns:
            Tuple of (has_valid_label, is_background, has_invalid_coords) where:
            - has_valid_label: True if label file exists and has content
            - is_background: True if all class IDs are not in valid_class_ids (background)
            - has_invalid_coords: True if any coordinates are outside [0, 1]
        """
        if not label_path.exists():
            return False, False, False
        
        # Check if file is empty
        if label_path.stat().st_size == 0:
            return False, False, False
        
        # Check for invalid class IDs and invalid coordinates
        try:
            with open(label_path, 'r') as f:
                content = f.read().strip()
                if not content:
                    return False, False, False
                
                lines = content.split('\n')
                has_valid_class = False
                has_background_class = False
                has_invalid_coords = False
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    # Detection format requires at least 5 values
                    # Segmentation format requires at least 3 values (class_id + 1 coordinate pair)
                    if len(parts) < 3:
                        continue
                    try:
                        class_id = int(parts[0])
                        
                        # Check class ID against valid_class_ids
                        if class_id in self.valid_class_ids:
                            has_valid_class = True
                        else:
                            has_background_class = True
                        
                        # Check all coordinates - they should all be within [0, 1]
                        # For detection: parts[1:5] are x_center, y_center, width, height
                        # For segmentation: parts[1:] are all (xi, yi) coordinate pairs
                        coords = parts[1:]
                        for coord_str in coords:
                            coord = float(coord_str)
                            if coord < 0 or coord > 1:
                                has_invalid_coords = True
                                break
                            
                    except ValueError:
                        continue
                
                # If only background classes, mark as background
                if has_background_class and not has_valid_class:
                    return True, True, has_invalid_coords
                
                return True, False, has_invalid_coords
        except Exception:
            return False, False, False
    
    def find_unlabelled_images(self) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
        """
        Find all images without valid labels, with background labels, or with invalid coordinates.
        
        Returns:
            Tuple of (unlabelled_images, invalid_coords_images) where each is a list of tuples
            (image_path, relative_path)
        """
        images_dir = self.dataset_dir / "images"
        labels_dir = self.dataset_dir / "labels"
        image_extensions = self._get_image_extensions()
        
        unlabelled_images = []
        invalid_coords_images = []
        
        # Walk through all subdirectories in the images folder
        for split_dir in images_dir.iterdir():
            if not split_dir.is_dir():
                continue
            
            split_name = split_dir.name  # train, val, test, etc.
            
            # Process each subdirectory within the split
            for sub_dir in split_dir.iterdir():
                if not sub_dir.is_dir():
                    continue
                
                # Find corresponding label directory
                label_sub_dir = labels_dir / split_name / sub_dir.name
                
                # Process all image files in this directory
                for img_file in sub_dir.iterdir():
                    if img_file.suffix.lower() not in image_extensions:
                        continue
                    
                    # Construct the corresponding label file path
                    label_file = label_sub_dir / f"{img_file.stem}.txt"
                    
                    # Check if label is missing, empty, has background classes, or has invalid coords
                    has_valid, is_background, has_invalid_coords = self._has_valid_label(label_file)
                    
                    # Calculate relative path from images directory
                    relative_path = img_file.relative_to(images_dir)
                    
                    if not has_valid or is_background:
                        unlabelled_images.append((img_file, relative_path))
                    
                    if has_invalid_coords:
                        invalid_coords_images.append((img_file, relative_path))
        
        return unlabelled_images, invalid_coords_images
    
    def find_orphan_labels(self) -> List[Tuple[Path, Path]]:
        """
        Find all label files without corresponding images (orphan labels).
        
        Returns:
            List of tuples (label_path, relative_path) for orphan labels
        """
        images_dir = self.dataset_dir / "images"
        labels_dir = self.dataset_dir / "labels"
        image_extensions = self._get_image_extensions()
        
        orphan_labels = []
        
        # Build a set of all image stems (filename without extension) in the images directory
        all_image_stems = set()
        for split_dir in images_dir.iterdir():
            if not split_dir.is_dir():
                continue
            
            for sub_dir in split_dir.iterdir():
                if not sub_dir.is_dir():
                    continue
                
                for img_file in sub_dir.iterdir():
                    if img_file.suffix.lower() in image_extensions:
                        all_image_stems.add(img_file.stem)
        
        # Walk through all label files and check if corresponding image exists
        for split_dir in labels_dir.iterdir():
            if not split_dir.is_dir():
                continue
            
            split_name = split_dir.name
            
            for sub_dir in split_dir.iterdir():
                if not sub_dir.is_dir():
                    continue
                
                # Find corresponding images directory
                image_sub_dir = images_dir / split_name / sub_dir.name
                
                for label_file in sub_dir.iterdir():
                    if label_file.suffix.lower() != ".txt":
                        continue
                    
                    # Check if corresponding image exists
                    if label_file.stem not in all_image_stems:
                        relative_path = label_file.relative_to(labels_dir)
                        orphan_labels.append((label_file, relative_path))
        
        return orphan_labels
    
    def move_images_to_unlabelled(self, images: List[Tuple[Path, Path]]) -> int:
        """
        Move images to the unlabelled directory while preserving structure.
        
        Args:
            images: List of tuples (image_path, relative_path)
            
        Returns:
            Number of images moved
        """
        if not self.output_dir:
            return 0
            
        output_images_dir = self.output_dir / "images"
        output_labels_dir = self.output_dir / "labels"
        
        moved_count = 0
        
        for img_path, relative_path in images:
            # Create destination path preserving directory structure
            dest_path = output_images_dir / relative_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move the image
            shutil.move(str(img_path), str(dest_path))
            moved_count += 1
            
            # Also move the label file if it exists (even if empty or invalid)
            # This helps keep the dataset consistent
            label_path = self.dataset_dir / "labels" / relative_path.with_suffix(".txt")
            if label_path.exists():
                dest_label_path = output_labels_dir / relative_path.with_suffix(".txt")
                dest_label_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(label_path), str(dest_label_path))
        
        return moved_count
    
    def move_orphan_labels(self, labels: List[Tuple[Path, Path]]) -> int:
        """
        Move orphan label files to the unlabelled directory while preserving structure.
        
        Args:
            labels: List of tuples (label_path, relative_path)
            
        Returns:
            Number of labels moved
        """
        if not self.output_dir:
            return 0
            
        output_labels_dir = self.output_dir / "labels"
        
        moved_count = 0
        
        for label_path, relative_path in labels:
            # Create destination path preserving directory structure
            dest_path = output_labels_dir / relative_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move the label file
            shutil.move(str(label_path), str(dest_path))
            moved_count += 1
        
        return moved_count
    
    def clean(self, should_move: bool = True) -> dict:
        """
        Find and optionally move mismatched files in the dataset.
        
        Args:
            should_move: Whether to actually move files (default: True)
            
        Returns:
            Dictionary with summary of findings
        """
        # Find unlabelled images and images with invalid coordinates
        unlabelled, invalid_coords = self.find_unlabelled_images()
        
        # Find orphan labels
        orphan_labels = self.find_orphan_labels()
        
        result = {
            'unlabelled': unlabelled,
            'invalid_coords': invalid_coords,
            'orphan_labels': orphan_labels,
            'moved_images': 0,
            'moved_labels': 0
        }
        
        # Move files if should_move is True
        if should_move and self.output_dir:
            all_to_move = unlabelled + invalid_coords
            result['moved_images'] = self.move_images_to_unlabelled(all_to_move)
            result['moved_labels'] = self.move_orphan_labels(orphan_labels)
        
        return result
