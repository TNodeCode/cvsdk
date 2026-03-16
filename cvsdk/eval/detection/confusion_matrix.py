"""Confusion matrix computation service for object detection evaluation."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ConfusionMatrixResult:
    """Result of confusion matrix computation."""
    matrix: np.ndarray
    class_names: List[str]
    total_tp: int
    total_fp: int
    total_fn: int
    precision: float
    recall: float
    f1_score: float


class DetectionConfusionMatrix:
    """Service class for computing confusion matrices for object detection.
    
    Takes two DataFrames as input:
    - Ground truth DataFrame: x0, y0, x1, y1, image, label, width, height
    - Detection DataFrame: x0, y0, x1, y1, image, label, score, width, height
    
    The detection DataFrame has an additional 'score' column for confidence threshold.
    """
    
    def __init__(
        self,
        ground_truth_df: pd.DataFrame,
        detections_df: pd.DataFrame,
        iou_threshold: float = 0.5,
        score_threshold: float = 0.0
    ):
        """Initialize the confusion matrix service.
        
        Args:
            ground_truth_df: DataFrame with ground truth bounding boxes.
                           Required columns: x0, y0, x1, y1, image, label
            detections_df: DataFrame with predicted bounding boxes.
                          Required columns: x0, y0, x1, y1, image, label, score
            iou_threshold: Minimum IoU threshold for matching (default: 0.5)
            score_threshold: Minimum confidence score for detections (default: 0.0)
        """
        self.ground_truth_df = ground_truth_df.copy()
        self.detections_df = detections_df.copy()
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        
        # Normalize column names for consistency
        self._normalize_columns()
        
        # Get all unique classes
        self._all_classes = self._get_all_classes()
        
    def _normalize_columns(self) -> None:
        """Normalize column names to handle legacy formats."""
        # Handle legacy column names for ground truth
        if 'x_min' in self.ground_truth_df.columns and 'x0' not in self.ground_truth_df.columns:
            self.ground_truth_df = self.ground_truth_df.rename(columns={
                'x_min': 'x0',
                'y_min': 'y0',
                'x_max': 'x1',
                'y_max': 'y1',
                'class_id': 'label'
            })
        
        # Handle legacy column names for detections
        if 'x_min' in self.detections_df.columns and 'x0' not in self.detections_df.columns:
            self.detections_df = self.detections_df.rename(columns={
                'x_min': 'x0',
                'y_min': 'y0',
                'x_max': 'x1',
                'y_max': 'y1',
                'class_id': 'label'
            })
        
        # Filter detections by score threshold
        if 'score' in self.detections_df.columns:
            self.detections_df = self.detections_df[
                self.detections_df['score'] >= self.score_threshold
            ]
    
    def _get_all_classes(self) -> List[int]:
        """Get all unique class labels from both dataframes."""
        gt_classes = set(self.ground_truth_df['label'].unique())
        det_classes = set(self.detections_df['label'].unique())
        return sorted(gt_classes | det_classes)
    
    @staticmethod
    def compute_iou(box1: Tuple[float, float, float, float], 
                    box2: Tuple[float, float, float, float]) -> float:
        """Compute Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: Tuple of (x0, y0, x1, y1) for first box
            box2: Tuple of (x0, y0, x1, y1) for second box
            
        Returns:
            IoU value between 0 and 1
        """
        x0_1, y0_1, x1_1, y1_1 = box1
        x0_2, y0_2, x1_2, y1_2 = box2
        
        # Compute intersection coordinates
        x0_i = max(x0_1, x0_2)
        y0_i = max(y0_1, y0_2)
        x1_i = min(x1_1, x1_2)
        y1_i = min(y1_1, y1_2)
        
        # Check if there is an intersection
        if x1_i <= x0_i or y1_i <= y0_i:
            return 0.0
        
        # Compute intersection area
        intersection_area = (x1_i - x0_i) * (y1_i - y0_i)
        
        # Compute union area
        box1_area = (x1_1 - x0_1) * (y1_1 - y0_1)
        box2_area = (x1_2 - x0_2) * (y1_2 - y0_2)
        union_area = box1_area + box2_area - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    
    def _match_detections_to_ground_truth(
        self,
        gt_boxes: pd.DataFrame,
        det_boxes: pd.DataFrame
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match detections to ground truth boxes using IoU threshold.
        
        Args:
            gt_boxes: DataFrame with ground truth boxes for one image
            det_boxes: DataFrame with detection boxes for one image
            
        Returns:
            Tuple of (matches, unmatched_det_indices, unmatched_gt_indices)
            - matches: List of (det_idx, gt_idx) tuples
            - unmatched_det_indices: List of unmatched detection indices
            - unmatched_gt_indices: List of unmatched ground truth indices
        """
        if len(gt_boxes) == 0:
            return [], list(range(len(det_boxes))), []
        
        if len(det_boxes) == 0:
            return [], [], list(range(len(gt_boxes)))
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(det_boxes), len(gt_boxes)))
        for d_idx, det_row in det_boxes.iterrows():
            det_box = (det_row['x0'], det_row['y0'], det_row['x1'], det_row['y1'])
            for g_idx, gt_row in gt_boxes.iterrows():
                gt_box = (gt_row['x0'], gt_row['y0'], gt_row['x1'], gt_row['y1'])
                iou_matrix[det_boxes.index.get_loc(d_idx), gt_boxes.index.get_loc(g_idx)] = \
                    self.compute_iou(det_box, gt_box)
        
        # Greedy matching: match each detection to the best available ground truth
        matches = []
        used_gt = set()
        used_det = set()
        
        # Sort by IoU (highest first)
        flat_ious = []
        for d_idx in range(len(det_boxes)):
            for g_idx in range(len(gt_boxes)):
                flat_ious.append((iou_matrix[d_idx, g_idx], d_idx, g_idx))
        
        flat_ious.sort(reverse=True)
        
        for iou_val, d_idx, g_idx in flat_ious:
            if d_idx in used_det or g_idx in used_gt:
                continue
            if iou_val >= self.iou_threshold:
                matches.append((d_idx, g_idx))
                used_det.add(d_idx)
                used_gt.add(g_idx)
        
        unmatched_det = [i for i in range(len(det_boxes)) if i not in used_det]
        unmatched_gt = [i for i in range(len(gt_boxes)) if i not in used_gt]
        
        return matches, unmatched_det, unmatched_gt
    
    def compute(self) -> ConfusionMatrixResult:
        """Compute the confusion matrix.
        
        Returns:
            ConfusionMatrixResult with matrix and metrics
        """
        num_classes = len(self._all_classes)
        # +1 for background/no-match class
        matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int)
        
        # Group by image
        gt_by_image = self.ground_truth_df.groupby('image')
        det_by_image = self.detections_df.groupby('image')
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        # Process each image
        for image_name in self.ground_truth_df['image'].unique():
            gt_boxes = gt_by_image.get_group(image_name) if image_name in gt_by_image.groups else pd.DataFrame()
            det_boxes = det_by_image.get_group(image_name) if image_name in det_by_image.groups else pd.DataFrame()
            
            if len(gt_boxes) == 0 and len(det_boxes) == 0:
                continue
            
            # Compute IoU matrix
            if len(gt_boxes) > 0 and len(det_boxes) > 0:
                iou_matrix = np.zeros((len(det_boxes), len(gt_boxes)))
                det_boxes_list = det_boxes.reset_index(drop=True)
                gt_boxes_list = gt_boxes.reset_index(drop=True)
                
                for d_idx, det_row in det_boxes_list.iterrows():
                    det_box = (det_row['x0'], det_row['y0'], det_row['x1'], det_row['y1'])
                    for g_idx, gt_row in gt_boxes_list.iterrows():
                        gt_box = (gt_row['x0'], gt_row['y0'], gt_row['x1'], gt_row['y1'])
                        iou_matrix[d_idx, g_idx] = self.compute_iou(det_box, gt_box)
                
                # Greedy matching
                matches = []
                used_gt = set()
                used_det = set()
                
                flat_ious = []
                for d_idx in range(len(det_boxes_list)):
                    for g_idx in range(len(gt_boxes_list)):
                        flat_ious.append((iou_matrix[d_idx, g_idx], d_idx, g_idx))
                
                flat_ious.sort(reverse=True)
                
                for iou_val, d_idx, g_idx in flat_ious:
                    if d_idx in used_det or g_idx in used_gt:
                        continue
                    if iou_val >= self.iou_threshold:
                        matches.append((d_idx, g_idx))
                        used_det.add(d_idx)
                        used_gt.add(g_idx)
                
                # Update confusion matrix with matches
                for d_idx, g_idx in matches:
                    det_label = det_boxes_list.iloc[d_idx]['label']
                    gt_label = gt_boxes_list.iloc[g_idx]['label']
                    
                    det_class_idx = self._all_classes.index(det_label) if det_label in self._all_classes else num_classes
                    gt_class_idx = self._all_classes.index(gt_label) if gt_label in self._all_classes else num_classes
                    
                    matrix[gt_class_idx, det_class_idx] += 1
                    
                    if det_label == gt_label:
                        total_tp += 1
                
                # False positives (detections not matched)
                for d_idx in range(len(det_boxes_list)):
                    if d_idx not in used_det:
                        det_label = det_boxes_list.iloc[d_idx]['label']
                        det_class_idx = self._all_classes.index(det_label) if det_label in self._all_classes else num_classes
                        # Background column for FP
                        matrix[num_classes, det_class_idx] += 1
                        total_fp += 1
                
                # False negatives (ground truth not matched)
                for g_idx in range(len(gt_boxes_list)):
                    if g_idx not in used_gt:
                        gt_label = gt_boxes_list.iloc[g_idx]['label']
                        gt_class_idx = self._all_classes.index(gt_label) if gt_label in self._all_classes else num_classes
                        # Background row for FN
                        matrix[gt_class_idx, num_classes] += 1
                        total_fn += 1
            
            elif len(gt_boxes) > 0:
                # All detections are false negatives
                for _, gt_row in gt_boxes.iterrows():
                    gt_label = gt_row['label']
                    gt_class_idx = self._all_classes.index(gt_label) if gt_label in self._all_classes else num_classes
                    matrix[gt_class_idx, num_classes] += 1
                    total_fn += 1
            
            elif len(det_boxes) > 0:
                # All ground truth are false positives
                for _, det_row in det_boxes.iterrows():
                    det_label = det_row['label']
                    det_class_idx = self._all_classes.index(det_label) if det_label in self._all_classes else num_classes
                    matrix[num_classes, det_class_idx] += 1
                    total_fp += 1
        
        # Handle images with detections but no ground truth
        gt_images = set(self.ground_truth_df['image'].unique())
        det_images = set(self.detections_df['image'].unique())
        
        # Images with detections but no ground truth
        for image_name in det_images - gt_images:
            det_boxes = det_by_image.get_group(image_name)
            for _, det_row in det_boxes.iterrows():
                det_label = det_row['label']
                det_class_idx = self._all_classes.index(det_label) if det_label in self._all_classes else num_classes
                matrix[num_classes, det_class_idx] += 1
                total_fp += 1
        
        # Compute metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Create class names list
        class_names = [f"class_{c}" for c in self._all_classes] + ["background"]
        
        return ConfusionMatrixResult(
            matrix=matrix,
            class_names=class_names,
            total_tp=total_tp,
            total_fp=total_fp,
            total_fn=total_fn,
            precision=precision,
            recall=recall,
            f1_score=f1_score
        )
    
    def get_per_class_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get precision, recall, and F1 score for each class.
        
        Returns:
            Dictionary mapping class names to their metrics
        """
        result = self.compute()
        metrics = {}
        
        for i, class_name in enumerate(result.class_names):
            # True positives: diagonal element
            tp = result.matrix[i, i]
            
            # False positives: column sum minus diagonal
            fp = int(result.matrix[:, i].sum()) - tp
            
            # False negatives: row sum minus diagonal
            fn = int(result.matrix[i, :].sum()) - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics[class_name] = {
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        return metrics
    
    def plot(self, output_path: str, figsize: Tuple[int, int] = (12, 10)) -> None:
        """Plot and save the confusion matrix.
        
        Args:
            output_path: Path to save the confusion matrix plot
            figsize: Figure size (width, height) in inches
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        result = self.compute()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            result.matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=result.class_names,
            yticklabels=result.class_names,
            ax=ax
        )
        
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('True Class')
        ax.set_title(f'Confusion Matrix (IoU={self.iou_threshold}, Score={self.score_threshold})\n'
                     f'Precision: {result.precision:.3f}, Recall: {result.recall:.3f}, F1: {result.f1_score:.3f}')
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


def compute_confusion_matrix(
    ground_truth_df: pd.DataFrame,
    detections_df: pd.DataFrame,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0
) -> ConfusionMatrixResult:
    """Convenience function to compute confusion matrix.
    
    Args:
        ground_truth_df: DataFrame with ground truth bounding boxes
        detections_df: DataFrame with predicted bounding boxes
        iou_threshold: Minimum IoU threshold for matching (default: 0.5)
        score_threshold: Minimum confidence score for detections (default: 0.0)
        
    Returns:
        ConfusionMatrixResult with matrix and metrics
    """
    calculator = DetectionConfusionMatrix(
        ground_truth_df=ground_truth_df,
        detections_df=detections_df,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold
    )
    return calculator.compute()
