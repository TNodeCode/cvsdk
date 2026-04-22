"""Precision-Confidence curve computation service for object detection evaluation."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from cvsdk.eval.detection.confusion_matrix import DetectionConfusionMatrix


@dataclass
class PrecisionConfidenceResult:
    """Result of precision-confidence curve computation."""
    confidence_thresholds: np.ndarray
    per_class_precision: Dict[str, np.ndarray]
    average_precision: np.ndarray
    class_names: List[str]


class PrecisionConfidenceCurve:
    """Service class for computing Precision vs Confidence curves.

    Takes the same input DataFrames as DetectionConfusionMatrix:
    - Ground truth DataFrame: x0, y0, x1, y1, image, label, width, height
    - Detection DataFrame: x0, y0, x1, y1, image, label, score, width, height

    Sweeps the confidence threshold and computes precision at each step,
    producing one curve per class and one averaged over all classes.
    """

    def __init__(
        self,
        ground_truth_df: pd.DataFrame,
        detections_df: pd.DataFrame,
        iou_threshold: float = 0.5,
        num_thresholds: int = 101,
    ):
        """Initialize the precision-confidence curve service.

        Args:
            ground_truth_df: DataFrame with ground truth bounding boxes.
                           Required columns: x0, y0, x1, y1, image, label
            detections_df: DataFrame with predicted bounding boxes.
                          Required columns: x0, y0, x1, y1, image, label, score
            iou_threshold: Minimum IoU threshold for matching (default: 0.5)
            num_thresholds: Number of confidence thresholds to evaluate
                          from 0.0 to 1.0 (default: 101)
        """
        self.ground_truth_df = ground_truth_df.copy()
        self.detections_df = detections_df.copy()
        self.iou_threshold = iou_threshold
        self.num_thresholds = num_thresholds

        # Normalize column names (same logic as DetectionConfusionMatrix)
        self._normalize_columns()

        self._all_classes = sorted(
            set(self.ground_truth_df['label'].unique())
            | set(self.detections_df['label'].unique())
        )
        self._class_names = [f"class_{c}" for c in self._all_classes]

    def _normalize_columns(self) -> None:
        """Normalize column names to handle legacy formats."""
        for df in (self.ground_truth_df, self.detections_df):
            if 'x_min' in df.columns and 'x0' not in df.columns:
                df.rename(columns={
                    'x_min': 'x0', 'y_min': 'y0',
                    'x_max': 'x1', 'y_max': 'y1',
                    'class_id': 'label',
                }, inplace=True)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _compute_tp_fp_at_threshold(
        self, score_threshold: float
    ) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
        """Compute TP, FP, FN per class at a given confidence threshold.

        Returns:
            Tuple of (tp_per_class, fp_per_class, fn_per_class) where each
            dict maps class labels to counts.
        """
        dets = self.detections_df[self.detections_df['score'] >= score_threshold]

        tp: Dict[str, int] = {c: 0 for c in self._all_classes}
        fp: Dict[str, int] = {c: 0 for c in self._all_classes}
        fn: Dict[str, int] = {c: 0 for c in self._all_classes}

        # Pre-count ground truths per class (for FN accounting)
        gt_per_class: Dict[str, int] = self.ground_truth_df['label'].value_counts().to_dict()

        gt_by_image = self.ground_truth_df.groupby('image')
        det_by_image = dets.groupby('image') if len(dets) > 0 else None

        all_images = set(self.ground_truth_df['image'].unique())
        if len(dets) > 0:
            all_images |= set(dets['image'].unique())

        for image_name in all_images:
            gt_boxes = (
                gt_by_image.get_group(image_name)
                if image_name in gt_by_image.groups
                else pd.DataFrame()
            )
            det_boxes = (
                det_by_image.get_group(image_name)
                if det_by_image is not None and image_name in det_by_image.groups
                else pd.DataFrame()
            )

            if len(gt_boxes) > 0 and len(det_boxes) > 0:
                # Compute IoU matrix
                gt_list = gt_boxes.reset_index(drop=True)
                det_list = det_boxes.reset_index(drop=True)
                iou_mat = np.zeros((len(det_list), len(gt_list)))

                for d_idx, det_row in det_list.iterrows():
                    det_box = (det_row['x0'], det_row['y0'], det_row['x1'], det_row['y1'])
                    for g_idx, gt_row in gt_list.iterrows():
                        gt_box = (gt_row['x0'], gt_row['y0'], gt_row['x1'], gt_row['y1'])
                        iou_mat[d_idx, g_idx] = DetectionConfusionMatrix.compute_iou(
                            det_box, gt_box
                        )

                # Greedy matching (highest IoU first)
                used_gt: set = set()
                used_det: set = set()
                flat_ious = sorted(
                    ((iou_mat[d, g], d, g) for d in range(len(det_list)) for g in range(len(gt_list))),
                    reverse=True,
                )
                matches: List[Tuple[int, int]] = []
                for iou_val, d_idx, g_idx in flat_ious:
                    if d_idx in used_det or g_idx in used_gt:
                        continue
                    if iou_val >= self.iou_threshold:
                        matches.append((d_idx, g_idx))
                        used_det.add(d_idx)
                        used_gt.add(g_idx)

                # Count TP / misclassifications
                for d_idx, g_idx in matches:
                    det_label = det_list.iloc[d_idx]['label']
                    gt_label = gt_list.iloc[g_idx]['label']
                    if det_label == gt_label:
                        tp[gt_label] += 1
                    else:
                        # Misclassified detection: FP for the predicted class,
                        # FN for the ground-truth class
                        fp[det_label] += 1
                        fn[gt_label] += 1

                # Unmatched detections → FP
                for d_idx in range(len(det_list)):
                    if d_idx not in used_det:
                        fp[det_list.iloc[d_idx]['label']] += 1

                # Unmatched ground truths → FN
                for g_idx in range(len(gt_list)):
                    if g_idx not in used_gt:
                        fn[gt_list.iloc[g_idx]['label']] += 1

            elif len(gt_boxes) > 0:
                for _, gt_row in gt_boxes.iterrows():
                    fn[gt_row['label']] += 1

            elif len(det_boxes) > 0:
                for _, det_row in det_boxes.iterrows():
                    fp[det_row['label']] += 1

        return tp, fp, fn

    def compute(self) -> PrecisionConfidenceResult:
        """Compute precision-confidence curves.

        Returns:
            PrecisionConfidenceResult with curves per class and averaged.
        """
        thresholds = np.linspace(0.0, 1.0, self.num_thresholds)

        per_class_precision: Dict[str, List[float]] = {
            c: [] for c in self._all_classes
        }
        avg_precision: List[float] = []

        for thresh in thresholds:
            tp, fp, fn = self._compute_tp_fp_at_threshold(thresh)

            total_tp = sum(tp.values())
            total_fp = sum(fp.values())

            avg_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            avg_precision.append(avg_prec)

            for c in self._all_classes:
                c_tp = tp.get(c, 0)
                c_fp = fp.get(c, 0)
                c_prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
                per_class_precision[c].append(c_prec)

        return PrecisionConfidenceResult(
            confidence_thresholds=thresholds,
            per_class_precision={
                name: np.array(per_class_precision[label])
                for label, name in zip(self._all_classes, self._class_names)
            },
            average_precision=np.array(avg_precision),
            class_names=self._class_names,
        )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(
        self,
        output_path: str,
        figsize: Tuple[int, int] = (10, 7),
    ) -> None:
        """Plot and save the Precision-Confidence curves.

        Args:
            output_path: Path to save the plot
            figsize: Figure size (width, height) in inches
        """
        import matplotlib.pyplot as plt

        result = self.compute()

        fig, ax = plt.subplots(figsize=figsize)

        for class_name, prec_curve in result.per_class_precision.items():
            ax.plot(result.confidence_thresholds, prec_curve, label=class_name, alpha=0.7)

        ax.plot(
            result.confidence_thresholds,
            result.average_precision,
            label='average',
            linewidth=2,
            linestyle='--',
            color='black',
        )

        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision vs Confidence (IoU={self.iou_threshold})')
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


def compute_precision_confidence_curve(
    ground_truth_df: pd.DataFrame,
    detections_df: pd.DataFrame,
    iou_threshold: float = 0.5,
    num_thresholds: int = 101,
) -> PrecisionConfidenceResult:
    """Convenience function to compute precision-confidence curves.

    Args:
        ground_truth_df: DataFrame with ground truth bounding boxes
        detections_df: DataFrame with predicted bounding boxes
        iou_threshold: Minimum IoU threshold for matching (default: 0.5)
        num_thresholds: Number of confidence thresholds to evaluate (default: 101)

    Returns:
        PrecisionConfidenceResult with curves per class and averaged
    """
    service = PrecisionConfidenceCurve(
        ground_truth_df=ground_truth_df,
        detections_df=detections_df,
        iou_threshold=iou_threshold,
        num_thresholds=num_thresholds,
    )
    return service.compute()
