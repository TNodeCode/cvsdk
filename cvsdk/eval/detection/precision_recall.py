"""Precision-Recall curve computation service for object detection evaluation."""
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

from cvsdk.eval.detection.precision_confidence import PrecisionConfidenceCurve


@dataclass
class PrecisionRecallResult:
    """Result of precision-recall curve computation."""
    confidence_thresholds: np.ndarray
    per_class_precision: Dict[str, np.ndarray]
    per_class_recall: Dict[str, np.ndarray]
    average_precision: np.ndarray
    average_recall: np.ndarray
    class_names: List[str]


class PrecisionRecallCurve(PrecisionConfidenceCurve):
    """Service class for computing Precision vs Recall curves.

    Inherits the threshold-sweep and matching logic from
    PrecisionConfusionCurve and derives both precision and recall at each
    confidence threshold.

    Takes the same input DataFrames:
    - Ground truth DataFrame: x0, y0, x1, y1, image, label, width, height
    - Detection DataFrame: x0, y0, x1, y1, image, label, score, width, height

    Sweeps the confidence threshold, computes precision and recall at each
    step, and plots recall on the x-axis and precision on the y-axis —
    one curve per class and one averaged over all classes.
    """

    def compute(self) -> PrecisionRecallResult:
        """Compute precision-recall curves.

        Returns:
            PrecisionRecallResult with precision and recall curves per class
            and averaged.
        """
        thresholds = np.linspace(0.0, 1.0, self.num_thresholds)

        per_class_precision: Dict[str, List[float]] = {
            c: [] for c in self._all_classes
        }
        per_class_recall: Dict[str, List[float]] = {
            c: [] for c in self._all_classes
        }
        avg_precision: List[float] = []
        avg_recall: List[float] = []

        for thresh in thresholds:
            tp, fp, fn = self._compute_tp_fp_at_threshold(thresh)

            total_tp = sum(tp.values())
            total_fp = sum(fp.values())
            total_fn = sum(fn.values())

            avg_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            avg_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            avg_precision.append(avg_prec)
            avg_recall.append(avg_rec)

            for c in self._all_classes:
                c_tp = tp.get(c, 0)
                c_fp = fp.get(c, 0)
                c_fn = fn.get(c, 0)
                c_prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
                c_rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
                per_class_precision[c].append(c_prec)
                per_class_recall[c].append(c_rec)

        return PrecisionRecallResult(
            confidence_thresholds=thresholds,
            per_class_precision={
                name: np.array(per_class_precision[label])
                for label, name in zip(self._all_classes, self._class_names)
            },
            per_class_recall={
                name: np.array(per_class_recall[label])
                for label, name in zip(self._all_classes, self._class_names)
            },
            average_precision=np.array(avg_precision),
            average_recall=np.array(avg_recall),
            class_names=self._class_names,
        )

    def plot(self, output_path: str, figsize=(10, 7)) -> None:
        """Plot and save the Precision-Recall curves.

        Args:
            output_path: Path to save the plot
            figsize: Figure size (width, height) in inches
        """
        import matplotlib.pyplot as plt

        result = self.compute()

        fig, ax = plt.subplots(figsize=figsize)

        for class_name in result.class_names:
            ax.plot(
                result.per_class_recall[class_name],
                result.per_class_precision[class_name],
                label=class_name,
                alpha=0.7,
            )

        ax.plot(
            result.average_recall,
            result.average_precision,
            label='average',
            linewidth=2,
            linestyle='--',
            color='black',
        )

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision vs Recall (IoU={self.iou_threshold})')
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


def compute_precision_recall_curve(
    ground_truth_df,
    detections_df,
    iou_threshold: float = 0.5,
    num_thresholds: int = 101,
) -> PrecisionRecallResult:
    """Convenience function to compute precision-recall curves.

    Args:
        ground_truth_df: DataFrame with ground truth bounding boxes
        detections_df: DataFrame with predicted bounding boxes
        iou_threshold: Minimum IoU threshold for matching (default: 0.5)
        num_thresholds: Number of confidence thresholds to evaluate (default: 101)

    Returns:
        PrecisionRecallResult with precision and recall curves per class and averaged
    """
    service = PrecisionRecallCurve(
        ground_truth_df=ground_truth_df,
        detections_df=detections_df,
        iou_threshold=iou_threshold,
        num_thresholds=num_thresholds,
    )
    return service.compute()
