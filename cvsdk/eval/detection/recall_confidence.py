"""Recall-Confidence curve computation service for object detection evaluation."""
import numpy as np
from typing import Dict, List

from cvsdk.eval.detection.precision_confidence import PrecisionConfidenceCurve


class RecallConfidenceCurve(PrecisionConfidenceCurve):
    """Service class for computing Recall vs Confidence curves.

    Inherits the threshold-sweep and matching logic from
    PrecisionConfidenceCurve and derives recall instead of precision.

    Takes the same input DataFrames:
    - Ground truth DataFrame: x0, y0, x1, y1, image, label, width, height
    - Detection DataFrame: x0, y0, x1, y1, image, label, score, width, height

    Sweeps the confidence threshold and computes recall at each step,
    producing one curve per class and one averaged over all classes.
    """

    def compute(self):
        """Compute recall-confidence curves.

        Returns:
            PrecisionConfidenceResult (reused as container) with recall values
            per class and averaged.
        """
        from cvsdk.eval.detection.precision_confidence import PrecisionConfidenceResult

        thresholds = np.linspace(0.0, 1.0, self.num_thresholds)

        per_class_recall: Dict[str, List[float]] = {
            c: [] for c in self._all_classes
        }
        avg_recall: List[float] = []

        for thresh in thresholds:
            tp, fp, fn = self._compute_tp_fp_at_threshold(thresh)

            total_tp = sum(tp.values())
            total_fn = sum(fn.values())

            avg_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            avg_recall.append(avg_rec)

            for c in self._all_classes:
                c_tp = tp.get(c, 0)
                c_fn = fn.get(c, 0)
                c_rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
                per_class_recall[c].append(c_rec)

        return PrecisionConfidenceResult(
            confidence_thresholds=thresholds,
            per_class_precision={
                name: np.array(per_class_recall[label])
                for label, name in zip(self._all_classes, self._class_names)
            },
            average_precision=np.array(avg_recall),
            class_names=self._class_names,
        )

    def plot(self, output_path: str, figsize=(10, 7)) -> None:
        """Plot and save the Recall-Confidence curves.

        Args:
            output_path: Path to save the plot
            figsize: Figure size (width, height) in inches
        """
        import matplotlib.pyplot as plt

        result = self.compute()

        fig, ax = plt.subplots(figsize=figsize)

        for class_name, rec_curve in result.per_class_precision.items():
            ax.plot(result.confidence_thresholds, rec_curve, label=class_name, alpha=0.7)

        ax.plot(
            result.confidence_thresholds,
            result.average_precision,
            label='average',
            linewidth=2,
            linestyle='--',
            color='black',
        )

        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Recall')
        ax.set_title(f'Recall vs Confidence (IoU={self.iou_threshold})')
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


def compute_recall_confidence_curve(
    ground_truth_df,
    detections_df,
    iou_threshold: float = 0.5,
    num_thresholds: int = 101,
):
    """Convenience function to compute recall-confidence curves.

    Args:
        ground_truth_df: DataFrame with ground truth bounding boxes
        detections_df: DataFrame with predicted bounding boxes
        iou_threshold: Minimum IoU threshold for matching (default: 0.5)
        num_thresholds: Number of confidence thresholds to evaluate (default: 101)

    Returns:
        PrecisionConfidenceResult with recall curves per class and averaged
    """
    service = RecallConfidenceCurve(
        ground_truth_df=ground_truth_df,
        detections_df=detections_df,
        iou_threshold=iou_threshold,
        num_thresholds=num_thresholds,
    )
    return service.compute()
