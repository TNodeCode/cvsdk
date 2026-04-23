"""Precision-Confidence curve computation service for classification evaluation."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ClassificationPrecisionConfidenceResult:
    """Result of precision-confidence curve computation."""

    confidence_thresholds: np.ndarray
    per_class_precision: Dict[str, np.ndarray]
    average_precision: np.ndarray
    class_names: List[str]


class ClassificationPrecisionConfidenceCurve:
    """Precision vs Confidence threshold for classification.

    Sweeps the confidence threshold and computes precision at each step.
    Works with any probability mode (single_column or per_class).
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidence: np.ndarray,
        class_names: List[str],
        num_thresholds: int = 101,
    ):
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.confidence = np.asarray(confidence)
        self.class_names = class_names
        self.num_thresholds = num_thresholds

        self._all_classes = sorted(set(self.y_true.tolist()) | set(self.y_pred.tolist()))

    def compute(self) -> ClassificationPrecisionConfidenceResult:
        thresholds = np.linspace(0.0, 1.0, self.num_thresholds)

        per_class_precision: Dict[str, List[float]] = {
            c: [] for c in self._all_classes
        }
        avg_precision: List[float] = []

        for thresh in thresholds:
            mask = self.confidence >= thresh
            if mask.sum() == 0:
                for c in self._all_classes:
                    per_class_precision[c].append(0.0)
                avg_precision.append(0.0)
                continue

            y_true_f = self.y_true[mask]
            y_pred_f = self.y_pred[mask]

            total_tp = int((y_true_f == y_pred_f).sum())
            total_fp = int((y_true_f != y_pred_f).sum())
            avg_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            avg_precision.append(avg_prec)

            for c in self._all_classes:
                c_mask_true = y_true_f == c
                c_mask_pred = y_pred_f == c
                c_tp = int((c_mask_true & c_mask_pred).sum())
                c_fp = int((~c_mask_true & c_mask_pred).sum())
                c_prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
                per_class_precision[c].append(c_prec)

        name_map = {c: n for c, n in zip(self._all_classes, self.class_names)}

        return ClassificationPrecisionConfidenceResult(
            confidence_thresholds=thresholds,
            per_class_precision={
                name_map.get(c, str(c)): np.array(per_class_precision[c])
                for c in self._all_classes
            },
            average_precision=np.array(avg_precision),
            class_names=self.class_names,
        )

    def plot(self, output_path: str, figsize: Tuple[int, int] = (10, 7)) -> None:
        import matplotlib.pyplot as plt

        result = self.compute()

        fig, ax = plt.subplots(figsize=figsize)

        for class_name, prec_curve in result.per_class_precision.items():
            ax.plot(result.confidence_thresholds, prec_curve, label=class_name, alpha=0.7)

        ax.plot(
            result.confidence_thresholds,
            result.average_precision,
            label="average",
            linewidth=2,
            linestyle="--",
            color="black",
        )

        ax.set_xlabel("Confidence Score")
        ax.set_ylabel("Precision")
        ax.set_title("Precision vs Confidence")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()


def compute_precision_confidence_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    class_names: List[str],
    num_thresholds: int = 101,
) -> ClassificationPrecisionConfidenceResult:
    service = ClassificationPrecisionConfidenceCurve(
        y_true=y_true,
        y_pred=y_pred,
        confidence=confidence,
        class_names=class_names,
        num_thresholds=num_thresholds,
    )
    return service.compute()
