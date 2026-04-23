"""Recall-Confidence curve computation service for classification evaluation."""
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class ClassificationRecallConfidenceResult:
    """Result of recall-confidence curve computation."""

    confidence_thresholds: np.ndarray
    per_class_recall: Dict[str, np.ndarray]
    average_recall: np.ndarray
    class_names: List[str]


class ClassificationRecallConfidenceCurve:
    """Recall vs Confidence threshold for classification.

    Sweeps the confidence threshold and computes recall at each step.
    Samples below the threshold are treated as false negatives.
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

    def compute(self) -> ClassificationRecallConfidenceResult:
        thresholds = np.linspace(0.0, 1.0, self.num_thresholds)

        per_class_recall: Dict[str, List[float]] = {
            c: [] for c in self._all_classes
        }
        avg_recall: List[float] = []

        total_samples = len(self.y_true)

        for thresh in thresholds:
            mask = self.confidence >= thresh
            if mask.sum() == 0:
                for c in self._all_classes:
                    per_class_recall[c].append(0.0)
                avg_recall.append(0.0)
                continue

            y_true_f = self.y_true[mask]
            y_pred_f = self.y_pred[mask]

            total_tp = int((y_true_f == y_pred_f).sum())
            avg_rec = total_tp / total_samples if total_samples > 0 else 0.0
            avg_recall.append(avg_rec)

            for c in self._all_classes:
                c_mask_true = self.y_true == c
                c_tp = int(((y_true_f == c) & (y_pred_f == c)).sum())
                c_total = int(c_mask_true.sum())
                c_rec = c_tp / c_total if c_total > 0 else 0.0
                per_class_recall[c].append(c_rec)

        name_map = {c: n for c, n in zip(self._all_classes, self.class_names)}

        return ClassificationRecallConfidenceResult(
            confidence_thresholds=thresholds,
            per_class_recall={
                name_map.get(c, str(c)): np.array(per_class_recall[c])
                for c in self._all_classes
            },
            average_recall=np.array(avg_recall),
            class_names=self.class_names,
        )

    def plot(self, output_path: str, figsize: Tuple[int, int] = (10, 7)) -> None:
        import matplotlib.pyplot as plt

        result = self.compute()

        fig, ax = plt.subplots(figsize=figsize)

        for class_name, rec_curve in result.per_class_recall.items():
            ax.plot(result.confidence_thresholds, rec_curve, label=class_name, alpha=0.7)

        ax.plot(
            result.confidence_thresholds,
            result.average_recall,
            label="average",
            linewidth=2,
            linestyle="--",
            color="black",
        )

        ax.set_xlabel("Confidence Score")
        ax.set_ylabel("Recall")
        ax.set_title("Recall vs Confidence")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()


def compute_recall_confidence_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    class_names: List[str],
    num_thresholds: int = 101,
) -> ClassificationRecallConfidenceResult:
    service = ClassificationRecallConfidenceCurve(
        y_true=y_true,
        y_pred=y_pred,
        confidence=confidence,
        class_names=class_names,
        num_thresholds=num_thresholds,
    )
    return service.compute()
