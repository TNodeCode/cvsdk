"""Precision-Recall curve computation service for classification evaluation."""
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import auc, precision_recall_curve


@dataclass
class ClassificationPRResult:
    """Result of classification precision-recall curve computation."""

    precision: Dict[str, np.ndarray]
    recall: Dict[str, np.ndarray]
    auc: Dict[str, float]
    class_names: List[str]


class ClassificationPrecisionRecallCurve:
    """Precision-Recall curve for classification (one-vs-rest).

    Requires per-class probabilities.
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        class_names: List[str],
    ):
        self.y_true = np.asarray(y_true)
        self.y_prob = np.asarray(y_prob)
        self.class_names = class_names
        self._all_classes = sorted(set(self.y_true.tolist()))

    def compute(self) -> ClassificationPRResult:
        n_classes = len(self._all_classes)
        y_bin = np.zeros((len(self.y_true), n_classes))
        for i, cls in enumerate(self._all_classes):
            y_bin[:, i] = (self.y_true == cls).astype(int)

        precision: Dict[str, np.ndarray] = {}
        recall: Dict[str, np.ndarray] = {}
        aucs: Dict[str, float] = {}

        name_map = {c: n for c, n in zip(self._all_classes, self.class_names)}

        for i, cls in enumerate(self._all_classes):
            prec, rec, _ = precision_recall_curve(y_bin[:, i], self.y_prob[:, i])
            name = name_map.get(cls, str(cls))
            precision[name] = prec
            recall[name] = rec
            aucs[name] = auc(rec, prec)

        return ClassificationPRResult(
            precision=precision,
            recall=recall,
            auc=aucs,
            class_names=self.class_names,
        )

    def plot(self, output_path: str, figsize: Tuple[int, int] = (10, 7)) -> None:
        import matplotlib.pyplot as plt

        result = self.compute()

        fig, ax = plt.subplots(figsize=figsize)

        for class_name in result.class_names:
            ax.plot(
                result.recall[class_name],
                result.precision[class_name],
                label=f"{class_name} (AUC={result.auc[class_name]:.3f})",
                alpha=0.7,
            )

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves (One-vs-Rest)")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()


def compute_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
) -> ClassificationPRResult:
    service = ClassificationPrecisionRecallCurve(
        y_true=y_true, y_prob=y_prob, class_names=class_names
    )
    return service.compute()
