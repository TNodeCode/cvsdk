"""Confusion matrix computation service for classification evaluation."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix


@dataclass
class ClassificationConfusionMatrixResult:
    """Result of classification confusion matrix computation."""

    matrix: np.ndarray
    class_names: List[str]
    accuracy: float
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    per_class_f1: Dict[str, float]
    per_class_support: Dict[str, int]
    overall_precision: float
    overall_recall: float
    overall_f1: float


class ClassificationConfusionMatrix:
    """Service class for computing confusion matrices for classification."""

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
    ):
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self._all_classes = sorted(
            set(self.y_true.tolist()) | set(self.y_pred.tolist())
        )
        self.class_names = (
            class_names if class_names is not None
            else [str(c) for c in self._all_classes]
        )

    def compute(self) -> ClassificationConfusionMatrixResult:
        matrix = sklearn_confusion_matrix(
            self.y_true, self.y_pred, labels=self._all_classes
        )

        total = matrix.sum()
        accuracy = np.trace(matrix) / total if total > 0 else 0.0

        per_class_precision = {}
        per_class_recall = {}
        per_class_f1 = {}
        per_class_support = {}

        for i, name in enumerate(self.class_names):
            tp = int(matrix[i, i])
            fp = int(matrix[:, i].sum()) - tp
            fn = int(matrix[i, :].sum()) - tp
            support = int(matrix[i, :].sum())

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            per_class_precision[name] = prec
            per_class_recall[name] = rec
            per_class_f1[name] = f1
            per_class_support[name] = support

        total_tp = int(np.trace(matrix))
        total_fp = int(matrix.sum(axis=0).sum()) - total_tp
        total_fn = int(matrix.sum(axis=1).sum()) - total_tp

        overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = (
            2 * overall_prec * overall_rec / (overall_prec + overall_rec)
            if (overall_prec + overall_rec) > 0 else 0.0
        )

        return ClassificationConfusionMatrixResult(
            matrix=matrix,
            class_names=self.class_names,
            accuracy=accuracy,
            per_class_precision=per_class_precision,
            per_class_recall=per_class_recall,
            per_class_f1=per_class_f1,
            per_class_support=per_class_support,
            overall_precision=overall_prec,
            overall_recall=overall_rec,
            overall_f1=overall_f1,
        )

    def get_metrics_dataframe(self) -> pd.DataFrame:
        result = self.compute()
        rows = []
        for name in result.class_names:
            rows.append({
                "class": name,
                "precision": result.per_class_precision[name],
                "recall": result.per_class_recall[name],
                "f1_score": result.per_class_f1[name],
                "support": result.per_class_support[name],
            })
        rows.append({
            "class": "overall",
            "precision": result.overall_precision,
            "recall": result.overall_recall,
            "f1_score": result.overall_f1,
            "support": int(result.matrix.sum()),
        })
        return pd.DataFrame(rows)

    def plot(self, output_path: str, figsize: Tuple[int, int] = (12, 10)) -> None:
        import matplotlib.pyplot as plt
        import seaborn as sns

        result = self.compute()

        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            result.matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=result.class_names,
            yticklabels=result.class_names,
            ax=ax,
        )

        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")
        ax.set_title(
            f"Confusion Matrix\n"
            f"Accuracy: {result.accuracy:.3f}, "
            f"Precision: {result.overall_precision:.3f}, "
            f"Recall: {result.overall_recall:.3f}, "
            f"F1: {result.overall_f1:.3f}"
        )

        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> ClassificationConfusionMatrixResult:
    service = ClassificationConfusionMatrix(
        y_true=y_true, y_pred=y_pred, class_names=class_names
    )
    return service.compute()
