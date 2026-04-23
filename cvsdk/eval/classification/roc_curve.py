"""ROC curve computation service for classification evaluation (one-vs-rest)."""
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import auc, roc_curve


@dataclass
class ROCCurveResult:
    """Result of ROC curve computation."""

    fpr: Dict[str, np.ndarray]
    tpr: Dict[str, np.ndarray]
    auc: Dict[str, float]
    class_names: List[str]
    fpr_micro: np.ndarray
    tpr_micro: np.ndarray
    auc_micro: float


class ROCCurve:
    """One-vs-rest ROC curves for multi-class classification.

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

    def compute(self) -> ROCCurveResult:
        # Build one-hot matrix manually (label_binarize collapses binary to 1 column)
        n_classes = len(self._all_classes)
        y_bin = np.zeros((len(self.y_true), n_classes))
        for i, cls in enumerate(self._all_classes):
            y_bin[:, i] = (self.y_true == cls).astype(int)

        # Micro-average
        fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), self.y_prob.ravel())
        auc_micro = auc(fpr_micro, tpr_micro)

        fpr: Dict[str, np.ndarray] = {}
        tpr: Dict[str, np.ndarray] = {}
        aucs: Dict[str, float] = {}

        name_map = {c: n for c, n in zip(self._all_classes, self.class_names)}

        for i, cls in enumerate(self._all_classes):
            fpr[cls], tpr[cls], _ = roc_curve(y_bin[:, i], self.y_prob[:, i])
            aucs[cls] = auc(fpr[cls], tpr[cls])

        return ROCCurveResult(
            fpr={name_map.get(c, str(c)): v for c, v in fpr.items()},
            tpr={name_map.get(c, str(c)): v for c, v in tpr.items()},
            auc={name_map.get(c, str(c)): v for c, v in aucs.items()},
            class_names=self.class_names,
            fpr_micro=fpr_micro,
            tpr_micro=tpr_micro,
            auc_micro=auc_micro,
        )

    def plot(self, output_path: str, figsize: Tuple[int, int] = (10, 7)) -> None:
        import matplotlib.pyplot as plt

        result = self.compute()

        fig, ax = plt.subplots(figsize=figsize)

        for class_name in result.class_names:
            ax.plot(
                result.fpr[class_name],
                result.tpr[class_name],
                label=f"{class_name} (AUC={result.auc[class_name]:.3f})",
                alpha=0.7,
            )

        ax.plot(
            result.fpr_micro,
            result.tpr_micro,
            label=f"micro-average (AUC={result.auc_micro:.3f})",
            linewidth=2,
            linestyle="--",
            color="black",
        )

        ax.plot([0, 1], [0, 1], "k:", alpha=0.3)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves (One-vs-Rest)")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()


def compute_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
) -> ROCCurveResult:
    service = ROCCurve(y_true=y_true, y_prob=y_prob, class_names=class_names)
    return service.compute()
