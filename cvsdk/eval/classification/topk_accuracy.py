"""Top-k accuracy curve computation service for classification evaluation."""
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class TopKAccuracyResult:
    """Result of top-k accuracy computation."""

    k_values: List[int]
    overall_accuracy: Dict[int, float]
    per_class_accuracy: Dict[str, Dict[int, float]]
    class_names: List[str]


class TopKAccuracyCurve:
    """Top-k accuracy curves per class.

    For each class, computes top-k accuracy: the fraction of samples
    where the true class is among the top-k predicted classes.
    Requires per-class probabilities.
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        class_names: List[str],
        max_k: int = 5,
    ):
        self.y_true = np.asarray(y_true)
        self.y_prob = np.asarray(y_prob)
        self.class_names = class_names
        self.max_k = max_k
        self._all_classes = sorted(set(self.y_true.tolist()))

    def compute(self) -> TopKAccuracyResult:
        k_values = list(range(1, self.max_k + 1))

        # Map class labels to indices
        class_to_idx = {c: i for i, c in enumerate(self._all_classes)}

        # Convert y_true to indices
        y_true_idx = np.array([class_to_idx[y] for y in self.y_true])

        # Get sorted indices (descending probability)
        sorted_indices = np.argsort(-self.y_prob, axis=1)

        overall_accuracy: Dict[int, float] = {}
        per_class_accuracy: Dict[str, Dict[int, float]] = {
            n: {} for n in self.class_names
        }

        name_map = {c: n for c, n in zip(self._all_classes, self.class_names)}

        for k in k_values:
            top_k = sorted_indices[:, :k]
            correct = np.array([y_true_idx[i] in top_k[i] for i in range(len(y_true_idx))])
            overall_accuracy[k] = float(correct.mean())

            for cls in self._all_classes:
                name = name_map.get(cls, str(cls))
                cls_idx = class_to_idx[cls]
                cls_mask = y_true_idx == cls_idx
                if cls_mask.sum() > 0:
                    per_class_accuracy[name][k] = float(correct[cls_mask].mean())
                else:
                    per_class_accuracy[name][k] = 0.0

        return TopKAccuracyResult(
            k_values=k_values,
            overall_accuracy=overall_accuracy,
            per_class_accuracy=per_class_accuracy,
            class_names=self.class_names,
        )

    def plot(self, output_path: str, figsize: Tuple[int, int] = (10, 7)) -> None:
        import matplotlib.pyplot as plt

        result = self.compute()

        fig, ax = plt.subplots(figsize=figsize)

        for class_name in result.class_names:
            accs = [result.per_class_accuracy[class_name][k] for k in result.k_values]
            ax.plot(result.k_values, accs, label=class_name, marker="o", alpha=0.7)

        overall_accs = [result.overall_accuracy[k] for k in result.k_values]
        ax.plot(
            result.k_values,
            overall_accs,
            label="overall",
            linewidth=2,
            linestyle="--",
            color="black",
            marker="s",
        )

        ax.set_xlabel("k")
        ax.set_ylabel("Accuracy")
        ax.set_title("Top-k Accuracy Curves")
        ax.set_xticks(result.k_values)
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()


def compute_topk_accuracy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    max_k: int = 5,
) -> TopKAccuracyResult:
    service = TopKAccuracyCurve(
        y_true=y_true, y_prob=y_prob, class_names=class_names, max_k=max_k
    )
    return service.compute()
