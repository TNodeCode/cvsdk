"""Probability KDE computation service for classification evaluation."""
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import gaussian_kde


@dataclass
class ProbabilityKDEResult:
    """Result of probability KDE computation."""

    probability_grid: np.ndarray
    densities: Dict[str, Dict[str, np.ndarray]]


class ProbabilityKDE:
    """Kernel Density Estimate of predicted probabilities, split by correct/incorrect.

    For each class, plots the KDE of the predicted probability for that class,
    separated into correct and incorrect predictions.
    Requires per-class probabilities.
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        class_names: List[str],
        n_points: int = 200,
    ):
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.y_prob = np.asarray(y_prob)
        self.class_names = class_names
        self.n_points = n_points
        self._all_classes = sorted(set(self.y_true.tolist()))

    def compute(self) -> ProbabilityKDEResult:
        grid = np.linspace(0.0, 1.0, self.n_points)

        densities: Dict[str, Dict[str, np.ndarray]] = {}
        name_map = {c: n for c, n in zip(self._all_classes, self.class_names)}

        for i, cls in enumerate(self._all_classes):
            name = name_map.get(cls, str(cls))
            prob = self.y_prob[:, i]

            correct_mask = (self.y_true == cls) & (self.y_pred == cls)
            incorrect_mask = (self.y_true == cls) & (self.y_pred != cls)

            result = {}

            correct_probs = prob[correct_mask]
            if len(correct_probs) > 1:
                kde = gaussian_kde(correct_probs)
                kde.set_bandwidth(bw_method="scott")
                result["correct"] = kde(grid)
            elif len(correct_probs) == 1:
                result["correct"] = np.zeros_like(grid)
            else:
                result["correct"] = np.zeros_like(grid)

            incorrect_probs = prob[incorrect_mask]
            if len(incorrect_probs) > 1:
                kde = gaussian_kde(incorrect_probs)
                kde.set_bandwidth(bw_method="scott")
                result["incorrect"] = kde(grid)
            elif len(incorrect_probs) == 1:
                result["incorrect"] = np.zeros_like(grid)
            else:
                result["incorrect"] = np.zeros_like(grid)

            densities[name] = result

        return ProbabilityKDEResult(
            probability_grid=grid,
            densities=densities,
        )

    def plot(self, output_path: str, figsize: Tuple[int, int] = (10, 7)) -> None:
        import matplotlib.pyplot as plt

        result = self.compute()
        n_classes = len(self.class_names)

        n_cols = min(3, n_classes)
        n_rows = (n_classes + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows / max(n_cols, 1)))
        if n_classes == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, class_name in enumerate(self.class_names):
            ax = axes[idx]
            d = result.densities[class_name]

            ax.plot(
                result.probability_grid,
                d["correct"],
                label="Correct",
                color="green",
            )
            ax.fill_between(result.probability_grid, d["correct"], alpha=0.2, color="green")

            ax.plot(
                result.probability_grid,
                d["incorrect"],
                label="Incorrect",
                color="red",
                linestyle="--",
            )
            ax.fill_between(result.probability_grid, d["incorrect"], alpha=0.2, color="red")

            ax.set_xlabel("Predicted probability")
            ax.set_ylabel("Density")
            ax.set_title(class_name)
            ax.legend(fontsize=8)

        for idx in range(n_classes, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle("Probability KDE per Class", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()


def compute_probability_kde(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    n_points: int = 200,
) -> ProbabilityKDEResult:
    service = ProbabilityKDE(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        class_names=class_names,
        n_points=n_points,
    )
    return service.compute()
