"""Reliability diagram computation service for classification evaluation."""
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class ReliabilityDiagramResult:
    """Result of reliability diagram computation."""

    mean_predicted: Dict[str, np.ndarray]
    mean_observed: Dict[str, np.ndarray]
    bin_edges: np.ndarray
    class_names: List[str]
    ece: Dict[str, float]


class ReliabilityDiagram:
    """Calibration reliability diagram: predicted probability vs observed frequency.

    Bins predicted probabilities and compares to observed class frequencies.
    Requires per-class probabilities.
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        class_names: List[str],
        n_bins: int = 10,
    ):
        self.y_true = np.asarray(y_true)
        self.y_prob = np.asarray(y_prob)
        self.class_names = class_names
        self.n_bins = n_bins
        self._all_classes = sorted(set(self.y_true.tolist()))

    def compute(self) -> ReliabilityDiagramResult:
        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)

        mean_predicted: Dict[str, np.ndarray] = {}
        mean_observed: Dict[str, np.ndarray] = {}
        ece: Dict[str, float] = {}

        name_map = {c: n for c, n in zip(self._all_classes, self.class_names)}
        total = len(self.y_true)

        for i, cls in enumerate(self._all_classes):
            name = name_map.get(cls, str(cls))
            y_bin = (self.y_true == cls).astype(float)
            prob = self.y_prob[:, i]

            preds = np.zeros(self.n_bins)
            obs = np.zeros(self.n_bins)
            counts = np.zeros(self.n_bins)

            for b in range(self.n_bins):
                mask = (prob >= bin_edges[b]) & (prob < bin_edges[b + 1])
                if b == self.n_bins - 1:
                    mask = mask | (prob == bin_edges[b + 1])
                count = mask.sum()
                counts[b] = count
                if count > 0:
                    preds[b] = prob[mask].mean()
                    obs[b] = y_bin[mask].mean()

            # ECE
            ece_val = np.sum(counts / total * np.abs(preds - obs))
            ece[name] = float(ece_val)
            mean_predicted[name] = preds
            mean_observed[name] = obs

        return ReliabilityDiagramResult(
            mean_predicted=mean_predicted,
            mean_observed=mean_observed,
            bin_edges=bin_edges,
            class_names=self.class_names,
            ece=ece,
        )

    def plot(self, output_path: str, figsize: Tuple[int, int] = (10, 7)) -> None:
        import matplotlib.pyplot as plt

        result = self.compute()
        n_classes = len(result.class_names)

        n_cols = min(3, n_classes)
        n_rows = (n_classes + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows / max(n_cols, 1)))
        if n_classes == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        bin_centers = (result.bin_edges[:-1] + result.bin_edges[1:]) / 2
        bar_width = (result.bin_edges[1] - result.bin_edges[0]) * 0.8

        for idx, class_name in enumerate(result.class_names):
            ax = axes[idx]
            ax.bar(
                bin_centers,
                result.mean_predicted[class_name],
                width=bar_width,
                alpha=0.6,
                label="Predicted prob",
            )
            ax.plot(
                [0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration"
            )
            gaps = result.mean_observed[class_name] - result.mean_predicted[class_name]
            ax.bar(
                bin_centers,
                gaps,
                width=bar_width,
                bottom=result.mean_predicted[class_name],
                alpha=0.3,
                color="red",
                label="Gap",
            )
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Observed frequency")
            ax.set_title(f"{class_name} (ECE={result.ece[class_name]:.4f})")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8)

        for idx in range(n_classes, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle("Reliability Diagrams", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()


def compute_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    n_bins: int = 10,
) -> ReliabilityDiagramResult:
    service = ReliabilityDiagram(
        y_true=y_true, y_prob=y_prob, class_names=class_names, n_bins=n_bins
    )
    return service.compute()
