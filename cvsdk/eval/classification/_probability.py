"""Probability specification helper for classification evaluation."""
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ProbabilitySpec:
    """Encapsulates predicted probability information for classification.

    Three modes:
      1. none          - no probabilities available
      2. single_column - one column of max-class probabilities
      3. per_class     - dict mapping class name -> column name, one probability per class
    """

    mode: str
    single_column: Optional[str] = None
    per_class_columns: Optional[Dict[str, str]] = None
    class_names: Optional[List[str]] = None

    @staticmethod
    def none() -> "ProbabilitySpec":
        return ProbabilitySpec(mode="none")

    @staticmethod
    def from_single_column(col_name: str) -> "ProbabilitySpec":
        return ProbabilitySpec(mode="single_column", single_column=col_name)

    @staticmethod
    def from_prefix(
        pred_df: pd.DataFrame, prefix: str, class_names: List[str]
    ) -> "ProbabilitySpec":
        mapping = {}
        for cls in class_names:
            col = f"{prefix}{cls}"
            if col not in pred_df.columns:
                raise ValueError(
                    f"Expected probability column '{col}' not found. "
                    f"Available columns: {list(pred_df.columns)}"
                )
            mapping[cls] = col
        return ProbabilitySpec(
            mode="per_class", per_class_columns=mapping, class_names=class_names
        )

    @staticmethod
    def from_column_list(
        pred_df: pd.DataFrame, column_names: Dict[str, str]
    ) -> "ProbabilitySpec":
        for cls, col in column_names.items():
            if col not in pred_df.columns:
                raise ValueError(
                    f"Probability column '{col}' for class '{cls}' not found. "
                    f"Available columns: {list(pred_df.columns)}"
                )
        return ProbabilitySpec(
            mode="per_class",
            per_class_columns=column_names,
            class_names=list(column_names.keys()),
        )

    @property
    def has_probabilities(self) -> bool:
        return self.mode != "none"

    @property
    def has_per_class(self) -> bool:
        return self.mode == "per_class"

    def get_probability_matrix(self, pred_df: pd.DataFrame) -> np.ndarray:
        if not self.has_per_class:
            raise ValueError("Per-class probabilities not available.")
        return pred_df[
            [self.per_class_columns[c] for c in self.class_names]
        ].values

    def get_confidence(self, pred_df: pd.DataFrame) -> np.ndarray:
        if self.mode == "single_column":
            return pred_df[self.single_column].values
        elif self.mode == "per_class":
            return self.get_probability_matrix(pred_df).max(axis=1)
        else:
            raise ValueError("No probabilities available.")
