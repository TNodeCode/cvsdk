"""Classification evaluation metrics."""
from cvsdk.eval.classification._probability import ProbabilitySpec
from cvsdk.eval.classification.confusion_matrix import (
    ClassificationConfusionMatrix,
    ClassificationConfusionMatrixResult,
    compute_confusion_matrix,
)
from cvsdk.eval.classification.roc_curve import (
    ROCCurve,
    ROCCurveResult,
    compute_roc_curve,
)
from cvsdk.eval.classification.precision_recall import (
    ClassificationPrecisionRecallCurve,
    ClassificationPRResult,
    compute_precision_recall_curve,
)
from cvsdk.eval.classification.precision_confidence import (
    ClassificationPrecisionConfidenceCurve,
    ClassificationPrecisionConfidenceResult,
    compute_precision_confidence_curve,
)
from cvsdk.eval.classification.recall_confidence import (
    ClassificationRecallConfidenceCurve,
    ClassificationRecallConfidenceResult,
    compute_recall_confidence_curve,
)
from cvsdk.eval.classification.f1_confidence import (
    ClassificationF1ConfidenceCurve,
    ClassificationF1ConfidenceResult,
    compute_f1_confidence_curve,
)
from cvsdk.eval.classification.reliability_diagram import (
    ReliabilityDiagram,
    ReliabilityDiagramResult,
    compute_reliability_diagram,
)
from cvsdk.eval.classification.probability_kde import (
    ProbabilityKDE,
    ProbabilityKDEResult,
    compute_probability_kde,
)
from cvsdk.eval.classification.topk_accuracy import (
    TopKAccuracyCurve,
    TopKAccuracyResult,
    compute_topk_accuracy,
)

__all__ = [
    "ProbabilitySpec",
    "ClassificationConfusionMatrix",
    "ClassificationConfusionMatrixResult",
    "compute_confusion_matrix",
    "ROCCurve",
    "ROCCurveResult",
    "compute_roc_curve",
    "ClassificationPrecisionRecallCurve",
    "ClassificationPRResult",
    "compute_precision_recall_curve",
    "ClassificationPrecisionConfidenceCurve",
    "ClassificationPrecisionConfidenceResult",
    "compute_precision_confidence_curve",
    "ClassificationRecallConfidenceCurve",
    "ClassificationRecallConfidenceResult",
    "compute_recall_confidence_curve",
    "ClassificationF1ConfidenceCurve",
    "ClassificationF1ConfidenceResult",
    "compute_f1_confidence_curve",
    "ReliabilityDiagram",
    "ReliabilityDiagramResult",
    "compute_reliability_diagram",
    "ProbabilityKDE",
    "ProbabilityKDEResult",
    "compute_probability_kde",
    "TopKAccuracyCurve",
    "TopKAccuracyResult",
    "compute_topk_accuracy",
]
