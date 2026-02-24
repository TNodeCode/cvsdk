import pandas as pd
import glob
import re
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from structlog import get_logger

logger = get_logger()


def read_csv_predictions_as_df(csv_filepath: str) -> pd.DataFrame:
    """Read predictions from CSV file.
    
    Args:
        csv_filepath (str): Path to CSV file
        
    Returns:
        pd.DataFrame: DataFrame with predictions
    """
    return pd.read_csv(csv_filepath)


def read_ground_truth_annotations(ann_file: str) -> pd.DataFrame:
    """Read ground truth annotations.
    
    Args:
        ann_file (str): Path to annotation file
        
    Returns:
        pd.DataFrame: DataFrame with ground truth labels
    """
    # This should be adapted based on the annotation format
    # For now, assume CSV format with columns: filename, label
    return pd.read_csv(ann_file)


def evaluate_predictions(
        gt_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        threshold: float = 0.5,
) -> dict:
    """Evaluate predictions against ground truth.
    
    Args:
        gt_df (pd.DataFrame): Ground truth DataFrame
        pred_df (pd.DataFrame): Predictions DataFrame
        threshold (float, optional): Score threshold. Defaults to 0.5.
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    # Filter predictions by threshold
    pred_df = pred_df[pred_df["pred_score"] >= threshold]
    
    # Merge ground truth and predictions on filename
    merged = pd.merge(gt_df, pred_df, on="filename", how="inner")
    
    if len(merged) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    
    # Calculate metrics
    y_true = merged["label"]
    y_pred = merged["pred_label"]
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    
    return metrics


def evaluate_training_epochs(
        gt_file_path: str,
        model_epoch_filename_func,
        max_epochs: int,
        score_threshold: float = 0.5,
):
    """Evaluate model performance across training epochs.
    
    Args:
        gt_file_path (str): Path to ground truth annotations
        model_epoch_filename_func: Function to generate prediction file path for each epoch
        max_epochs (int): Maximum number of epochs to evaluate
        score_threshold (float, optional): Score threshold. Defaults to 0.5.
        
    Returns:
        pd.DataFrame: DataFrame with metrics for each epoch
    """
    # In this list we store the metrics of all epochs
    epoch_metrics = []

    # These are the ground truth annotations the predictions are evaluated against
    gt_df = read_ground_truth_annotations(gt_file_path)

    for i in range(1, max_epochs + 1):
        # Build the filename of the file where predictions of i-th epoch can be found
        filename = model_epoch_filename_func(i)
        if not os.path.exists(filename):
            continue
        logger.info(f"Processing epoch {i} ...")
        # Read this CSV file
        pred_df = read_csv_predictions_as_df(filename)
        
        # Compute the metrics for the i-th epoch
        metrics = evaluate_predictions(
            gt_df=gt_df,
            pred_df=pred_df,
            threshold=score_threshold
        )
        metrics["epoch"] = i
        # Add metrics of i-th epoch to list
        epoch_metrics.append(metrics)
    return pd.DataFrame(epoch_metrics)


def evaluate(
        gt_file_path: str,
        model_name: str,
        model_type: str,
        csv_file_pattern: str,
        results_file: str,
        max_epochs: int,
        score_threshold: float = 0.5,
        work_dir: str = "work_dirs"
):
    """Evaluate model across epochs.
    
    Args:
        gt_file_path (str): Path to ground truth annotations
        model_name (str): Name of the model
        model_type (str): Type of the model
        csv_file_pattern (str): Pattern for prediction CSV files ($i will be replaced by epoch number)
        results_file (str): Path to save evaluation results
        max_epochs (int): Maximum number of epochs to evaluate
        score_threshold (float, optional): Score threshold. Defaults to 0.5.
        work_dir (str, optional): Working directory. Defaults to "work_dirs".
    """
    runs_dir = f"{work_dir}/{model_type}/{model_name}"
    df_metrics = evaluate_training_epochs(
        gt_file_path=gt_file_path,
        model_epoch_filename_func=lambda i: runs_dir + "/" + csv_file_pattern.replace("$i", str(i)),
        max_epochs=max_epochs,
        score_threshold=score_threshold,
    )
    results_filename = runs_dir + "/" + results_file
    os.makedirs(os.path.split(results_filename)[0], exist_ok=True)
    
    df_metrics.to_csv(results_filename, index=False)
    logger.info("Evaluation results saved", filename=results_filename)
