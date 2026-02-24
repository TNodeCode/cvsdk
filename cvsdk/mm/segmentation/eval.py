import pandas as pd
import glob
import os
import numpy as np
from structlog import get_logger

logger = get_logger()


def calculate_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, num_classes: int) -> dict:
    """Calculate Intersection over Union for segmentation masks.
    
    Args:
        pred_mask: Predicted segmentation mask
        gt_mask: Ground truth segmentation mask
        num_classes: Number of classes
        
    Returns:
        Dictionary with IoU metrics per class and mean IoU
    """
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred_mask == cls)
        gt_cls = (gt_mask == cls)
        
        intersection = np.logical_and(pred_cls, gt_cls).sum()
        union = np.logical_or(pred_cls, gt_cls).sum()
        
        if union > 0:
            iou = intersection / union
            ious.append(iou)
        else:
            ious.append(np.nan)
    
    return {
        'ious_per_class': ious,
        'mean_iou': np.nanmean(ious)
    }


def evaluate_segmentation(
        pred_results_file: str,
        gt_masks_dir: str,
        num_classes: int,
        results_file: str,
        work_dir: str = "work_dirs"
) -> None:
    """Evaluate segmentation predictions.
    
    Args:
        pred_results_file: CSV file with prediction results
        gt_masks_dir: Directory containing ground truth segmentation masks
        num_classes: Number of segmentation classes
        results_file: Output file for evaluation metrics
        work_dir: Working directory
    """
    # Read predictions CSV
    pred_df = pd.read_csv(f"{work_dir}/{pred_results_file}")
    
    evaluation_results = []
    
    for _, row in pred_df.iterrows():
        epoch = row['epoch']
        filename = row['filename']
        
        # Load predicted segmentation map
        pred_path = f"{work_dir}/{pred_results_file}".replace('.csv', f'_{filename.replace("/", "_")}.npy')
        if not os.path.exists(pred_path):
            logger.warning(f"Prediction file not found: {pred_path}")
            continue
            
        pred_mask = np.load(pred_path)
        
        # Load ground truth segmentation map
        gt_path = os.path.join(gt_masks_dir, filename.replace('.jpg', '.png').replace('.jpeg', '.png'))
        if not os.path.exists(gt_path):
            logger.warning(f"Ground truth file not found: {gt_path}")
            continue
            
        # For simplicity, assume gt is an image that needs to be loaded
        # In practice, you'd use appropriate loading based on your format
        from PIL import Image
        gt_mask = np.array(Image.open(gt_path))
        
        # Calculate metrics
        metrics = calculate_iou(pred_mask.squeeze(), gt_mask, num_classes)
        
        evaluation_results.append({
            'epoch': epoch,
            'filename': filename,
            'mean_iou': metrics['mean_iou']
        })
    
    # Save evaluation results
    results_path = f"{work_dir}/{results_file}"
    os.makedirs(os.path.split(results_path)[0], exist_ok=True)
    pd.DataFrame(evaluation_results).to_csv(results_path, index=False)
    logger.info("Saved evaluation results", results_path=results_path)


def evaluate(
        gt_masks_dir: str,
        model_name: str,
        model_type: str,
        csv_file_pattern: str,
        results_file: str,
        num_classes: int,
        max_epochs: int,
        work_dir: str = "work_dirs"
):
    """Evaluate trained segmentation model.
    
    Args:
        gt_masks_dir: Directory with ground truth masks
        model_name: Name of the model
        model_type: Type of the model
        csv_file_pattern: Pattern for CSV files
        results_file: Name of results file
        num_classes: Number of segmentation classes
        max_epochs: Maximum number of epochs to evaluate
        work_dir: Working directory
    """
    runs_dir = f"{work_dir}/{model_type}/{model_name}"
    
    epoch_metrics = []
    
    for i in range(1, max_epochs+1):
        pred_file = runs_dir + "/" + csv_file_pattern.replace("$i", str(i))
        if not os.path.exists(pred_file):
            continue
            
        logger.info(f"Evaluating epoch {i}")
        evaluate_segmentation(
            pred_results_file=pred_file,
            gt_masks_dir=gt_masks_dir,
            num_classes=num_classes,
            results_file=f"epoch_{i}_eval.csv",
            work_dir=runs_dir
        )
    
    results_filename = runs_dir + "/" + results_file
    logger.info("Evaluation complete", results_file=results_filename)
