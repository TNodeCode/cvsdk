import pandas as pd


def compute_iou_xyxy(box1: pd.Series, box2: pd.Series) -> float:
    """
    Compute Intersection over Union (IoU) of two bounding boxes in XYXY format.

    Each box must have 'xmin', 'ymin', 'xmax', 'ymax'.
    """
    x1_min, y1_min = box1['xmin'], box1['ymin']
    x1_max, y1_max = box1['xmax'], box1['ymax']

    x2_min, y2_min = box2['xmin'], box2['ymin']
    x2_max, y2_max = box2['xmax'], box2['ymax']

    # Intersection coords
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    # Intersection area
    inter_width = max(0.0, inter_xmax - inter_xmin)
    inter_height = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_width * inter_height

    # Union area = area1 + area2 - intersection
    area1 = max(0.0, (x1_max - x1_min)) * max(0.0, (y1_max - y1_min))
    area2 = max(0.0, (x2_max - x2_min)) * max(0.0, (y2_max - y2_min))
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area

class DetectionsEvaluator:
    """Evaluate detected bounding boxes against ground truth annotations stored in XYXY format.

    Methods
    -------
    evaluate(
        detections_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame,
        iou_threshold: float = 0.5
    ) -> pd.DataFrame
        Returns a DataFrame with all detections and ground-truths labeled as:
        - 'TP' (true positive)
        - 'FP' (false positive)
        - 'FN' (false negative)

    The returned DataFrame has the same columns as `detections_df` plus an
    'evaluation' column. FN rows will have NaN for detection-only fields (e.g. 'score').
    """

    @staticmethod
    def evaluate(
        detections_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame,
        iou_threshold: float = 0.5
    ) -> pd.DataFrame:
        """Evaluate detected bounding boxes by comparing it to ground truth.

        Args:
            detections_df (pd.DataFrame): Data frame contianing detected bounding boxes
            ground_truth_df (pd.DataFrame): Data frame containing ground truth bounding boxes
            iou_threshold (float, optional): Overlap threshold for matching. Defaults to 0.5.

        Returns:
            pd.DataFrame: Detection evaluation data frame
        """
        # Copy inputs
        detections = detections_df.copy().reset_index(drop=True)
        ground_truths = ground_truth_df.copy().reset_index(drop=True)

        # Prepare evaluation column
        detections['evaluation'] = None

        # Track matched ground truths
        matched_gt = set()

        # Sort detections by score descending if available
        if 'score' in detections.columns:
            det_order = detections.sort_values('score', ascending=False).index
        else:
            det_order = detections.index

        # Greedy matching
        for det_idx in det_order:
            det = detections.loc[det_idx]
            # filter GT by image and category
            mask = (
                ground_truths['image_id'] == det['image_id']
            ) & (
                ground_truths['category_id'] == det['category_id']
            )
            cands = ground_truths[mask]

            best_iou = 0.0
            best_gt_idx = None
            for gt_idx, gt in cands.iterrows():
                if gt_idx in matched_gt:
                    continue
                iou_val = compute_iou_xyxy(det, gt)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx is not None:
                detections.at[det_idx, 'evaluation'] = 'TP'
                matched_gt.add(best_gt_idx)
            else:
                detections.at[det_idx, 'evaluation'] = 'FP'

        # Add FN rows for unmatched GT
        fn_rows = []
        for gt_idx, gt in ground_truths.iterrows():
            if gt_idx not in matched_gt:
                # fill detection columns with NA or GT values for coords if desired
                fn_entry = {col: pd.NA for col in detections_df.columns}
                fn_entry.update({
                    'image_id': gt['image_id'],
                    'category_id': gt['category_id'],
                    'xmin': gt['xmin'],
                    'ymin': gt['ymin'],
                    'xmax': gt['xmax'],
                    'ymax': gt['ymax'],
                })
                fn_entry['evaluation'] = 'FN'
                fn_rows.append(fn_entry)

        result_df = pd.concat([detections, pd.DataFrame(fn_rows)], ignore_index=True, sort=False)
        return result_df
