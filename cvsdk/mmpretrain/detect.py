import os
import glob
import time
import numpy as np
import pandas as pd
from mmpretrain.apis import inference_model, init_model
from structlog import get_logger
from tqdm import tqdm

logger = get_logger()


def detect(
        config_file: str,
        epoch: int,
        results_file: str,
        dataset_dir: str,
        image_files: str,
        batch_size: int = 8,
        score_threshold: float = 0.5,
        device: str = "cuda:0",
        work_dir: str = "./work_dirs"
) -> None:
    """Perform image classification/prediction on images.

    Args:
        config_file (str): Path to an MMPretrain configuration file
        epoch (int): Epoch number to perform predictions for (-1 for all epochs)
        results_file (str): Path where the results should be stored
        dataset_dir (str): Path where the dataset is stored
        image_files (str): Glob expression for image paths in dataset directory
        batch_size (int, optional): Batch size for inference. Defaults to 8.
        score_threshold (float, optional): Score threshold for predictions. Defaults to 0.5.
        device (str, optional): Device to use for inference. Defaults to "cuda:0".
        work_dir (str, optional): Path where checkpoints are stored. Defaults to "./work_dirs".
    """
    config_file_path = f"{work_dir}/{config_file}"
    epochs = list(range(1, len(glob.glob(f"{work_dir}/epoch_*.pth")) + 1)) if epoch == -1 else [epoch]

    logger.info(
        "Configuration",
        config_file_path=config_file_path,
        epoch=epoch,
        device=device,
        batch_size=batch_size
    )

    predictions = []
    for epoch in epochs:
        logger.info(f"Predictions for epoch {epoch} ...")
        weight_file = f"{work_dir}/epoch_{epoch}.pth"
        # Initialize the model
        logger.info("Loading model ...")
        model = init_model(
            config_file_path,
            weight_file,
            device=device
        )
        logger.info("Model loaded")

        image_file_path = os.path.join(dataset_dir, image_files)
        logger.info("Image files", image_file_path=image_file_path)
        filenames = glob.glob(image_file_path)
        n_files = len(filenames)
        logger.info(f"Found {n_files} images")
        if n_files == 0:
            return
        n_batches = (n_files // batch_size) + 1

        durations = []
        pbar = tqdm(range((n_files // batch_size) + 1))
        for b in pbar:
            pbar.set_description(f"Processing batch {b}/{n_batches} ...")
            batch_files = filenames[b * batch_size:(b + 1) * batch_size]
            if len(batch_files) < 1:
                continue
            # Send images through model
            start = time.time()
            for img_file in batch_files:
                result = inference_model(model, img_file)
                end = time.time()
                durations.append(end - start)
                
                # Extract prediction results
                filename = img_file.replace(os.sep, '/').replace(dataset_dir, "")
                pred_label = result['pred_label']
                pred_score = result['pred_score']
                pred_class = result.get('pred_class', 'unknown')
                
                if pred_score >= score_threshold:
                    predictions.append({
                        "epoch": epoch,
                        "filename": filename,
                        "pred_label": int(pred_label),
                        "pred_class": str(pred_class),
                        "pred_score": float(pred_score)
                    })
        del model  # free memory

    # Create the CSV file that contains the predictions
    csv_filename = f"{work_dir}/{results_file}"
    os.makedirs(os.path.split(csv_filename)[0], exist_ok=True)
    pd.DataFrame(predictions).to_csv(csv_filename, index=False)
    logger.info("Saved CSV file at", csv_filename=csv_filename)

    # Print some statistics about the prediction process
    durations_arr = np.array(durations)
    logger.info(f"Inference took {durations_arr.mean():.4f}s per image on average, std={durations_arr.std():.4f}")
