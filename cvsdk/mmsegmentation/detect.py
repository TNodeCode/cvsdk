import os
import glob
import time
import numpy as np
import pandas as pd
from mmseg.apis import inference_model, init_model
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
    """Perform semantic segmentation on images.

    Args:
        config_file (str): Path to an MMSeg configuration file
        epoch (int): Number of epochs to perform detection on
        results_file (str): Path where the results should be stored
        dataset_dir (str): Path where the dataset is stored
        image_files (str): Glob expression for image paths in dataset directory
        batch_size (int, optional): Batch size for inference. Defaults to 8.
        score_threshold (float, optional): Score threshold for segmentation. Defaults to 0.5.
        device (str, optional): Device to use for inference. Defaults to "cuda:0".
        work_dir (str, optional): Path where checkpoints are stored. Defaults to "./work_dirs".
    """
    config_file_path = f"{work_dir}/{config_file}"
    epochs = list(range(1, len(glob.glob(f"{work_dir}/iter_*.pth"))+1)) if epoch > 0 else [epoch]

    logger.info(
        "Configuration",
        config_file_path=config_file_path,
        epoch=epoch,
        device=device,
        batch_size=batch_size
    )

    segmentation_results = []
    for epoch in epochs:
        logger.info(f"Segmentation for epoch {epoch} ...")
        weight_file = f"{work_dir}/iter_{epoch}.pth"
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
        n_batches=(n_files // batch_size) + 1

        durations = []
        pbar = tqdm(range((n_files // batch_size) + 1))
        for b in pbar:
            pbar.set_description(f"Processing batch {b}/{n_batches} ...")
            if (len(filenames[b*batch_size:(b+1)*batch_size])) < 1:
                continue
            # Send images through model
            start = time.time()
            for filename in filenames[b*batch_size:(b+1)*batch_size]:
                result = inference_model(
                    model=model,
                    img=filename
                )
                # Store segmentation result
                filename_rel = filename.replace(os.sep, '/').replace(dataset_dir, "")
                segmentation_results.append({
                    "epoch": epoch,
                    "filename": filename_rel,
                    "segmentation_map": result.pred_sem_seg.data.cpu().numpy()
                })
            end = time.time()
            durations.append(end - start)
        del model # free memory

    # Create the output file that contains the segmentation results
    output_filename = f"{work_dir}/{results_file}"
    os.makedirs(os.path.split(output_filename)[0], exist_ok=True)
    
    # Save segmentation maps
    for result in segmentation_results:
        seg_output_path = output_filename.replace('.csv', f'_{result["filename"].replace("/", "_")}.npy')
        np.save(seg_output_path, result["segmentation_map"])
    
    # Create CSV summary
    summary_data = [{
        "epoch": r["epoch"],
        "filename": r["filename"]
    } for r in segmentation_results]
    pd.DataFrame(summary_data).to_csv(output_filename, index=False)
    logger.info("Saved results at", output_filename=output_filename)

    # Print some statistics about the segmentation process
    durations_arr = np.array(durations)
    logger.info(f"Inference took {durations_arr.mean()}, per batch on average, std={durations_arr.std()}")
    logger.info(f"Inference took {durations_arr.mean() / batch_size} per image on average, std={durations_arr.std() / batch_size}")
