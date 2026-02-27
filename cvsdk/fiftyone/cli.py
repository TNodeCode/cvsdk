import os
import click
import numpy as np
import fiftyone as fo
import fiftyone.brain as fob
import pandas as pd


@click.group()
def fiftyone() -> None:
    """CLI for training and managing a YOLO model on a custom dataset."""


@fiftyone.command()
@click.option('--root-dir', type=click.Path(exists=True), required=True, help='Root directory containing images and annotations')
@click.option('--annotations', type=str, required=True, help='COCO JSON annotation file')
@click.option('--images', type=str, required=True, help='Directory where images are stored')
def show_coco(root_dir: str, annotations: str, images: str) -> None:
    """Starts a FiftyOne application to inspect a COCO object detection dataset."""
    # Load the COCO dataset into FiftyOne
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=os.path.join(root_dir, images),
        labels_path=os.path.join(root_dir, annotations),
    )

    # Launch FiftyOne app
    session = fo.launch_app(dataset)

    print("FiftyOne app is running. Press CTRL-C to stop.")

    try:
        # Keep the script running until CTRL-C is pressed
        session.wait()
    except KeyboardInterrupt:
        print("\nSession terminated by user.")


@fiftyone.command()
@click.option('--root-dir', type=click.Path(exists=True), required=True, help='Root directory containing images and annotations')
@click.option('--images', type=str, required=True, help='Directory where images are stored')
@click.option('--labels', type=str, required=True, help='YOLO labels directory (relative to root-dir)')
@click.option('--classes', type=str, required=True, help='Path to classes.txt file (relative to root-dir)')
def show_yolo(root_dir: str, images: str, labels: str, classes: str) -> None:
    """Starts a FiftyOne application to inspect a YOLO object detection dataset."""
    # Load the YOLO dataset into FiftyOne
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.YOLOv4Dataset,
        data_path=os.path.join(root_dir, images),
        labels_path=os.path.join(root_dir, labels),
        classes_path=os.path.join(root_dir, classes),
    )

    # Launch FiftyOne app
    session = fo.launch_app(dataset)

    print("FiftyOne app is running. Press CTRL-C to stop.")

    try:
        # Keep the script running until CTRL-C is pressed
        session.wait()
    except KeyboardInterrupt:
        print("\nSession terminated by user.")


@fiftyone.command()
@click.option('--root-dir', type=click.Path(exists=True), required=True,
              help='Root directory containing images and annotations')
@click.option('--annotations', type=str, required=True,
              help='Path to the input COCO JSON annotation file')
@click.option('--images', type=str, required=True,
              help='Directory where images are stored relative to root-dir')
@click.option('--output', type=str, required=True,
              help='Path to save the new COCO JSON annotation file')
def annotate(root_dir: str, annotations: str, images: str, output: str) -> None:
    """Loads a COCO dataset, sends it to CVAT for annotation, merges the new annotations, and exports them as a new COCO JSON file."""
    #os.environ["FIFTYONE_CVAT_USERNAME"] = "john.doe"
    #os.environ["FIFTYONE_CVAT_PASSWORD"] = "Test_1234"

    # 1. Load the COCO dataset into FiftyOne
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=os.path.join(root_dir, images),
        labels_path=os.path.join(root_dir, annotations),
    )

    print(f"Loaded dataset '{dataset.name}' with {len(dataset)} samples.")

    # (Optional) Launch the FiftyOne App to inspect your dataset
    session = fo.launch_app(dataset)
    print("FiftyOne app is running. You can inspect your dataset now.")

    # 2. Send the dataset to CVAT for annotation.
    # Choose a unique annotation run key and specify the label field to annotate.
    anno_key = "cvat_anno_run"
    label_types = ['classification', 'classifications', 'detection', 'detections', 'instance', 'instances', 'polyline', 'polylines', 'polygon', 'polygons', 'keypoint', 'keypoints', 'segmentation', 'scalar']
    label_type = label_types[3]
    label_fields = dataset._get_label_fields()
    label_field = label_fields[0]
    print("LABEL TYPE: ", label_type)
    print("LABEL FIELDS: ", label_fields)
    print("LABEL FIELD: ", label_field)


    # This call uploads the samples (and existing labels, if any) to CVAT.
    # The 'launch_editor=True' flag will automatically open the CVAT editor.
    dataset.annotate(
        anno_key,
        label_field=label_type,  # change this if your labels are stored under a different field
        label_type=label_field,
        classes=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        backend="cvat",
        url="http://localhost:8080", # TODO use environment variables
        username="john.doe", # TODO use environment variables
        password="Test_1234", # TODO use environment variables
        launch_editor=True,
    )
    print("Dataset sent to CVAT. Please annotate the data in CVAT and save your changes.")

    # Wait for the user to finish annotating in CVAT
    input("Press Enter after you have completed annotation in CVAT and saved your work...")

    # 3. Merge the new annotations back into your FiftyOne dataset.
    dataset.load_annotations(anno_key)
    print("Annotations have been loaded back into the dataset.")
    
    # (Optional) Update the FiftyOne App view to inspect the annotated data
    session.dataset = dataset

    # 4. Export the updated dataset (annotations only) in COCO format.
    # Here, we export the labels to a new JSON file.
    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    dataset.export(
        export_dir=output_dir,
        dataset_type=fo.types.COCODetectionDataset,
        label_field=label_field,
        labels_path=output,
        include_media=False  # Set to True if you want to copy the image files as well
    )

    # Delete tasks from CVAT
    results = dataset.load_annotation_results(anno_key)
    results.cleanup()

    # Delete run record (not the labels) from FiftyOne
    dataset.delete_annotation_run(anno_key)
    print(f"New COCO annotations saved to: {output}")


@fiftyone.command()
@click.option('--root-dir', type=click.Path(exists=True), required=True, help='Root directory containing images and annotations')
@click.option('--annotations', type=str, required=True, help='COCO JSON annotation file')
@click.option('--images', type=str, required=True, help='Directory where images are stored')
@click.option('--detections-file', type=str, required=True, help='CSV file containing model detections')
def detections(root_dir: str, annotations: str, images: str, detections_file: str) -> None:
    """Starts a FiftyOne application to inspect a computer vision object detection dataset."""
    # Load the COCO dataset into FiftyOne
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=os.path.join(root_dir, images),
        labels_path=os.path.join(root_dir, annotations),
    )

    # Parse the detections CSV
    detections_path = os.path.join(root_dir, detections_file)
    detections_df = pd.read_csv(detections_path)

    # Group detections by image filename
    grouped_detections = detections_df.groupby('filename')

    # Iterate over samples and add detections
    for sample in dataset:
        filename = os.path.basename(sample.filepath)
        if filename in grouped_detections.groups:
            model_detections = []
            for _, row in grouped_detections.get_group(filename).iterrows():
                # Assuming the CSV contains 'label', 'xmin', 'ymin', 'xmax', 'ymax', and 'confidence' columns
                label = row['label']
                confidence = row['confidence']
                # Convert absolute coordinates to relative [0, 1] bounding box
                bounding_box = [
                    row['xmin'] / sample.metadata.width,
                    row['ymin'] / sample.metadata.height,
                    (row['xmax'] - row['xmin']) / sample.metadata.width,
                    (row['ymax'] - row['ymin']) / sample.metadata.height,
                ]
                model_detections.append(
                    fo.Detection(
                        label=label,
                        bounding_box=bounding_box,
                        confidence=confidence,
                    )
                )
            # Add detections to the sample
            sample['detections'] = fo.Detections(detections=model_detections)
            sample.save()

    # Launch FiftyOne app
    session = fo.launch_app(dataset)

    print("FiftyOne app is running. Press CTRL-C to stop.")
    
    try:
        # Keep the script running until CTRL-C is pressed
        session.wait()
    except KeyboardInterrupt:
        print("\nSession terminated by user.")


@fiftyone.command()
@click.option('--root-dir', type=click.Path(exists=True), required=True, help='Root directory containing images and annotations')
@click.option('--annotations', type=str, required=True, help='COCO JSON annotation file')
@click.option('--images', type=str, required=True, help='Directory where images are stored')
def embeddings(root_dir: str, annotations: str, images: str) -> None:
    """Visualize embeddings.

    Args:
        root_dir (str): Dataset root dir
        annotations (str): Path to annotations relative to root_dir
        images (str): Path to images relative to root_dir
    """
    # Step 1: Load the COCO dataset into FiftyOne
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=os.path.join(root_dir, images),
        labels_path=os.path.join(root_dir, annotations),
    )

    # Step 2: Load embeddings from the .npy file
    # TODO load from npy file
    embeddings = np.random.normal(size=(len(dataset), 128))

    # Ensure the number of embeddings matches the number of samples
    assert len(embeddings) == len(dataset), "Mismatch between embeddings and samples"

    # Step 3: Associate embeddings with samples
    for sample, embedding in zip(dataset, embeddings):
        sample["embedding"] = embedding
        sample.save()

    # Step 4: Visualize embeddings
    _ = fob.compute_visualization(
        dataset,
        embeddings=embeddings,
        method="umap",  # or "tsne", "pca" TODO add pacmap and create CLI parameter for this
        num_dims=2,
        brain_key="embedding_viz",
    )

    # Launch the FiftyOne app to explore the embeddings
    session = fo.launch_app(dataset)

    print("FiftyOne app is running. Press CTRL-C to stop.")

    try:
        # Keep the script running until CTRL-C is pressed
        session.wait()
    except KeyboardInterrupt:
        print("\nSession terminated by user.")
