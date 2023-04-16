from mass.thor.segmentation_config import CLASS_TO_COLOR
from detectron2.data import Metadata
import detectron2.utils.visualizer as visualizer
import argparse
import glob
import json
import os
import cv2


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Data Visualization Agent")

    parser.add_argument("--logdir", type=str,
                        default="/home/btrabucco/test-data")

    args = parser.parse_args()

    os.makedirs(os.path.join(
        args.logdir, "visualizations"), exist_ok=True)

    metadata = Metadata()

    metadata.thing_classes = list(CLASS_TO_COLOR.keys())
    metadata.stuff_classes = list(CLASS_TO_COLOR.keys())

    metadata.thing_colors = list(CLASS_TO_COLOR.values())
    metadata.stuff_colors = list(CLASS_TO_COLOR.values())

    for annotation in glob.glob(
            os.path.join(args.logdir, "annotations/*.json")):

        with open(annotation, "r") as f:
            annotation_data = json.load(f)

        annotation_data["file_name"] = os.path.join(
            args.logdir, annotation_data['file_name'])

        annotation_data["sem_seg_file_name"] = os.path.join(
            args.logdir, annotation_data['sem_seg_file_name'])

        annotation_data["pan_seg_file_name"] = os.path.join(
            args.logdir, annotation_data['pan_seg_file_name'])

        annotation_data.pop("sem_seg_file_name")
        annotation_data.pop("pan_seg_file_name")

        image_id = annotation_data["image_id"]

        if os.path.exists(annotation_data["file_name"]):

            image = cv2.imread(annotation_data["file_name"])

            annotated_image = visualizer.Visualizer(
                image[..., ::-1], metadata=metadata)\
                .draw_dataset_dict(annotation_data).get_image()[..., ::-1]

            cv2.imwrite(os.path.join(
                args.logdir, "visualizations",
                f"{image_id:07d}-vis.png"), annotated_image)

