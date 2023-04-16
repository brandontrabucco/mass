import glob
import json
import os
import tqdm
import random
import logging

from multiprocessing import Pool
from collections import OrderedDict
from mass.thor.segmentation_config import CLASS_TO_COLOR

from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog

from detectron2 import model_zoo
from detectron2.config import get_cfg

from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.evaluation import COCOEvaluator

from detectron2.engine import DefaultTrainer, default_argument_parser
from detectron2.engine import default_setup, hooks, launch


def process(file):

    with open(file, "r") as file:
        annotation = json.load(file)

    annotation["file_name"] = os.path.join(
        args.dataset, annotation["file_name"])

    annotation["sem_seg_file_name"] = os.path.join(
        args.dataset, annotation["sem_seg_file_name"])

    annotation["pan_seg_file_name"] = os.path.join(
        args.dataset, annotation["pan_seg_file_name"])

    return annotation


if __name__ == "__main__":

    parser = default_argument_parser()

    parser.add_argument("--dataset", type=str,
                        default="/home/btrabucc/mask-rcnn-dataset")
    parser.add_argument("--val-ratio",
                        type=float, default=0.05)

    args = parser.parse_args()

    filenames = list(glob.glob(
        os.path.join(args.dataset, "annotations/*.json")))

    random.shuffle(filenames)

    annotations = []

    with Pool() as pool:
        for x in tqdm.tqdm(pool.imap(
                process, filenames), total=len(filenames)):

            annotations.append(x)

    with open(os.path.join(
            args.dataset, "training.json"), "w") as f:

        json.dump(annotations[:-int(
            len(annotations) * args.val_ratio)], f)

    with open(os.path.join(
            args.dataset, "validation.json"), "w") as f:

        json.dump(annotations[-int(
            len(annotations) * args.val_ratio):], f)
