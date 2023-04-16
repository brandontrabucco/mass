import glob
import json
import math
import os
import random
import logging

from collections import OrderedDict
from mass.thor.segmentation_config import CLASS_TO_COLOR, SegmentationConfig

from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog

from detectron2 import model_zoo
from detectron2.config import get_cfg

from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.evaluation import COCOEvaluator

from detectron2.engine import DefaultTrainer, default_argument_parser
from detectron2.engine import default_setup, hooks, launch


class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):

        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        return COCOEvaluator(dataset_name,
                             output_dir=output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        logger.info("Running inference with test-time augmentation ...")

        model = GeneralizedRCNNWithTTA(cfg, model)

        evaluators = [cls.build_evaluator(
            cfg, name, output_folder=os.path.join(
                cfg.OUTPUT_DIR, "inference_TTA")) for name in cfg.DATASETS.TEST]

        return OrderedDict({k + "_TTA": v for k, v in
                            cls.test(cfg, model, evaluators).items()})


def main(args):

    class_names = list(CLASS_TO_COLOR.keys())

    with open(os.path.join(args.dataset, "training.json"), "r") as f:
        training_set = json.load(f)

    with open(os.path.join(args.dataset, "validation.json"), "r") as f:
        validation_set = json.load(f)

    DatasetCatalog.register("ai2thor-train", lambda: training_set)
    DatasetCatalog.register("ai2thor-val", lambda: validation_set)

    MetadataCatalog.get("ai2thor-train").set(thing_classes=class_names)
    MetadataCatalog.get("ai2thor-val").set(thing_classes=class_names)

    training_metadata = MetadataCatalog.get("ai2thor-train")
    training_metadata.set(thing_classes=class_names)

    validation_metadata = MetadataCatalog.get("ai2thor-val")
    validation_metadata.set(thing_classes=class_names)

    cfg = model_zoo.get_config("COCO-InstanceSegmentation/"
                               "mask_rcnn_R_50_FPN_3x.yaml")

    num_epoch = args.epochs
    num_batch = math.ceil(len(training_set) /
                          cfg.SOLVER.IMS_PER_BATCH)

    cfg.MODEL.MASK_ON = True
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)

    cfg.DATASETS.TEST = ('ai2thor-val',)
    cfg.DATASETS.TRAIN = ('ai2thor-train',)

    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"

    cfg.INPUT.MIN_SIZE_TRAIN = (SegmentationConfig.SCREEN_SIZE,)
    cfg.INPUT.MAX_SIZE_TRAIN = SegmentationConfig.SCREEN_SIZE

    cfg.INPUT.MIN_SIZE_TEST = SegmentationConfig.SCREEN_SIZE
    cfg.INPUT.MAX_SIZE_TEST = SegmentationConfig.SCREEN_SIZE

    cfg.TEST.AUG.MIN_SIZES = (SegmentationConfig.SCREEN_SIZE,)
    cfg.TEST.AUG.MAX_SIZE = SegmentationConfig.SCREEN_SIZE

    cfg.SOLVER.MAX_ITER = num_epoch * num_batch

    cfg.SOLVER.STEPS = ((num_epoch - 2) * num_batch,
                        (num_epoch - 1) * num_batch)

    cfg.OUTPUT_DIR = args.logdir

    cfg.merge_from_list(args.opts)
    cfg.freeze()

    default_setup(cfg, args)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks([hooks.EvalHook(
            0, lambda: trainer.test_with_TTA(cfg, trainer.model))])

    return trainer.train()


if __name__ == "__main__":

    parser = default_argument_parser()

    parser.add_argument("--logdir", type=str,
                        default="/home/btrabucc/mask-rcnn")
    parser.add_argument("--dataset", type=str,
                        default="/home/btrabucc/ai2thor-dataset")
    parser.add_argument("--epochs",
                        type=int, default=5)

    args = parser.parse_args()

    print("Command Line Args:", args)

    launch(main, args.num_gpus,
           num_machines=args.num_machines,
           machine_rank=args.machine_rank,
           dist_url=args.dist_url, args=(args,))
