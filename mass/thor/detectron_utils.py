from detectron2 import model_zoo
from detectron2.engine.defaults import DefaultPredictor
import os


def load_maskrcnn(CLASS_TO_COLOR, SCREEN_SIZE):

    class_names = list(CLASS_TO_COLOR.keys())

    cfg = model_zoo.get_config("COCO-InstanceSegmentation/"
                               "mask_rcnn_R_50_FPN_3x.yaml")

    cfg.MODEL.MASK_ON = True
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)

    cfg.DATASETS.TEST = ('ai2thor-val',)
    cfg.DATASETS.TRAIN = ('ai2thor-train',)

    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"

    cfg.INPUT.MIN_SIZE_TRAIN = (SCREEN_SIZE,)
    cfg.INPUT.MAX_SIZE_TRAIN = SCREEN_SIZE

    cfg.INPUT.MIN_SIZE_TEST = SCREEN_SIZE
    cfg.INPUT.MAX_SIZE_TEST = SCREEN_SIZE

    cfg.TEST.AUG.MIN_SIZES = (SCREEN_SIZE,)
    cfg.TEST.AUG.MAX_SIZE = SCREEN_SIZE

    cfg.MODEL.WEIGHTS = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), "model_final.pth")

    return DefaultPredictor(cfg)

