import torch
import cv2
import torch.nn.functional as functional
import numpy as np
import torchvision.transforms as transforms

from gym.spaces import MultiDiscrete
from abc import ABC
from typing import Optional, Sequence, Dict, Union, Tuple, Any, cast, List
from collections import OrderedDict

from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.sensor import SensorSuite
from allenact.embodiedai.sensors.vision_sensors import VisionSensor
from allenact_plugins.ithor_plugin.ithor_sensors \
    import RelativePositionChangeTHORSensor

from rearrange.sensors import RGBRearrangeSensor
from rearrange.sensors import UnshuffledRGBRearrangeSensor
from rearrange.sensors import DepthRearrangeSensor
from rearrange.tensorf_sensor import NeRFDepthSensor

from rearrange.tasks import RearrangeTaskSampler
from rearrange.tasks import UnshuffleTask
from rearrange.tasks import WalkthroughTask

from rearrange.environment import RearrangeTHOREnvironment
from baseline_configs.rearrange_base import RearrangeBaseExperimentConfig

from allenact.base_abstractions.misc import EnvType
from allenact.base_abstractions.task import SubTaskType
from allenact.utils.misc_utils import prepare_locals_for_super
from ai2thor.platform import CloudRendering

import mass.thor.alfworld_constants
from mass.thor.alfworld_mrcnn import load_pretrained_model
from mass.thor.detectron_utils import load_maskrcnn
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN


# objects that respond the pickup action
PICKABLE_TO_COLOR = OrderedDict([
    ('Candle', (233, 102, 178)),
    ('SoapBottle', (168, 222, 137)),
    ('ToiletPaper', (162, 204, 152)),
    ('SoapBar', (43, 97, 155)),
    ('SprayBottle', (89, 126, 121)),
    ('TissueBox', (98, 43, 249)),
    ('DishSponge', (166, 58, 136)),
    ('PaperTowelRoll', (144, 173, 28)),
    ('Book', (43, 31, 148)),
    ('CreditCard', (56, 235, 12)),
    ('Dumbbell', (45, 57, 144)),
    ('Pen', (239, 130, 152)),
    ('Pencil', (177, 226, 23)),
    ('CellPhone', (227, 98, 136)),
    ('Laptop', (20, 107, 222)),
    ('CD', (65, 112, 172)),
    ('AlarmClock', (184, 20, 170)),
    ('Statue', (243, 75, 41)),
    ('Mug', (8, 94, 186)),
    ('Bowl', (209, 182, 193)),
    ('TableTopDecor', (126, 204, 158)),
    ('Box', (60, 252, 230)),
    ('RemoteControl', (187, 19, 208)),
    ('Vase', (83, 152, 69)),
    ('Watch', (242, 6, 88)),
    ('Newspaper', (19, 196, 2)),
    ('Plate', (188, 154, 128)),
    ('WateringCan', (147, 67, 249)),
    ('Fork', (54, 200, 25)),
    ('PepperShaker', (5, 204, 214)),
    ('Spoon', (235, 57, 90)),
    ('ButterKnife', (135, 147, 55)),
    ('Pot', (132, 237, 87)),
    ('SaltShaker', (36, 222, 26)),
    ('Cup', (35, 71, 130)),
    ('Spatula', (30, 98, 242)),
    ('WineBottle', (53, 130, 252)),
    ('Knife', (211, 157, 122)),
    ('Pan', (246, 212, 161)),
    ('Ladle', (174, 98, 216)),
    ('Egg', (240, 75, 163)),
    ('Kettle', (7, 83, 48)),
    ('Bottle', (64, 80, 115))])


# objects that respond to the open action
OPENABLE_TO_COLOR = OrderedDict([
    ('Drawer', (155, 30, 210)),
    ('Toilet', (21, 27, 163)),
    ('ShowerCurtain', (60, 12, 39)),
    ('ShowerDoor', (36, 253, 61)),
    ('Cabinet', (210, 149, 89)),
    ('Blinds', (214, 223, 197)),
    ('LaundryHamper', (35, 109, 26)),
    ('Safe', (198, 238, 160)),
    ('Microwave', (54, 96, 202)),
    ('Fridge', (91, 156, 207))])

# mapping from classes to colors for segmentation
CLASS_TO_COLOR = OrderedDict(
    [("OccupiedSpace", (243, 246, 208))]
    + list(PICKABLE_TO_COLOR.items())
    + list(OPENABLE_TO_COLOR.items()))


# number of semantic segmentation classes we process
NUM_CLASSES = len(CLASS_TO_COLOR)


# quickly determine if a class is pickable or openable
ID_TO_PICKABLE = [key in PICKABLE_TO_COLOR
                  for key in CLASS_TO_COLOR.keys()]
ID_TO_OPENABLE = [key in OPENABLE_TO_COLOR
                  for key in CLASS_TO_COLOR.keys()]


class SemanticRearrangeSensor(VisionSensor[EnvType, SubTaskType], ABC):
    """Creates a semantic segmentation sensor for the AI2-THOR rearrange
    task family, which involves rendering a colorized class image, and
    converting that class image to class ids by looking up colors.

    Arguments:

    device: str
        specifies the device used by torch during the color lookup
        operation, which can be accelerated when set to a cuda device.
    map_height: Optional[int]
        the map_height of the image that is rendered by the thor simulator,
        used to define the observation space shape.
    map_width: Optional[int]
        the map_width of the image that is rendered by the thor simulator,
        used to define the observation space shape.
    uuid: str
        a string that defines the name associated with the sensor values
        in the observation dict returned by the thor rearrange task.
    output_shape: Optional[Tuple[int, ...]]
        an optional shape to override the previous map_height and map_width values
        to define the sensor observation space.
    output_channels: int
        an integer that defines the number of output channels in the
        observation space, which should be set to one for this sensor.

    ground_truth: bool
        a boolean that controls whether to use ground truth semantic
        segmentation or to use Mask R-CNN for segmentation.
    small_detection_threshold: float
        a threshold that controls how many predictions from Mask R-CNN
        are considered confident detections and added to the map.
    large_detection_threshold: float
        a threshold that controls how many predictions from Mask R-CNN
        are considered confident detections and added to the map.

    """

    def __init__(self, device: str = 'cuda:0', height: Optional[int] = None,
                 width: Optional[int] = None, uuid: str = "semantic",
                 which_task_env: str = None,
                 output_shape: Optional[Tuple[int, ...]] = None,
                 output_channels: int = 1, ground_truth: bool = False,
                 detection_threshold: float = 0.8, **kwargs: Any):
        """Creates a semantic segmentation sensor for the AI2-THOR rearrange
        task family, which involves rendering a colorized class image, and
        converting that class image to class ids by looking up colors.

        Arguments:

        device: str
            specifies the device used by torch during the color lookup
            operation, which can be accelerated when set to a cuda device.
        map_height: Optional[int]
            the map_height of the image that is rendered by the thor simulator,
            used to define the observation space shape.
        map_width: Optional[int]
            the map_width of the image that is rendered by the thor simulator,
            used to define the observation space shape.
        uuid: str
            a string that defines the name associated with the sensor values
            in the observation dict returned by the thor rearrange task.
        output_shape: Optional[Tuple[int, ...]]
            an optional shape to override the previous map_height and map_width values
            to define the sensor observation space.
        output_channels: int
            an integer that defines the number of output channels in the
            observation space, which should be set to one for this sensor.

        ground_truth: bool
            a boolean that controls whether to use ground truth semantic
            segmentation or to use Mask R-CNN for segmentation.
        detection_threshold: float
            a threshold that controls how many predictions from Mask R-CNN
            are considered confident detections and added to the map.

        """

        self.index = 0

        # parameters that control the device Mask R-CNN is copied to
        # and whether to use ground truth semantic segmentation of Mask R-CNN
        self.device = device
        self.ground_truth = ground_truth
        self.which_task_env = which_task_env
        self.detection_threshold = detection_threshold

        if not self.ground_truth:  # only load Mask R-CNN if used

            # parameters controlling the order of classes in segmentation
            self._transform = transforms.Compose([transforms.ToTensor()])
            self._class_to_idx = list(CLASS_TO_COLOR.keys())

            # load a pretrained checkpoint for Mask R-CNN trained for AI2-THOR
            # and copy the weights to a specified device
            self.sem_seg_model = load_maskrcnn(
                CLASS_TO_COLOR, SegmentationConfig.SCREEN_SIZE)

            self.sem_seg_model.model.to(self.device)

        # mapping from object ids to colors for semantic segmentation
        self._class_to_color = torch.ShortTensor(
            list(CLASS_TO_COLOR.values())).to(device=device)

        # call the super constructor for the base VisionSensor class
        super().__init__(**prepare_locals_for_super(locals()))

    def _make_observation_space(self, output_shape: Optional[Tuple[int, ...]],
                                output_channels: Optional[int],
                                unnormalized_infimum: float,
                                unnormalized_supremum: float) -> MultiDiscrete:
        """Creates the observation space for this sensor, which is an image
        with the given map_height, map_width, and channels, where every pixel is
        replace by an integer class id representing a semantic segmentation.

        Arguments:

        output_shape: Optional[Tuple[int, ...]]
            an optional shape to override the previous map_height and map_width values
            to define the sensor observation space.
        output_channels: int
            an integer that defines the number of output channels in the
            observation space, which should be set to one for this sensor.
        unnormalized_infimum: float
            this parameter is unused for this sensor, as the observation space
            consists of the set of natural numbers up to len(classes).
        unnormalized_supremum: float
            this parameter is unused for this sensor, as the observation space
            consists of the set of natural numbers up to len(classes).

        Returns:

        space: gym.spaces.MultiDiscrete
            a discrete ovservation space in the shape of an image with the
            specified map_height, map_width, and number of channels.

        """

        # ensure the shape does not have multiple definitions
        assert output_shape is None or output_channels is None, (
            "In VisionSensor's config, "
            "only one of output_shape and output_channels can be not None.")

        # acquire the shape of semantic segmentation class maps for images
        shape = output_shape
        if self._height is not None and output_channels is not None:
            shape = (cast(int, self._height),
                     cast(int, self._width),
                     cast(int, output_channels))

        # create a discrete observation space representing segmentation
        return MultiDiscrete(np.full(shape, self._class_to_color.shape[0]))

    def get_segmentation(self, semantic_segmentation_frame, frame):
        """Helper function to generate an additional ground truth semantic
        segmentation image observations from either the Walkthrough or
        Unshuffle environment during a rollout (this is normally hidden).

        Arguments:

        env: RearrangeTHOREnvironment
            the current active THOR environment that will be used to generate
            a semantic segmentation observation for the current view.

        Returns:

        segmentation: torch.LongTensor
            the semantic segmentation represented as an image tensor with
            shape [map_height, map_width, 1], where each pixel is a class id.

        """

        # copy the frame tensor to the gpu, which is faster than the cpu
        seg_frame = semantic_segmentation_frame
        seg_frame = torch.ShortTensor(seg_frame.copy()).to(
            device=self.device).view(self._height, self._width, 1, 3)

        # generate a mask over object classes for each segmentation color
        c = self._class_to_color[1:].view(1, 1, NUM_CLASSES - 1, 3)
        class_mask = ((seg_frame - c).abs().sum(dim=3) == 0).to(torch.float32)

        # a segmentation where all unknown classes are set to obstacles
        gt_seg = functional.pad(class_mask, (
            1, 0), value=0.1).argmax(dim=2, keepdim=True)

        # if Mask R-CNN is not required return only the ground truth
        if self.ground_truth:
            return gt_seg.cpu().numpy()  # the ground truth is used directly

        # perform a forward pass on the Mask R-CNN models with the given
        # image and gather the results in two output dictionaries
        outputs = self.sem_seg_model(frame[:, :, ::-1])

        # create a buffer to store object segmentation predictions
        semantic_seg = torch.zeros(self._height, self._width,
                                   NUM_CLASSES,
                                   device=self.device,
                                   dtype=torch.float32)

        # iterate through each detection made by the model
        for i in range(len(outputs['instances'])):

            object_score = outputs['instances'].scores[i]

            if object_score < self.detection_threshold:
                continue  # skip if the model is not confident enough

            object_class = outputs['instances'].pred_classes[i]

            # otherwise, add the object mask to the segmentation buffer
            semantic_seg[:, :, object_class] += \
                outputs['instances'].pred_masks[i].to(torch.float32)

        # take argmax over the channels to identify one object per pixel
        semantic_seg = semantic_seg.argmax(dim=2, keepdim=True)

        # return either the ground truth segmentation of the prediction
        return semantic_seg.cpu().numpy()

    def get_observation(self, env: RearrangeTHOREnvironment,
                        task: Union[WalkthroughTask, UnshuffleTask],
                        *args: Any, **kwargs: Any) -> np.ndarray:
        """Utility function to obtain semantic segmentation observations from
        an AI2-THOR Rearrangement Task, which involves rendering a colorized
        class image and looking up class ids by color values.

        Arguments:

        env: RearrangeTHOREnvironment
            the base thor environment, which is not used in this function,
            see the task argument instead, for two-phase rearrangement.
        task: Union[WalkthroughTask, UnshuffleTask]
            tha base thor rearrangement task, which includes both a
            walkthrough and rearrangement thor environment.

        Returns:

        segmentation: torch.LongTensor
            the semantic segmentation represented as an image tensor with
            shape [map_height, map_width, 1], where each pixel is a class id.

        """

        env = (task.walkthrough_env if isinstance(task, WalkthroughTask)
               else (task.walkthrough_env if self.which_task_env
                     == "walkthrough" else task.unshuffle_env))

        # check which phase it is by which variant of the task we have
        return self.get_segmentation(
            env.last_event.semantic_segmentation_frame, env.last_event.frame)


class SegmentationConfig(RearrangeBaseExperimentConfig, ABC):
    """Create a training session using the AI2-THOR Rearrangement task,
    including additional map_depth and semantic segmentation observations
    and expose a task sampling function.

    """

    # interval between successive WalkthroughTasks every next_task call
    TRAIN_UNSHUFFLE_RUNS_PER_WALKTHROUGH: int = 1

    # these sensors define the observation space of the agent
    # the relative pose sensor returns the pose of the agent in the world
    SENSORS = [
        RGBRearrangeSensor(
            height=RearrangeBaseExperimentConfig.SCREEN_SIZE,
            width=RearrangeBaseExperimentConfig.SCREEN_SIZE,
            uuid=RearrangeBaseExperimentConfig.EGOCENTRIC_RGB_UUID,
            use_resnet_normalization=False
        ),
        DepthRearrangeSensor(
            height=RearrangeBaseExperimentConfig.SCREEN_SIZE,
            width=RearrangeBaseExperimentConfig.SCREEN_SIZE
        ),
        RelativePositionChangeTHORSensor()
    ]

    @classmethod
    def make_sampler_fn(cls, stage: str, force_cache_reset: bool,
                        allowed_scenes: Optional[Sequence[str]], seed: int,
                        epochs: Union[str, float, int],
                        scene_to_allowed_rearrange_inds:
                        Optional[Dict[str, Sequence[int]]] = None,
                        x_display: Optional[str] = None,
                        sensors: Optional[Sequence[Sensor]] = None,
                        only_one_unshuffle_per_walkthrough: bool = False,
                        thor_controller_kwargs: Optional[Dict] = None,
                        device: str = 'cuda:0', ground_truth: bool = False,
                        detection_threshold: float = 0.8,
                        **kwargs) -> RearrangeTaskSampler:
        """Helper function that creates an object for sampling AI2-THOR 
        Rearrange tasks in walkthrough and unshuffle phases, where additional 
        semantic segmentation and map_depth observations are provided.

        Arguments:

        device: str
            specifies the device used by torch during the color lookup
            operation, which can be accelerated when set to a cuda device.

        Returns:

        sampler: RearrangeTaskSampler
            an instance of RearrangeTaskSampler that implements next_task()
            for generating walkthrough and unshuffle tasks successively.

        """

        # carrying this check over from the example, not sure if required
        assert not cls.RANDOMIZE_START_ROTATION_DURING_TRAINING
        if "mp_ctx" in kwargs:
            del kwargs["mp_ctx"]

        # add a semantic segmentation observation sensor to the list
        sensors = (SemanticRearrangeSensor(
            height=RearrangeBaseExperimentConfig.SCREEN_SIZE,
            width=RearrangeBaseExperimentConfig.SCREEN_SIZE,
            device=device, ground_truth=ground_truth,
            detection_threshold=detection_threshold),
                   *(cls.SENSORS if sensors is None else sensors))

        # allow default controller arguments to be overridden
        controller_kwargs = dict(**cls.THOR_CONTROLLER_KWARGS)
        if thor_controller_kwargs is not None:
            controller_kwargs.update(thor_controller_kwargs)

        # create a task sampler and carry over settings from the example
        # and ensure the environment will generate a semantic segmentation
        return RearrangeTaskSampler.from_fixed_dataset(
            run_walkthrough_phase=True,
            run_unshuffle_phase=True,
            stage=stage,
            allowed_scenes=allowed_scenes,
            scene_to_allowed_rearrange_inds=scene_to_allowed_rearrange_inds,
            rearrange_env_kwargs=dict(
                force_cache_reset=force_cache_reset,
                **cls.REARRANGE_ENV_KWARGS,
                controller_kwargs={
                    "platform": CloudRendering,
                    "renderDepthImage": any(
                        isinstance(sensor, DepthRearrangeSensor)
                        for sensor in sensors
                    ),
                    "renderSemanticSegmentation": any(
                        isinstance(sensor, SemanticRearrangeSensor)
                        for sensor in sensors
                    ),
                    **controller_kwargs,
                },
            ),
            seed=seed,
            sensors=SensorSuite(sensors),
            max_steps=cls.MAX_STEPS,
            discrete_actions=cls.actions(),
            require_done_action=cls.REQUIRE_DONE_ACTION,
            force_axis_aligned_start=cls.FORCE_AXIS_ALIGNED_START,
            unshuffle_runs_per_walkthrough=
            cls.TRAIN_UNSHUFFLE_RUNS_PER_WALKTHROUGH
            if (not only_one_unshuffle_per_walkthrough) and stage == "train"
            else None,
            epochs=epochs, **kwargs)


class OnePhaseSegmentationConfig(RearrangeBaseExperimentConfig, ABC):
    """Create a training session using the AI2-THOR Rearrangement task,
    including additional map_depth and semantic segmentation observations
    and expose a task sampling function.

    """

    SENSORS = [
        RGBRearrangeSensor(
            height=RearrangeBaseExperimentConfig.SCREEN_SIZE,
            width=RearrangeBaseExperimentConfig.SCREEN_SIZE,
            use_resnet_normalization=False,
            uuid=RearrangeBaseExperimentConfig.EGOCENTRIC_RGB_UUID,
        ),
        UnshuffledRGBRearrangeSensor(
            height=RearrangeBaseExperimentConfig.SCREEN_SIZE,
            width=RearrangeBaseExperimentConfig.SCREEN_SIZE,
            use_resnet_normalization=False,
            uuid=RearrangeBaseExperimentConfig.UNSHUFFLED_RGB_UUID,
        ),
        DepthRearrangeSensor(
            height=RearrangeBaseExperimentConfig.SCREEN_SIZE,
            width=RearrangeBaseExperimentConfig.SCREEN_SIZE
        ),
    ]

    @classmethod
    def make_sampler_fn(cls, stage: str, force_cache_reset: bool,
                        allowed_scenes: Optional[Sequence[str]],
                        seed: int, epochs: int,
                        scene_to_allowed_rearrange_inds:
                        Optional[Dict[str, Sequence[int]]] = None,
                        x_display: Optional[str] = None,
                        sensors: Optional[Sequence[Sensor]] = None,
                        thor_controller_kwargs: Optional[Dict] = None,
                        device: str = 'cuda:0', ground_truth: bool = False,
                        detection_threshold: float = 0.8,
                        **kwargs) -> RearrangeTaskSampler:
        """Helper function that creates an object for sampling AI2-THOR
        Rearrange tasks in walkthrough and unshuffle phases, where additional
        semantic segmentation and map_depth observations are provided.

        Arguments:

        device: str
            specifies the device used by torch during the color lookup
            operation, which can be accelerated when set to a cuda device.

        Returns:

        sampler: RearrangeTaskSampler
            an instance of RearrangeTaskSampler that implements next_task()
            for generating walkthrough and unshuffle tasks successively.

        """

        assert not cls.RANDOMIZE_START_ROTATION_DURING_TRAINING
        if "mp_ctx" in kwargs:
            del kwargs["mp_ctx"]

        # add a semantic segmentation observation sensor to the list
        sensors = (
            SemanticRearrangeSensor(
                height=RearrangeBaseExperimentConfig.SCREEN_SIZE,
                width=RearrangeBaseExperimentConfig.SCREEN_SIZE,
                device=device, ground_truth=ground_truth,
                which_task_env="walkthrough", uuid="semantic",
                detection_threshold=detection_threshold,
            ),
            SemanticRearrangeSensor(
                height=RearrangeBaseExperimentConfig.SCREEN_SIZE,
                width=RearrangeBaseExperimentConfig.SCREEN_SIZE,
                device=device, ground_truth=ground_truth,
                which_task_env="unshuffle", uuid="unshuffled_semantic",
                detection_threshold=detection_threshold,
            ),
            *(cls.SENSORS if sensors is None else sensors)
        )

        # allow default controller arguments to be overridden
        controller_kwargs = dict(**cls.THOR_CONTROLLER_KWARGS)
        if thor_controller_kwargs is not None:
            controller_kwargs.update(thor_controller_kwargs)

        return RearrangeTaskSampler.from_fixed_dataset(
            run_walkthrough_phase=False,
            run_unshuffle_phase=True,
            stage=stage,
            allowed_scenes=allowed_scenes,
            scene_to_allowed_rearrange_inds=scene_to_allowed_rearrange_inds,
            rearrange_env_kwargs=dict(
                force_cache_reset=force_cache_reset,
                **cls.REARRANGE_ENV_KWARGS,
                controller_kwargs={
                    "platform": CloudRendering,
                    "renderDepthImage": any(
                        isinstance(sensor, DepthRearrangeSensor)
                        for sensor in sensors
                    ),
                    "renderSemanticSegmentation": any(
                        isinstance(sensor, SemanticRearrangeSensor)
                        for sensor in sensors
                    ),
                    **controller_kwargs,
                },
            ),
            seed=seed,
            sensors=SensorSuite(sensors),
            max_steps=cls.MAX_STEPS,
            discrete_actions=cls.actions(),
            require_done_action=cls.REQUIRE_DONE_ACTION,
            force_axis_aligned_start=cls.FORCE_AXIS_ALIGNED_START,
            epochs=epochs,
            **kwargs,
        )
