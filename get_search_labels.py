import numpy as np
import pandas as pd
import cv2
import torch

import argparse
import os
import re
import json
import shutil

from mass.navigation_policy import NavigationPolicy

from mass.nn.applications\
    .occupancy_projection_layer import OccupancyProjectionLayer
from mass.nn.applications\
    .semantic_projection_layer import SemanticProjectionLayer

from mass.utils.experimentation import run_experiment_with_restart
from mass.utils.experimentation import TimeoutDueToUnityCrash
from mass.utils.experimentation import NumpyJSONEncoder
from mass.utils.experimentation import handle_read_only

from mass.utils.experimentation import get_scene_differences
from mass.utils.experimentation import predict_scene_differences

from skvideo.io import FFmpegWriter

from mass.thor.segmentation_config import SegmentationConfig

from mass.thor.segmentation_config import PICKABLE_TO_COLOR
from mass.thor.segmentation_config import OPENABLE_TO_COLOR
from mass.thor.segmentation_config import CLASS_TO_COLOR
from mass.thor.segmentation_config import NUM_CLASSES


# used for converting from camel case object names to snake case
pattern = re.compile(r'(?<!^)(?=[A-Z])')


# used for visualizing the semantic map by rendering voxels according to
# which semantic category each voxel contains
class_to_colors = (np.array(list(CLASS_TO_COLOR.values())) / 255.0).tolist()


def semantic_mapping_experiment(args, occupancy_projection_layer,
                                semantic_projection_layer0,
                                semantic_projection_layer1):
    """Semantic Mapping agent for solving AI2-THOR Rearrangement in PyTorch,
    by building two semantic maps, one for the goal state of the world,
    and a second for the current world state, and tracking disagreement.

    """

    # create arguments for the training and testing tasks
    with TimeoutDueToUnityCrash(300):  # wait 300 seconds for unity
        task_params = SegmentationConfig.stagewise_task_sampler_args(
            stage=args.stage, process_ind=0, total_processes=1, devices=[0])

    task_params["ground_truth"] = args.ground_truth_segmentation
    task_params["detection_threshold"] = args.detection_threshold

    # generate a sampler for training or testing evaluation
    with TimeoutDueToUnityCrash(300):  # wait 300 seconds for unity
        task_sampler = SegmentationConfig.make_sampler_fn(
            **task_params, force_cache_reset=True,
            epochs=1, only_one_unshuffle_per_walkthrough=True)

    for _ in range(args.start_task):
        with TimeoutDueToUnityCrash():  # wait 60 seconds for unity to connect
            next(task_sampler.task_spec_iterator)

    # perform evaluation using every task in the task sampler
    for task_id in range(args.start_task, args.start_task + (
            args.total_tasks * args.every_tasks), args.every_tasks):

        # sample the next task in sequence for evaluation
        with TimeoutDueToUnityCrash():  # wait 60 seconds for unity
            task = task_sampler.next_task()

        origin = task.env.get_agent_location()
        origin_kwargs = dict(origin_y=origin["z"],
                             origin_x=origin["x"], origin_z=origin["y"])

        # get initial position of the agent and set this as the origin of map
        # to ensure the map is centered and navmesh grid is aligned
        occupancy_projection_layer.reset(**origin_kwargs)
        semantic_projection_layer0.reset(**origin_kwargs)
        semantic_projection_layer1.reset(**origin_kwargs)

        # sample the next task in sequence for evaluation
        with TimeoutDueToUnityCrash():  # wait 60 seconds for unity
            task = task_sampler.next_task()

        walkthrough_semantic_search_goals = []
        unshuffle_semantic_search_goals = []

        for object_two, object_one in zip(*task.env.poses[1:]):

            correct = not object_one["broken"] and \
                task.env.are_poses_equal(object_one, object_two)

            if not correct:

                object_two_loc = np.array([object_two["position"]["x"],
                                           object_two["position"]["z"],
                                           object_two["position"]["y"]])

                object_one_loc = np.array([object_one["position"]["x"],
                                           object_one["position"]["z"],
                                           object_one["position"]["y"]])

                object_two_loc = occupancy_projection_layer\
                    .world_to_map(object_two_loc).cpu().numpy()

                object_one_loc = occupancy_projection_layer\
                    .world_to_map(object_one_loc).cpu().numpy()

                walkthrough_semantic_search_goals.append(object_two_loc)
                unshuffle_semantic_search_goals.append(object_one_loc)

        walkthrough_semantic_search_goals = \
            np.stack(walkthrough_semantic_search_goals, axis=0)

        np.save(os.path.join(args.logdir, f"walkthrough-labels-{task_id}.npy"),
                walkthrough_semantic_search_goals)

        unshuffle_semantic_search_goals = \
            np.stack(unshuffle_semantic_search_goals, axis=0)

        np.save(os.path.join(args.logdir, f"unshuffle-labels-{task_id}.npy"),
                unshuffle_semantic_search_goals)

        # the environment has terminated by this point, and presumably the
        # semantic map is sufficiently detailed for path planning
        for _ in range(args.every_tasks - 1):
            with TimeoutDueToUnityCrash():  # wait 60 seconds for unity
                next(task_sampler.task_spec_iterator)

        args.start_task += args.every_tasks  # remove finished tasks
        args.total_tasks -= 1  # from experiment parameters


def run_experiment(args):
    """Semantic Mapping agent for solving AI2-THOR Rearrangement in PyTorch,
    by building two semantic maps, one for the goal state of the world,
    and a second for the current world state, and tracking disagreement.

    """

    name = (f"{args.start_task}-"  # slice of tasks per run
            f"{args.start_task + args.total_tasks * args.every_tasks}")

    # create a logging directory to dump evaluation metrics and videos
    os.makedirs(os.path.join(args.logdir, f"tmp-{name}"), exist_ok=True)

    # write the hyper-parameters that were used to obtain
    with open(os.path.join(  # these results in the experiment logdir
            args.logdir, f"params-{name}.json"), "w") as f:
        json.dump(vars(args), f, indent=4)  # write hyperparameters to file

    # this ensures that no processes share the same AI2-THOR executable
    # which occasionally crashes and does not release lock files
    os.environ["HOME"] = os.path.join(args.logdir, f"tmp-{name}")

    # create a set of semantic maps and occupancy maps that are
    # used for path planning and localizing objects.
    occupancy_projection_layer = OccupancyProjectionLayer(
        camera_height=SegmentationConfig.SCREEN_SIZE,
        camera_width=SegmentationConfig.SCREEN_SIZE,
        vertical_fov=args.vertical_fov,
        map_height=args.map_height,
        map_width=args.map_width,
        map_depth=args.map_depth,
        grid_resolution=args.grid_resolution).train().cuda()
    semantic_projection_layer0 = SemanticProjectionLayer(
        camera_height=SegmentationConfig.SCREEN_SIZE,
        camera_width=SegmentationConfig.SCREEN_SIZE,
        vertical_fov=args.vertical_fov,
        map_height=args.map_height,
        map_width=args.map_width,
        map_depth=args.map_depth,
        feature_size=NUM_CLASSES,
        grid_resolution=args.grid_resolution,
        class_to_colors=class_to_colors).train().cuda()
    semantic_projection_layer1 = SemanticProjectionLayer(
        camera_height=SegmentationConfig.SCREEN_SIZE,
        camera_width=SegmentationConfig.SCREEN_SIZE,
        vertical_fov=args.vertical_fov,
        map_height=args.map_height,
        map_width=args.map_width,
        map_depth=args.map_depth,
        feature_size=NUM_CLASSES,
        grid_resolution=args.grid_resolution,
        class_to_colors=class_to_colors).train().cuda()

    # run a rearrangement experiment and handle when unity
    run_experiment_with_restart(  # crashes by restarting the experiment
        semantic_mapping_experiment, args,
        occupancy_projection_layer,
        semantic_projection_layer0,
        semantic_projection_layer1)

    # we also use a separate home folder per experiment, which needs
    # to be cleaned up once the experiment successfully terminates
    shutil.rmtree(os.environ["HOME"], onerror=handle_read_only)


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Exploration Agent")

    parser.add_argument("--logdir", type=str,
                        default="/home/btrabucco/train-maps")

    parser.add_argument("--stage",
                        type=str, default="train")
    parser.add_argument("--start-task",
                        type=int, default=0)
    parser.add_argument("--every-tasks",
                        type=int, default=5)
    parser.add_argument("--total-tasks",
                        type=int, default=800)

    parser.add_argument("--ground-truth-segmentation",
                        action='store_true')
    parser.add_argument("--ground-truth-disagreement",
                        action='store_true')
    parser.add_argument("--ground-truth-semantic-search",
                        action='store_true')

    parser.add_argument("--exploration-budget-one",
                        type=int, default=50)
    parser.add_argument("--exploration-budget-two",
                        type=int, default=5)

    parser.add_argument("--detection-threshold",
                        type=float, default=0.9)

    parser.add_argument("--map-height",
                        type=int, default=384)
    parser.add_argument("--map-width",
                        type=int, default=384)
    parser.add_argument("--map-depth",
                        type=int, default=96)
    parser.add_argument("--grid-resolution",
                        type=float, default=0.05)

    parser.add_argument("--map-slice-start",
                        type=int, default=20)
    parser.add_argument("--map-slice-stop",
                        type=int, default=48)
    parser.add_argument("--vertical-fov",
                        type=float, default=90.0)

    parser.add_argument("--obstacle-threshold",
                        type=float, default=0.0)
    parser.add_argument("--obstacle-padding",
                        type=int, default=1)

    parser.add_argument("--contour-padding",
                        type=int, default=0)
    parser.add_argument("--contour-threshold",
                        type=float, default=0.0)
    parser.add_argument("--confidence-threshold",
                        type=float, default=0.0)

    parser.add_argument("--distance-threshold",
                        type=float, default=0.05)
    parser.add_argument("--deformation-threshold",
                        type=float, default=0.0)

    run_experiment(parser.parse_args())
