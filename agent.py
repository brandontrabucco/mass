import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as functional

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
from mass.nn.applications\
    .resnet_projection_layer import ResNetProjectionLayer

from mass.utils.experimentation import run_experiment_with_restart
from mass.utils.experimentation import TimeoutDueToUnityCrash
from mass.utils.experimentation import NumpyJSONEncoder
from mass.utils.experimentation import handle_read_only

from mass.utils.experimentation import get_scene_differences
from mass.utils.experimentation import get_scene_differences_pose
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


def visualization_callback(controller: NavigationPolicy, writer: FFmpegWriter,
                           slice_start: int = 4, slice_stop: int = 32):
    """Helper function that returns a callback function for visualization
    that writes video frames to the disk for the semantic maps, rbg vision,
    and semantic predictions from the Mask R-CNN.

    Arguments:

    controller: DeterministicController
        an instance of the DeterministicController that manages path
        planning and low level interactions with the semantic maps.
    writer: FFmpegWriter
        an instance of the FFmpegWriter, that writes to a task-specific
        file on the local disk for visualizing episodes.

    Returns:

    callback: Callable
        a callback function that will be called at every step of path
        planning for visualizing episodes and the semantic maps.

    """

    image_id = 0

    os.makedirs("images", exist_ok=True)

    def process_obs(obs):

        nonlocal image_id

        # cv2.imwrite(f"images/frame-{image_id}.png", (255.0 * obs["rgb"][..., ::-1]).astype(np.uint8))
        # cv2.imwrite(f"images/depth-{image_id}.png", (255.0 * (obs["depth"][..., 0] / 5.0)).clip(min=0, max=255).astype(np.uint8))

        image_id += 1

        writer.writeFrame(255.0 * np.concatenate(
            [obs["rgb"]] + [cv2.resize(visualization, (
                int(visualization.shape[1] /
                    visualization.shape[0] *
                    SegmentationConfig.SCREEN_SIZE),
                SegmentationConfig.SCREEN_SIZE),
                                       interpolation=cv2.INTER_AREA)
                            for visualization in (
                                controller.feature_maps["occupancy_projection_layer"]
                                    .visualize(obs, depth_slice=slice(slice_start, slice_stop)),
                                controller.feature_maps["semantic_projection_layer0"]
                                    .visualize(obs, depth_slice=slice(0, slice_stop)),
                                controller.feature_maps["semantic_projection_layer1"]
                                    .visualize(obs, depth_slice=slice(0, slice_stop)))], axis=1))

    # return a callback function for writing frames to the disk at
    # every iteration of path planning during an episode
    return process_obs


PHASE_ONE_MAPS_TO_UPDATE = [
    "occupancy_projection_layer",
    "semantic_projection_layer0",
    "resnet_projection_layer0"
]


PHASE_TWO_MAPS_TO_UPDATE = [
    "semantic_projection_layer1", 
    "resnet_projection_layer1"
]


def semantic_mapping_experiment(args, model, occupancy_projection_layer,
                                semantic_projection_layer0,
                                semantic_projection_layer1,
                                resnet_projection_layer0,
                                resnet_projection_layer1):
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

    # get the names of objects that can be detected by our Mask-RCNN
    class_names = list(CLASS_TO_COLOR.keys())
    depth_slice = slice(args.map_slice_start, args.map_slice_stop)

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
        if args.use_feature_matching:
            resnet_projection_layer0.reset(**origin_kwargs)
            resnet_projection_layer1.reset(**origin_kwargs)

        # build a controller that manages low level interactions with
        # the environment and path planning to goal locations
        controller = NavigationPolicy(
            task, "occupancy_projection_layer",
            step_size=5, depth_slice=depth_slice,
            padding=args.obstacle_padding,
            obstacle_threshold=args.obstacle_threshold,
            position_noise_std=args.position_noise_std,
            rotation_noise_std=args.rotation_noise_std,
            occupancy_projection_layer=occupancy_projection_layer,
            semantic_projection_layer0=semantic_projection_layer0,
            semantic_projection_layer1=semantic_projection_layer1,
            resnet_projection_layer0=resnet_projection_layer0,
            resnet_projection_layer1=resnet_projection_layer1)

        # write a video to the disk for each task in the videos folder
        writer = FFmpegWriter(  # yuv420p ensures videos work on Windows
            os.path.join(args.logdir, "videos", f"{task_id}.mp4"),
            outputdict={"-pix_fmt": "yuv420p", "-b": "30000000000"})

        # this is a hack, but the agent works better when facing downwards
        callback = visualization_callback(controller, writer,
                                          slice_start=args.map_slice_start,
                                          slice_stop=args.map_slice_stop)

        # sample the next task in sequence for evaluation
        with TimeoutDueToUnityCrash():  # wait 60 seconds for unity
            controller.task = task_sampler.next_task()

        if args.record_found_objects:
            agent_positions_unshuffle = []
            agent_positions_walkthrough = []

        walkthrough_semantic_search_goals = []
        unshuffle_semantic_search_goals = []

        rearrangement_analytics = []
        current_poses = controller.task.env.poses[2]
        for object_two, object_one in zip(*controller.task.env.poses[1:]):

            all_neighbor_distances = [np.linalg.norm(
                np.array(list(object_one["position"].values())) -
                np.array(list(x["position"].values())))
                for x in current_poses
                if x["name"] != object_one["name"]]

            pickable_neighbor_distances = [np.linalg.norm(
                np.array(list(object_one["position"].values())) -
                np.array(list(x["position"].values())))
                for x in current_poses
                if x["name"] != object_one["name"]
                and x["type"] in PICKABLE_TO_COLOR]

            type_neighbor_distances = [np.linalg.norm(
                np.array(list(object_one["position"].values())) -
                np.array(list(x["position"].values())))
                for x in current_poses
                if x["name"] != object_one["name"]
                and x["type"] == object_one["type"]]

            pos1 = np.array(list(object_one["position"].values()))
            pos2 = np.array(list(object_two["position"].values()))
            distance = np.linalg.norm(pos1 - pos2)

            correct = not object_one["broken"] and controller.\
                task.env.are_poses_equal(object_one, object_two)

            if not correct and args.ground_truth_semantic_search:

                walkthrough_semantic_search_goals.append(
                    np.array([object_two["position"]["x"],
                              object_two["position"]["z"],
                              object_two["position"]["y"]]))
                unshuffle_semantic_search_goals.append(
                    np.array([object_one["position"]["x"],
                              object_one["position"]["z"],
                              object_one["position"]["y"]]))

            size = 0
            if object_one["bounding_box"] is not None:
                bbox = np.array(object_one["bounding_box"])
                size = np.prod(bbox.max(axis=0) - bbox.min(axis=0))

            openness = 0
            if object_one["openness"] is not None:
                openness = np.abs(object_one["openness"] -
                                  object_two["openness"])

            rearrangement_analytics.append(dict(

                initial_openness=openness,
                initial_distance=distance,
                initial_correct=correct,

                initial_min_distance_all=(
                    np.min(all_neighbor_distances)
                    if len(all_neighbor_distances) > 0 else None
                ),
                initial_min_distance_pickable=(
                    np.min(pickable_neighbor_distances)
                    if len(pickable_neighbor_distances) > 0 else None
                ),
                initial_min_distance_type=(
                    np.min(type_neighbor_distances)
                    if len(type_neighbor_distances) > 0 else None
                ),

                initial_mean_distance_all=(
                    np.mean(all_neighbor_distances)
                    if len(all_neighbor_distances) > 0 else None
                ),
                initial_mean_distance_pickable=(
                    np.mean(pickable_neighbor_distances)
                    if len(pickable_neighbor_distances) > 0 else None
                ),
                initial_mean_distance_type=(
                    np.mean(type_neighbor_distances)
                    if len(type_neighbor_distances) > 0 else None
                ),

                size=size, type=object_one["type"],
                pickable=object_one["type"] in PICKABLE_TO_COLOR,
                openable=object_one["type"] in OPENABLE_TO_COLOR

            ))

        task_sampler.reset()

        for _ in range(args.start_task):
            with TimeoutDueToUnityCrash():
                next(task_sampler.task_spec_iterator)

        # sample the next task in sequence for evaluation
        with TimeoutDueToUnityCrash():  # wait 60 seconds for unity
            controller.task = task_sampler.next_task()

        # this is a hack, but the agent works better when facing downwards
        controller.task.step(action=controller
                             .task.action_names().index('look_down'))

        num_goals = 0
        while not controller.task.is_done() and \
                num_goals < args.exploration_budget_one:
            num_goals += 1

            # rebuild the navigation mesh at every step in the first five
            # random goals, otherwise rebuild only at the first step
            goal = controller.sample_navigation_goal(
                controller.get_observations(), "occupancy_projection_layer")

            if len(walkthrough_semantic_search_goals) > 0:
                goal = torch.as_tensor(walkthrough_semantic_search_goals
                                       .pop(0), dtype=torch.float32)

            if args.semantic_search_walkthrough:

                prediction = model(semantic_projection_layer0.data
                                   .amax(dim=2).unsqueeze(0).permute(0, 3, 1, 2))
                prediction = functional.softmax(
                    prediction.view(args.map_height * args.map_width), dim=0)

                goal = torch.multinomial(prediction, 1)
                goal = torch.cat([goal % args.map_width,
                                  (goal // args.map_width) % args.map_height,
                                  torch.zeros_like(goal)])
                goal = semantic_projection_layer0.map_to_world(goal)

            for current_observations in controller.navigate_to(
                    goal, "occupancy_projection_layer",
                    depth_slice=depth_slice,
                    padding=args.obstacle_padding,
                    obstacle_threshold=args.obstacle_threshold,
                    update_map=PHASE_ONE_MAPS_TO_UPDATE):

                callback(current_observations)  # visualize agent behavior

                if args.record_found_objects:
                    location = controller.task.env.get_agent_location()
                    agent_positions_walkthrough.append(
                        np.array([location["x"], location["z"]])
                    )

        # sample the next task in sequence for evaluation
        with TimeoutDueToUnityCrash():  # wait 60 seconds for unity
            controller.task = task_sampler.next_task()

        # this is a hack, but the agent works better when facing downwards
        controller.task.step(action=controller
                             .task.action_names().index('look_down'))

        # obtain a ground truth list of objects that have been shuffled
        object_ids_to_move = set([class_names.index(n) for n in
                                  get_scene_differences(controller.task)])
        print("[Task={}] Ground Truth: {}".format(task_id, ", ".join([
            class_names[x] for x in object_ids_to_move])))

        if args.record_found_objects:
            object_positions_unshuffle, object_positions_walkthrough = zip(*get_scene_differences_pose(controller.task))
            object_positions_unshuffle = np.stack(object_positions_unshuffle, axis=0)[:, :2]
            object_positions_walkthrough = np.stack(object_positions_walkthrough, axis=0)[:, :2]

        num_goals = 0
        while not controller.task.is_done() and \
                num_goals < args.exploration_budget_two:
            num_goals += 1

            # rebuild the navigation mesh at every step in the first five
            # random goals, otherwise rebuild only at the first step
            goal = controller.sample_navigation_goal(
                controller.get_observations(), "occupancy_projection_layer")

            if len(unshuffle_semantic_search_goals) > 0:
                goal = torch.as_tensor(unshuffle_semantic_search_goals
                                       .pop(0), dtype=torch.float32)

            if args.semantic_search_unshuffle:

                prediction = model(semantic_projection_layer1.data
                                   .amax(dim=2).unsqueeze(0).permute(0, 3, 1, 2))
                prediction = functional.softmax(
                    prediction.view(args.map_height * args.map_width), dim=0)

                goal = torch.multinomial(prediction, 1)
                goal = torch.cat([goal % args.map_width,
                                  (goal // args.map_width) % args.map_height,
                                  torch.zeros_like(goal)])
                goal = semantic_projection_layer1.map_to_world(goal)

            for current_observations in controller.navigate_to(
                    goal, "occupancy_projection_layer",
                    depth_slice=depth_slice,
                    padding=args.obstacle_padding,
                    obstacle_threshold=args.obstacle_threshold,
                    update_map=PHASE_TWO_MAPS_TO_UPDATE):

                callback(current_observations)  # visualize agent behavior

                if args.record_found_objects:
                    location = controller.task.env.get_agent_location()
                    agent_positions_unshuffle.append(
                        np.array([location["x"], location["z"]])
                    )

        # np.save("walkthrough-map.npy", semantic_projection_layer0.data.detach().cpu().numpy())
        # np.save("unshuffle-map.npy", semantic_projection_layer1.data.detach().cpu().numpy())

        # this needs to be improved later, but
        objects_moved = set()  # only allow each object to be moved once

        # after exploration, locate object differences between the maps
        while not controller.task.is_done():  # and iteratively correct them

            # compute a sorted list of object ids that have the largest map
            # disagreement, ignoring the first two semantic channels
            object_ids_to_move_pred = list(range(NUM_CLASSES))

            # enable ground truth injection of which objects need
            if args.ground_truth_disagreement:  # unshuffling by the agent
                object_ids_to_move_pred = object_ids_to_move

            # calculate which object to move next by tracking
            (object_to_move, object_goals0,  # where the maps disagree
             object_goals1) = predict_scene_differences(
                semantic_projection_layer0, semantic_projection_layer1,
                resnet_projection_layer0, resnet_projection_layer1,
                objects_moved, object_ids_to_move_pred,
                confidence_threshold=args.confidence_threshold,
                contour_padding=args.contour_padding,
                contour_threshold=args.contour_threshold,
                distance_threshold=args.distance_threshold,
                deformation_threshold=args.deformation_threshold)

            if object_to_move is None or controller.task.is_done():
                break  # if no object no map differences were detected

            # set the selected object as being moved in the current step
            objects_moved.add(object_to_move)
            print("[Task={}] Moving: {}"
                  .format(task_id, class_names[object_to_move]))

            # compute distances from each current object location
            distances = torch.norm(  # to each target object location
                torch.stack(object_goals0, dim=0).unsqueeze(1) -
                torch.stack(object_goals1, dim=0).unsqueeze(0), dim=2)
            rearrangement_order = distances.amin(
                dim=1).argsort(descending=True).cpu().tolist()

            # rearrange object in an order that is least likely
            object_goals0 = [object_goals0[idx]  # to cause conflicts
                             for idx in rearrangement_order]
            object_goals1 = [object_goals1[idx]
                             for idx in rearrangement_order]

            for object_goal0, object_goal1 in \
                    zip(object_goals0, object_goals1):

                for current_observations in controller.navigate_to(
                        object_goal1, "occupancy_projection_layer",
                        depth_slice=depth_slice,
                        padding=args.obstacle_padding,
                        obstacle_threshold=args.obstacle_threshold,
                        update_map=PHASE_TWO_MAPS_TO_UPDATE):

                    callback(current_observations)  # visualize behavior

                    if args.record_found_objects:
                        location = controller.task.env.get_agent_location()
                        agent_positions_unshuffle.append(
                            np.array([location["x"], location["z"]])
                        )

                if controller.task.is_done():
                    break  # break from the object main loop

                # send the pickup action corresponding to the object
                # or the open action if the object is openable not pickable
                action_name = ('pickup_{}' if class_names[object_to_move]
                               in PICKABLE_TO_COLOR
                               else 'open_by_type_{}').format(
                    pattern.sub('_', class_names[object_to_move]).lower())
                controller.task.step(action=controller.task
                                     .action_names().index(action_name))

                if controller.task.is_done():
                    break  # break from the object main loop

                elif class_names[object_to_move] in OPENABLE_TO_COLOR:
                    continue  # object that was rearranged was not pickable

                for current_observations in controller.navigate_to(
                        object_goal0, "occupancy_projection_layer",
                        depth_slice=depth_slice,
                        padding=args.obstacle_padding,
                        obstacle_threshold=args.obstacle_threshold,
                        update_map=PHASE_TWO_MAPS_TO_UPDATE):

                    callback(current_observations)  # visualize behavior

                    if args.record_found_objects:
                        location = controller.task.env.get_agent_location()
                        agent_positions_unshuffle.append(
                            np.array([location["x"], location["z"]])
                        )

                if controller.task.is_done():
                    break  # break from the object main loop

                # send the drop action corresponding to the target object
                # in the future this should check if goal is in the viewport
                controller.task.step(action=controller.task.action_names()
                                     .index('drop_held_object_with_snap'))

                if controller.task.is_done():
                    break  # break from the object main loop

        # if the simulator has not terminated automatically
        if not controller.task.is_done():  # then send a done action
            controller.task.step(action=controller
                                 .task.action_names().index('done'))

        writer.close()

        current_poses = controller.task.env.poses[2]
        for i, (object_two, object_one) in \
                enumerate(zip(*controller.task.env.poses[1:])):

            assert rearrangement_analytics[i]["type"] == object_one["type"]

            all_neighbor_distances = [np.linalg.norm(
                np.array(list(object_one["position"].values())) -
                np.array(list(x["position"].values())))
                for x in current_poses
                if x["name"] != object_one["name"]]

            pickable_neighbor_distances = [np.linalg.norm(
                np.array(list(object_one["position"].values())) -
                np.array(list(x["position"].values())))
                for x in current_poses
                if x["name"] != object_one["name"]
                and x["type"] in PICKABLE_TO_COLOR]

            type_neighbor_distances = [np.linalg.norm(
                np.array(list(object_one["position"].values())) -
                np.array(list(x["position"].values())))
                for x in current_poses
                if x["name"] != object_one["name"]
                and x["type"] == object_one["type"]]

            pos1 = np.array(list(object_one["position"].values()))
            pos2 = np.array(list(object_two["position"].values()))
            distance = np.linalg.norm(pos1 - pos2)

            correct = not object_one["broken"] and controller.\
                task.env.are_poses_equal(object_one, object_two)

            openness = 0
            if object_one["openness"] is not None:
                openness = np.abs(object_one["openness"] -
                                  object_two["openness"])

            rearrangement_analytics[i]["final_openness"] = openness
            rearrangement_analytics[i]["final_distance"] = distance
            rearrangement_analytics[i]["final_correct"] = correct

            rearrangement_analytics[i]["final_min_distance_all"] = (
                np.min(all_neighbor_distances)
                if len(all_neighbor_distances) > 0 else None
            )
            rearrangement_analytics[i]["final_min_distance_pickable"] = (
                np.min(pickable_neighbor_distances)
                if len(pickable_neighbor_distances) > 0 else None
            )
            rearrangement_analytics[i]["final_min_distance_type"] = (
                np.min(type_neighbor_distances)
                if len(type_neighbor_distances) > 0 else None
            )

            rearrangement_analytics[i]["final_mean_distance_all"] = (
                np.mean(all_neighbor_distances)
                if len(all_neighbor_distances) > 0 else None
            )
            rearrangement_analytics[i]["final_mean_distance_pickable"] = (
                np.mean(pickable_neighbor_distances)
                if len(pickable_neighbor_distances) > 0 else None
            )
            rearrangement_analytics[i]["final_mean_distance_type"] = (
                np.mean(type_neighbor_distances)
                if len(type_neighbor_distances) > 0 else None
            )

            rearrangement_analytics[i]["num_instances"] = sum([
                1 if instance["type"] == object_one["type"] else 0
                for instance in rearrangement_analytics
            ])

        df = pd.DataFrame.from_records(rearrangement_analytics)
        df.to_csv(os.path.join(args.logdir, "results", f"analytics-{task_id}.csv"))

        if args.record_found_objects:

            agent_positions_unshuffle = np.stack(agent_positions_unshuffle, axis=0)
            agent_positions_walkthrough = np.stack(agent_positions_walkthrough, axis=0)

            distances_unshuffle = np.linalg.norm(
                agent_positions_unshuffle[np.newaxis] - 
                object_positions_unshuffle[:, np.newaxis], axis=-1)

            distances_walkthrough = np.linalg.norm(
                agent_positions_walkthrough[np.newaxis] - 
                object_positions_walkthrough[:, np.newaxis], axis=-1)

            np.save(os.path.join(
                args.logdir, "results", 
                f"objects-found-unshuffle-{task_id}.npy"), distances_unshuffle)

            np.save(os.path.join(
                args.logdir, "results", 
                f"objects-found-walkthrough-{task_id}.npy"), distances_walkthrough)

        # calculate the evaluation metrics for this task
        with open(os.path.join(  # and write them to a results folder
                args.logdir, "results", f"{task_id}.json"), "w") as f:

            # generate a dictionary of evaluation metrics
            metrics = controller.task.metrics()

            # log what object the agent tried to move and if it should have
            metrics["unshuffle/objects_moved"] = [
                class_names[x] for x in objects_moved]
            metrics["unshuffle/objects_moved_accuracy"] = [
                1 if x in object_ids_to_move else 0 for x in objects_moved]

            # log what object the agent should have moved and they were
            metrics["unshuffle/objects_to_move"] = [
                class_names[x] for x in object_ids_to_move]
            metrics["unshuffle/objects_to_move_accuracy"] = [
                1 if x in objects_moved else 0 for x in object_ids_to_move]

            # write the metrics to a json file for each task
            json.dump(metrics, f, indent=4, cls=NumpyJSONEncoder)

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
    os.makedirs(os.path.join(args.logdir, f"videos"), exist_ok=True)
    os.makedirs(os.path.join(args.logdir, f"results"), exist_ok=True)
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

    resnet_projection_layer0 = None
    resnet_projection_layer1 = None

    if args.use_feature_matching:
        
        resnet_projection_layer0 = ResNetProjectionLayer(
            camera_height=SegmentationConfig.SCREEN_SIZE,
            camera_width=SegmentationConfig.SCREEN_SIZE,
            vertical_fov=args.vertical_fov,
            map_height=args.map_height,
            map_width=args.map_width,
            map_depth=args.map_depth,
            feature_size=256,
            grid_resolution=args.grid_resolution).train()
        resnet_projection_layer1 = ResNetProjectionLayer(
            camera_height=SegmentationConfig.SCREEN_SIZE,
            camera_width=SegmentationConfig.SCREEN_SIZE,
            vertical_fov=args.vertical_fov,
            map_height=args.map_height,
            map_width=args.map_width,
            map_depth=args.map_depth,
            feature_size=256,
            grid_resolution=args.grid_resolution).train()

    model = nn.Sequential(
        nn.Conv2d(54, 64, 3, padding=1),

        nn.GroupNorm(1, 64),
        nn.ReLU(),

        nn.Conv2d(64, 64, 3, padding=1),

        nn.GroupNorm(1, 64),
        nn.ReLU(),

        nn.Conv2d(64, 64, 3, padding=1),

        nn.GroupNorm(1, 64),
        nn.ReLU(),

        nn.Conv2d(64, 64, 3, padding=1),

        nn.GroupNorm(1, 64),
        nn.ReLU(),

        nn.Conv2d(64, 1, 3, padding=1),
    )

    model.load_state_dict(torch.load("policy.pth"))
    model.eval()
    model.cuda()

    # run a rearrangement experiment and handle when unity
    run_experiment_with_restart(  # crashes by restarting the experiment
        semantic_mapping_experiment, args, model,
        occupancy_projection_layer,
        semantic_projection_layer0,
        semantic_projection_layer1,
        resnet_projection_layer0,
        resnet_projection_layer1)

    # we also use a separate home folder per experiment, which needs
    # to be cleaned up once the experiment successfully terminates
    shutil.rmtree(os.environ["HOME"], onerror=handle_read_only)


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Exploration Agent")

    parser.add_argument("--logdir", type=str,
                        default="/home/btrabucco/test-data")

    parser.add_argument("--stage",
                        type=str, default="train")
    parser.add_argument("--start-task",
                        type=int, default=0)
    parser.add_argument("--every-tasks",
                        type=int, default=1)
    parser.add_argument("--total-tasks",
                        type=int, default=50)

    parser.add_argument("--ground-truth-segmentation",
                        action='store_true')
    parser.add_argument("--ground-truth-disagreement",
                        action='store_true')
    parser.add_argument("--ground-truth-semantic-search",
                        action='store_true')
    parser.add_argument("--semantic-search-walkthrough",
                        action='store_true')
    parser.add_argument("--semantic-search-unshuffle",
                        action='store_true')
    parser.add_argument("--use-feature-matching",
                        action='store_true')
    parser.add_argument("--record-found-objects",
                        action='store_true')

    parser.add_argument("--exploration-budget-one",
                        type=int, default=5)
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

    parser.add_argument("--position-noise-std",
                        type=float, default=0.0)
    parser.add_argument("--rotation-noise-std",
                        type=float, default=0.0)

    args = parser.parse_args()

    if not args.use_feature_matching:
        PHASE_ONE_MAPS_TO_UPDATE.pop()
        PHASE_TWO_MAPS_TO_UPDATE.pop()

    run_experiment(args)
