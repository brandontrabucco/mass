import numpy as np
import torch
import json
import os
import stat
import signal

from typing import Callable, List, Set
from scipy.optimize import linear_sum_assignment

from rearrange.tasks import UnshuffleTask
from ai2thor.exceptions import RestartError
from ai2thor.exceptions import UnityCrashException

from mass.thor.segmentation_config import PICKABLE_TO_COLOR
from mass.thor.segmentation_config import OPENABLE_TO_COLOR
from mass.thor.segmentation_config import ID_TO_PICKABLE
from mass.thor.segmentation_config import ID_TO_OPENABLE


class NumpyJSONEncoder(json.JSONEncoder):
    """Helper class for serializing scalar numpy arrays by converting
    them to integers, floating point number, or lists, depending
    on the data type of the associated numpy array.

    """

    def default(self, obj):
        if isinstance(obj, np.bool):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)


class TimeoutDueToUnityCrash(object):
    """Helper class for detecting when the unity server cannot be reached,
    which indicates that unity has likely crashed due to a cause that
    AI2-THOR does not handle, which causes infinite blocking.

    """

    def __init__(self, seconds: int = 60):
        self.seconds = seconds  # time to wait before raising an error

    def handle_timeout(self, signum, frame):
        raise UnityCrashException("unable to communicate with unity")

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)  # wait before throwing an error

    def __exit__(self, type, value, traceback):
        signal.alarm(0)  # the call finished, disable the alarm


def run_experiment_with_restart(run_experiment: Callable, *args, **kwargs):
    """Helper function for running experiments in AI2-THOR that handles when
    unity crashes and restarts the experiment so that restarts do not
    interfere with experiments running to completion.

    """

    while True:  # unity can be unstable on the cluster
        try:  # so run until the function successfully returns
            return run_experiment(*args, **kwargs)  # without a crash

        # unity will randomly crash on the cluster and we restart it
        except (UnityCrashException, RestartError) as error:
            print("Restarting Due To: {}".format(str(error)))


def handle_read_only(func, path, exc_info):
    """Helper function that allows shutil to recursively delete the temporary
    folder generated in this experiment, which contains files with read-only
    file access, which causes an error and must be modified.

    """

    # check if the path has read only access currently
    if not os.access(path, os.W_OK):  # and change file access
        os.chmod(path, stat.S_IWUSR)  # to file write access before
        func(path)  # calling the remove function on it again afterwards


def get_scene_differences(task: UnshuffleTask):
    """Helper function for extracting a ground truth list of object
    differences from a rearrangement task, and return a list of
    object names that require active rearrangement.

    Arguments:

    task: UnshuffleTask
        an instance of UnshuffleTask with a RearrangeTHOREnvironment
        which we use for calculating object pose differences.

    Returns:

    objects_that_moved: List[str]
        a list containing the names of up to five objects in the
        world that have moves, with possible redundancy.

    """

    # get the poses of objects from the RearrangeTHOREnvironment
    unshuffle_poses, walkthrough_poses, _ = task.env.poses

    # iterate through pairs of objects during the walkthrough phase and
    # unshuffle phase of the world to track differences
    for object_one, object_two in zip(
            unshuffle_poses, walkthrough_poses):

        # if the difference in pose is above an epsilon threshold,
        # then we consider the object to have been moved
        if not task.env.are_poses_equal(object_one, object_two) and (
                object_one["type"] in PICKABLE_TO_COLOR or
                object_one["type"] in OPENABLE_TO_COLOR):

            # yield the name as a string for every movable object
            yield object_one["type"]  # in the scene that is pickable


def get_scene_differences_pose(task: UnshuffleTask):
    """Helper function for extracting a ground truth list of object
    differences from a rearrangement task, and return a list of
    object names that require active rearrangement.

    Arguments:

    task: UnshuffleTask
        an instance of UnshuffleTask with a RearrangeTHOREnvironment
        which we use for calculating object pose differences.

    Returns:

    objects_that_moved: List[str]
        a list containing the names of up to five objects in the
        world that have moves, with possible redundancy.

    """

    # get the poses of objects from the RearrangeTHOREnvironment
    unshuffle_poses, walkthrough_poses, _ = task.env.poses

    # iterate through pairs of objects during the walkthrough phase and
    # unshuffle phase of the world to track differences
    for object_one, object_two in zip(
            unshuffle_poses, walkthrough_poses):

        # if the difference in pose is above an epsilon threshold,
        # then we consider the object to have been moved
        if not task.env.are_poses_equal(object_one, object_two) and (
                object_one["type"] in PICKABLE_TO_COLOR or
                object_one["type"] in OPENABLE_TO_COLOR):

            # yield the name as a string for every movable object
            yield np.array([object_one["position"]["x"], 
                            object_one["position"]["z"], 
                            object_one["position"]["y"]]), \
                  np.array([object_two["position"]["x"], 
                            object_two["position"]["z"], 
                            object_two["position"]["y"]])


def predict_scene_differences(semantic_projection_layer0,
                              semantic_projection_layer1,
                              resnet_projection_layer0,
                              resnet_projection_layer1,
                              objects_moved: Set[int],
                              object_ids_to_move_pred: Set[int],
                              confidence_threshold: float = 0.2,
                              contour_padding: int = 3,
                              contour_threshold: float = 0.0,
                              distance_threshold: float = 0.0,
                              deformation_threshold: float = 0.0):
    """Utility function for predicting objects that have moved given two
    semantic maps representing two states of the world, where
    semantic_projection_layer0 is considered the goal state of the world.

    Arguments:

    semantic_projection_layer0: SemanticProjectionLayer
        a semantic map that contain voxels with class probabilities
        that implements self.find() for localizing objects in the map.
    semantic_projection_layer1: SemanticProjectionLayer
        a semantic map that contain voxels with class probabilities
        that implements self.find() for localizing objects in the map.

    objects_moved: Set[int]
        a set containing the types of objects that have already been
        processed and were attempted to be rearranged.
    object_ids_to_move_pred: Set[int]
        a set containing the types of objects that are candidates to
        be rearranged, which may contains objects already moved.

    confidence_threshold: float
        a threshold for confidence that determines the minimum
        confidence a detection must have to be considered a positive.
    contour_padding: int
        an integer representing the radius of the kernel used for
        smoothing the semantic map when counting the number of objects.
    contour_threshold: float
        a threshold used to determine whether a pixel in the class map
        is considered to be a part of a contour by opencv.

    distance_threshold: float
        a threshold used for determining if two objects in the semantic
        map are considered moved and require being rearranged.
    deformation_threshold: float
        a threshold used for determining if two objects in the semantic
        map are considered opened and require being rearranged.

    Returns:

    object_to_move: int
        an integer representing the class id that was selected for
        rearrangement based on detected object differences.
    object_goals0: List[torch.Tensor]
        a list representing the locations of objects selected for
        rearrangement in semantic_projection_layer0 in world coordinates.
    object_goals1: List[torch.Tensor]
        a list representing the locations of objects selected for
        rearrangement in semantic_projection_layer1 in world coordinates.

    """

    # iterate through the list of objects that have moved
    object_to_move = None  # determine which object to un-shuffle
    object_goals0, object_goals1 = [], []  # store multiple detections

    for candidate_object in object_ids_to_move_pred:
        object_pickable = ID_TO_PICKABLE[candidate_object]
        object_openable = ID_TO_OPENABLE[candidate_object]

        if (candidate_object in objects_moved
                or not any([object_pickable, object_openable])):
            continue  # object already moved or is not recognized

        # get locations of the candidate object in the walk-through
        # and un-shuffle maps and the number of pixels observed
        conf0, goal0, size0, feature0 = semantic_projection_layer0.find(
            candidate_object, contour_padding=contour_padding,
            contour_threshold=contour_threshold,
            confidence_threshold=confidence_threshold,
            feature_map=resnet_projection_layer0)
        conf1, goal1, size1, feature1 = semantic_projection_layer1.find(
            candidate_object, contour_padding=contour_padding,
            contour_threshold=contour_threshold,
            confidence_threshold=confidence_threshold,
            feature_map=resnet_projection_layer1)

        if len(conf0) == 0 or len(conf1) == 0:
            continue  # not enough objects were detected

        # get expected size disparity for all pairs of objects
        # detected in the two phases of rearrangement
        if feature0 is not None and feature1 is not None:
            feature0 = torch.stack(feature0, dim=0)
            feature1 = torch.stack(feature1, dim=0)
            deformation = torch.linalg.norm(
                feature0.unsqueeze(1) - feature1.unsqueeze(0), dim=2)

        # get expected size disparity for all pairs of objects
        # detected in the two phases of rearrangement
        else:
            size0 = torch.stack(size0, dim=0)
            size1 = torch.stack(size1, dim=0)
            deformation = (size0.unsqueeze(1) -
                        size1.unsqueeze(0)).abs()

        # get expected unshuffle distance between the maps
        # for all pairs of objects detected in the two phases
        goal0 = torch.stack(goal0, dim=0)
        goal1 = torch.stack(goal1, dim=0)
        distance = torch.linalg.norm(
            goal0.unsqueeze(1) - goal1.unsqueeze(0), dim=2)

        # solve a minimum weight bipartite matching problem
        # to pair objects between the two rearrangement phases
        instance_ids0, instance_ids1 = linear_sum_assignment(
            (deformation  # using size if objects are pickable
             if object_pickable else  # otherwise use locations
             distance).detach().cpu().numpy())

        # filter objects for unshuffling based on their
        # distance during the two phases of rearrangement
        for instance0, instance1 in \
                zip(instance_ids0, instance_ids1):

            # determine if the object meets the criteria
            # for being moved by the agent during rearrangement
            instance_move = (object_pickable and distance[
                instance0, instance1] > distance_threshold)

            # determine if the object meets the criteria
            # for being opened by the agent during rearrangement
            instance_open = object_openable

            # if the object meets the rearrangement criteria
            # flag the object for rearrangement in this iteration
            if instance_move or instance_open:
                object_to_move = candidate_object
                object_goals0.append(goal0[instance0])
                object_goals1.append(goal1[instance1])

        if object_to_move is not None:
            break  # break to ensure no other objects are checked

    return object_to_move, object_goals0, object_goals1
