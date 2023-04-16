from mass.nn.base_projection_layer import BaseProjectionLayer
from typing import Any, Dict, Union, List
from itertools import product, count
import torch.nn.functional as functional
import torch
import networkx
import numpy as np


class NavigationPolicy(object):
    """Controller that wraps around the AI2-THOR environment and provides an
    interface for navigating to point-goals in the environment that can be
    sampled from an upper-level semantic exploration policy.

    Arguments:

    task: AbstractRearrangeTask
        an instance of the AI2-THOR Rearrange task that contains an agent
        that we will use for navigation.
    feature_maps: Dict[ProjectionLayer]
        a torch Module containing the semantic map and helper functions
        for scattering vision observations onto the semantic map.

    """

    def __init__(self, task, navigation_map: str, step_size: int = 5,
                 padding: int = 3, depth_slice: slice = None,
                 obstacle_threshold: float = 0.0,
                 position_noise_std: float = 0.005,
                 rotation_noise_std: float = 0.0087,
                 **feature_maps: BaseProjectionLayer):
        """Controller that wraps around the AI2-THOR environment and provides
        an interface for navigating to point-goals in the environment that
        can be sampled from an upper-level semantic exploration policy.

        Arguments:

        task: AbstractRearrangeTask
            an instance of the AI2-THOR Rearrange task that contains an
            agent that we will use for navigation.

        navigation_map: str
            a string that represents a key into the semantic_mappers dict
            and specifies which map to use for navigation.
        step_size: int
            the number of grid cells to skip when allocating a point in
            the navigation mesh representing the agent's step size.
        padding: int
            an integer representing the number of voxels around occupied
            space in manhattan distance to consider occupied.
        depth_slice: slice
            a depth slice that specifies which voxel to use when
            determining which areas in the map are blocked by obstacles.
        obstacle_threshold: float
            threshold on the one-norm of the feature vector at every channel
            in the feature map to determine if that voxel is occupied.

        feature_maps: Dict[BaseProjectionLayer]
            a torch Module containing the semantic map and helper functions
            for scattering vision observations onto the semantic map.

        """

        self.task = task
        self.feature_maps = feature_maps
        self.navigation_graph: networkx.Graph = None

        self.position_noise_std = position_noise_std
        self.rotation_noise_std = rotation_noise_std

        self.reset_navigation_graph(
            navigation_map, step_size=step_size, padding=padding,
            depth_slice=depth_slice, obstacle_threshold=obstacle_threshold)

    def get_observations(self, *args, **kwargs):
        return self.task.get_observations(*args, **kwargs)

    def process_position(self):
        """Utility function for processing the dict of observations from the
        environment and generating a position tensor in pytorch that will
        be added to the observations under the 'position' key.

        Returns:

        position: torch.Tensor
            a tensor representing the position of the agent in world
            coordinates that is used as the origin of the pinhole camera.

        """

        # grab the position of the agent and copy to the right device
        # the absolute position of the agent in meters
        location = self.task.env.get_agent_location()
        crouch_height_offset = 0.675 if not location["standing"] else 0.0
        return torch.FloatTensor([location["x"], location["z"],
                                  location["y"] - crouch_height_offset])

    def process_yaw(self):
        """Utility function for processing the dict of observations from the
        environment and generating a yaw tensor in pytorch that will
        be added to the observations under the 'yaw' key.

        Returns:

        yaw: torch.Tensor
            a tensor representing the yaw of the agent in counterclockwise
            radians that is used as the yaw of the pinhole camera.

        """

        # grab the yaw of the agent ensuring that zero yaw corresponds to
        # the x axis, and positive yaw is rotates counterclockwise
        yaw = self.task.env.get_agent_location()["rotation"] / 180.0 * np.pi
        return torch.tensor(-yaw + np.pi / 2, dtype=torch.float32)

    def process_elevation(self):
        """Utility function for processing the dict of observations from the
        environment and generating an elevation tensor in pytorch that will
        be added to the observations under the 'elevation' key.

        Returns:

        elevation: torch.Tensor
            a tensor representing the elevation of the agent in
            radians that is used as the elevation of the pinhole camera.

        """

        # grab the yaw of the agent ensuring that zero yaw corresponds to
        # the x axis, and positive yaw is rotates counterclockwise
        return torch.tensor(-self.task.env.get_agent_location()["horizon"]
                            / 180.0 * np.pi, dtype=torch.float32)

    def process_observations(self, observations: Dict[str, Any],
                             update_map: Union[str, List[str]] = None):
        """Utility function that grabs the current observation from the
        AI2-THOR Rearrange task environment, prepares the observations to an
        appropriate format, and updates the existing semantic map.

        Arguments:

        observations: Dict[str, Any]
            a dictionary containing observations from the environment,
            returned by calling the self.task.get_observations() function.
        update_map: Union[str, List[str]]
            a string representing a key into the semantic_mappers dict that
            controls which semantic maps to update.

        """

        # generate additional observations for the agent's pose which
        # are used to align the camera coordinate system during mapping
        observations["position"] = self.process_position()
        observations["yaw"] = self.process_yaw()
        observations["elevation"] = self.process_elevation()

        position_noise = np.random.normal() * self.position_noise_std
        rotation_noise = np.random.normal() * self.rotation_noise_std

        observations["position"] += position_noise
        observations["yaw"] += rotation_noise

        # scatter observations onto the selected feature map
        if update_map is not None:  # by constructing a point cloud

            # iterate through every map to update and copy
            for name in ([update_map] if not  # inputs to the right device
                         isinstance(update_map, list) else update_map):

                # update the feature map by projecting observations
                self.feature_maps[name].update(observations)

    def navigable_area(self, navigation_map: str, padding: int = 3,
                       depth_slice: slice = None,
                       obstacle_threshold: float = 0.0):
        """Helper function for analyzing the semantic map and determining
        which locations are traversable by the agent, defined as floor
        space that is not obstructed by objects, even movable ones.

        Arguments:

        navigation_map: str
            a string that represents a key into the semantic_mappers dict
            and specifies which map to use for navigation.
        padding: int
            an integer representing the number of voxels around occupied
            space in manhattan distance to consider occupied.
        depth_slice: slice
            a depth slice that specifies which voxel to use when
            determining which areas in the map are blocked by obstacles.
        obstacle_threshold: float
            threshold on the one-norm of the feature vector at every channel
            in the feature map to determine if that voxel is occupied.

        Returns:

        navigable: torch.Tensor
            a view of the semantic map that represents which locations in
            the semantic map are traversable by the agent.

        """

        # select the feature map to use for navigation to determine
        feature_map = self.feature_maps[navigation_map]  # traversability

        # build an occupancy map and determine which areas of the floor
        # the agent can traverse and what areas are obstructed
        navigable = torch.norm(feature_map.data,
                               p=1, dim=3) > obstacle_threshold

        # typically the map will include the ceiling which should
        if depth_slice is not None:  # not be used when calculating occupancy
            navigable = navigable[:, :, depth_slice]

        # a pixel is navigable if no voxel in the depth slice
        navigable = torch.logical_not(  # at that location is occupied
            navigable.any(dim=2)).to(dtype=feature_map.data.dtype)

        # add padding around obstacles to reduce the chance of getting stuck
        return 1 - functional.max_pool2d(1 - navigable.unsqueeze(
            0), 2 * padding + 1, stride=1, padding=padding).squeeze(0)

    def reset_navigation_graph(self, navigation_map: str, step_size: int = 5,
                               padding: int = 3, depth_slice: slice = None,
                               obstacle_threshold: float = 0.0):
        """Utility function that assembled a graph where every node
        corresponds to a pixel in the xy plane of the semantic map that is
        navigable, and a set of valid waypoints for navigation.

        navigation_map: str
            a string that represents a key into the semantic_mappers dict
            and specifies which map to use for navigation.
        step_size: int
            the number of grid cells to skip when allocating a point in
            the navigation mesh representing the agent's step size.
        padding: int
            an integer representing the number of voxels around occupied
            space in manhattan distance to consider occupied.
        slice: slice
            a depth slice that specifies which voxel to use when
            determining which areas in the map are blocked by obstacles.
        obstacle_threshold: float
            threshold on the one-norm of the feature vector at every channel
            in the feature map to determine if that voxel is occupied.

        """

        # build the current traversability image by checking where there is
        # floor in the semantic map, and no obstacles present
        navigable_area = self.navigable_area(navigation_map,
                                             padding=padding,
                                             depth_slice=depth_slice,
                                             obstacle_threshold=
                                             obstacle_threshold)

        # select the feature map to use for navigation to determine
        feature_map = self.feature_maps[navigation_map]  # traversability

        # grab the origin of the world coordinate system and copy it to
        # the right device to be converted to the map
        origin_x = torch.tensor(feature_map.origin_x, dtype=torch.float32,
                                device=feature_map.data.device)
        origin_y = torch.tensor(feature_map.origin_y, dtype=torch.float32,
                                device=feature_map.data.device)

        # offset required to ensure the voxel at the origin of the map
        offset = torch.stack([  # has a graph node given the agent step_size
            torch.bucketize(origin_x, feature_map.bins_x, right=True) - 1,
            feature_map.bins_y.size(dim=0) - 1 -
            torch.bucketize(origin_y, feature_map.bins_y,
                            right=True)], dim=-1).cpu().numpy() % step_size

        self.navigation_graph = networkx.Graph()

        # add nodes to the graph given the agent's step size
        for i, j in product(  # so that the origin of the map has a node
                range(offset[1], feature_map.map_height, step_size),
                range(offset[0], feature_map.map_width, step_size)):

            # get the xy position of neighboring nodes and ensure that
            # the new node is navigable before adding it
            for di, dj in ((step_size, 0), (0, step_size)):

                y, x = i + di, j + dj  # get neighbors coordinates

                # check that both bodes are within the map boundaries
                if (0 <= y < feature_map.map_height and
                        0 <= x < feature_map.map_width and

                        # check the area between nodes is not blocked
                        (navigable_area[min(i, y):max(i, y)+1,
                                        min(j, x):max(j, x)+1] == 1).all()):

                    self.navigation_graph.add_edge((j, i), (x, y))

    def update_navigation_graph(self, navigation_map: str, padding: int = 3,
                                depth_slice: slice = None,
                                obstacle_threshold: float = 0.0):
        """Utility function that assembled a graph where every node
        corresponds to a pixel in the xy plane of the semantic map that is
        navigable, and a set of valid waypoints for navigation.

        Arguments:

        navigation_map: str
            a string that represents a key into the semantic_mappers dict
            and specifies which map to use for navigation.
        padding: int
            an integer representing the number of voxels around occupied
            space in manhattan distance to consider occupied.
        depth_slice: slice
            a depth slice that specifies which voxel to use when
            determining which areas in the map are blocked by obstacles.
        obstacle_threshold: float
            threshold on the one-norm of the feature vector at every channel
            in the feature map to determine if that voxel is occupied.

        """

        # build the current traversability image by checking where there is
        # floor in the semantic map, and no obstacles present
        navigable_area = self.navigable_area(navigation_map,
                                             padding=padding,
                                             depth_slice=depth_slice,
                                             obstacle_threshold=
                                             obstacle_threshold)

        for (j, i) in list(self.navigation_graph.nodes()):

            # for every node, check if the space occupied by the node
            if navigable_area[i, j] == 0:  # is obstructed by an obstacle

                self.navigation_graph.remove_node((j, i))

        for (j, i), (x, y) in list(self.navigation_graph.edges()):

            # for every pair of nodes, check if there is any obstructed
            if (navigable_area[min(i, y):max(i, y)+1,  # space between nodes
                               min(j, x):max(j, x)+1] == 0).any():

                self.navigation_graph.remove_edge((j, i), (x, y))

    def navigable_points(self, navigation_map: str, position: torch.Tensor):
        """Utility function that takes in the current navmesh built from the
        semantic map and the current position of the agent and returns which
        points are navigable from the agent's location.

        Arguments:

        navigation_map: str
            a string that represents a key into the semantic_mappers dict
            and specifies which map to use for navigation.
        position: torch.Tensor
            a tensor representing the current location of the agent, which is
            used to find which points are navigable in the environment.

        Returns:

        navigable_points: torch.Tensor
            a list of possible waypoints that are nodes in the graph, no
            other coordinates make valid sources or targets.

        """

        # select the feature map to use for path planning and navigation
        feature_map = self.feature_maps[navigation_map]

        # convert the set of nodes in the graph to a torch tensor
        kwargs = dict(dtype=torch.int32, device=feature_map.data.device)
        points = torch.tensor(list(self.navigation_graph.nodes), **kwargs)

        # convert the source location from the world coordinate system to
        # the map coordinate system by binning the points
        position = feature_map.world_to_map(position)[..., :2]

        # calculate the closest source and target points in the support
        # of the nodes in the navigable graph
        source_idx = torch.sqrt(torch.square(points - position.view(
            1, 2)).sum(dim=-1).to(torch.float32)).argmin(dim=0)

        # convert the source and target to a format expected by networkx
        position = tuple(points[source_idx].cpu().numpy().astype(np.int32))

        # use a graph search to find all points that are connected to
        # the current position of the agent in the graph
        hierarchy = networkx.shortest_path(self.navigation_graph, position)
        return torch.tensor(list(hierarchy.keys()), **kwargs)

    def sample_navigation_goal(self, observations: Dict[str, Any],
                               navigation_map: str):
        """Utility function that samples a valid destination for the agent
        by performing rejection sampling on the semantic map and rejecting
        points on the map that are not navigable by the agent.

        Arguments:

        observations: Dict[str, Any]
            a dictionary containing observations from the environment,
            returned by calling the self.task.get_observations() function.
        navigation_map: str
            a string that represents a key into the semantic_mappers dict
            and specifies which map to use for navigation.

        Returns:

        point: torch.Tensor
            a tensor representing a point in the coordinate system of the
            world that the robot can traverse and navigate to.

        """

        if "position" not in observations:
            observations["position"] = self.process_position()

        # generate points that are reachable from the agent's position
        points = self.navigable_points(navigation_map,
                                       observations["position"])

        # select the feature map to use for path planning and navigation
        feature_map = self.feature_maps[navigation_map]

        # convert the sampled point from map coordinates to the
        # world coordinate system, with the z coordinate at the origin
        return feature_map.map_to_world(functional.pad(
            points[torch.randint(points.shape[0], (1,))[0]], (0, 1)))

    def shortest_path(self, navigation_map: str,
                      source: torch.Tensor, target: torch.Tensor):
        """Utility function that returns the shortest path between a source
        point to a target point using a navigation graph, where the target
        point need not correspond to a node contained in the graph.

        Arguments:

        source: torch.Tensor
            a tensor representing the navigation starting position of the
            agent in the world coordinate system.
        target: torch.Tensor
            a tensor representing the long-term navigation destination of the
            agent in the world coordinate system.

        Returns:

        shortest_path: torch.Tensor
            a tensor representing the shortest path the agent can take in
            navigable area from the source to the target location.

        """

        # select which map to use for path planning and navigation
        feature_map = self.feature_maps[navigation_map]
        kwargs = dict(dtype=torch.int32, device=feature_map.data.device)

        # calculate the closest source and target points in the support of
        # the nodes in the navigable graph
        points = self.navigable_points(navigation_map, source)

        # compute the index of closest navigable point to the initial pos
        source = feature_map.world_to_map(source[..., :2])
        source_idx = torch.sqrt(torch.square(points - source.view(
            1, 2)).sum(dim=-1).to(torch.float32)).argmin(dim=0)

        # compute the index of closest navigable point to the destination
        # to ensure that a path from the source to destination exists
        target = feature_map.world_to_map(target[..., :2])
        target_idx = torch.sqrt(torch.square(points - target.view(
            1, 2)).sum(dim=-1).to(torch.float32)).argmin(dim=0)

        # convert source and target nodes to a format expected by networkx
        source = tuple(source.cpu().numpy().astype(np.int32))
        target = tuple(points[target_idx].cpu().numpy().astype(np.int32))
        binned_source = tuple(points[source_idx]
                              .cpu().numpy().astype(np.int32))

        # using network x perform a shortest path graph search and
        # return a sequence of map coordinates as shortest path
        shortest_path = networkx.shortest_path(
            self.navigation_graph, source=binned_source, target=target)

        # if the first node in the path is not the agent's real location
        # then add the agent's real location as the new first node
        if any([a != b for a, b in zip(source, binned_source)]):
            shortest_path = [source, *shortest_path]

        return feature_map.map_to_world(functional.pad(
            torch.as_tensor(shortest_path, **kwargs), (0, 1)))

    @staticmethod
    def get_heading(observations: Dict[str, torch.Tensor],
                    goal: torch.Tensor):
        """Utility function that returns an egocentric yaw update for the
        agent, which can be used to orient the agent in the correct direction
        for a deterministic local navigation policy.

        Arguments:

        observations: Dict[str, Any]
            a dictionary containing observations from the environment,
            returned by calling the self.task.get_observations() function.
        goal: torch.Tensor
            a tensor representing the current navigation goal of the agent
            along the shortest path the agent is following.

        Returns:

        egocentric_yaw: torch.Tensor
            a tensor representing an egocentric yaw update for the agent
            given its location and orientation in the environment.

        """

        # calculate the target yaw so the agent is facing the goal
        goal_direction = goal - observations["position"]
        yaw = torch.atan2(goal_direction[1],
                          goal_direction[0]) - observations["yaw"]

        # ensure that the angle returned is in the range [-np.pi, np.pi]
        # and return the distance to that goal
        return (yaw if yaw.abs() <= np.pi else
                -torch.sign(yaw) * (2 * np.pi - yaw.abs()))

    def get_action(self, observations: Dict[str, torch.Tensor],
                   goal: torch.Tensor, navigation_map: str,
                   update_map: Union[str, List[str]] = None,
                   padding: int = 3, depth_slice: slice = None,
                   obstacle_threshold: float = 0.0,
                   update_navigation_graph: bool = True):
        """Helper function that wraps around a deterministic local policy
        and performs navigation to long-term goals in an environment, given
        either an initial path or a long-term navigation goal.

        Arguments:

        observations: Dict[str, Any]
            a dictionary containing observations from the environment,
            returned by calling the self.task.get_observations() function.
        goal: torch.Tensor
            a tensor representing the long-term  navigation goal of the agent
            the agent will attempt to reach by planning.

        navigation_map: str
            a string that represents a key into the semantic_mappers dict
            and specifies which map to use for navigation.
        update_map: Union[str, List[str]]
            a string representing a key into the semantic_mappers dict that
            controls which semantic maps to update.

        padding: int
            an integer representing the number of voxels around occupied
            space in manhattan distance to consider occupied.
        depth_slice: slice
            a depth slice that specifies which voxel to use when
            determining which areas in the map are blocked by obstacles.
        obstacle_threshold: float
            threshold on the one-norm of the feature vector at every channel
            in the feature map to determine if that voxel is occupied.

        update_navigation_graph: bool
            a boolean that controls whether the navigation graph is updated
            with the latest navigability information this step.

        Returns:

        action: int
            the action that will be executed in the environment in order
            for the agent to navigate towards the long-term navigation goal.

        """

        # process the latest vision observations from the environment and
        # update the semantic map by generating a point cloud
        self.process_observations(observations, update_map=update_map)

        # update the navigation graph at a given interval
        if update_navigation_graph:  # which improves runtime efficiency
            self.update_navigation_graph(navigation_map, padding=padding,
                                         depth_slice=depth_slice,
                                         obstacle_threshold=
                                         obstacle_threshold)

        # plan a shortest path from the current position of the agent
        observations["path"] = self.shortest_path(  # to the target position
            navigation_map, observations["position"], goal).cpu()

        # process the observations and determine the yaw heading
        observations["heading"] = (  # to the next navigation target
            torch.zeros_like(observations["yaw"])
            if torch.eq(observations["position"], goal.cpu()).all()
            else self.get_heading(
                observations, (goal.cpu() if
                               observations["path"].shape[0]
                               == 1 else observations["path"][1])))

        # if the agent is facing away from the goal to the left, turn right
        if observations["path"].shape[0] > 1 \
                and np.abs(observations["heading"]) <= np.pi / 4:
            return self.task.action_names().index('move_ahead')

        # if the agent is facing away from the goal to the right, turn left
        elif observations["heading"] > np.pi / 4:
            return self.task.action_names().index('rotate_left')

        # if the agent is facing away from the goal to the left, turn right
        elif observations["heading"] < -np.pi / 4:
            return self.task.action_names().index('rotate_right')

    def failed_action(self, observations: Dict[str, torch.Tensor],
                      action: int, navigation_map: str):
        """Utility function that updates the navigation graph to account for
        failed actions by removing the edge between the agent's current
        position and next position in the navigation graph.

        Arguments:

        observations: Dict[str, Any]
            a dictionary containing observations from the environment,
            returned by calling the self.task.get_observations() function.
        action: int
            the action that will be executed in the environment in order
            for the agent to navigate towards the long-term navigation goal.
        navigation_map: str
            a string that represents a key into the semantic_mappers dict
            and specifies which map to use for navigation.

        """

        # select which map to use for path planning and navigation
        feature_map = self.feature_maps[navigation_map]

        # if the agent walked into a wall, remove the node in the wall
        # if the agent could not rotate in place, remove the current node
        idx = 0 if "rotate" in self.task.action_names()[action] else 1

        # convert the source to the format expected by a networkx.Graph
        node = feature_map.world_to_map(observations["path"][idx][:2])
        node = tuple(node.cpu().numpy().astype(np.int32))

        # this check is necessary because the first point on
        # the path may be located outside the navigation graph at first
        while not self.navigation_graph.has_node(node):

            idx += 1  # jump to the next node in the path

            # convert the source to the format expected by a networkx.Graph
            node = feature_map.world_to_map(observations["path"][idx][:2])
            node = tuple(node.cpu().numpy().astype(np.int32))

        self.navigation_graph.remove_node(node)  # remove and re-plan

    def navigate_to(self, goal: torch.Tensor, navigation_map: str,
                    update_map: Union[str, List[str]] = None,
                    padding: int = 3, depth_slice: slice = None,
                    obstacle_threshold: float = 0.0,
                    update_navigation_graph_interval: int = 20):
        """Helper function that wraps around a deterministic local policy
        and performs navigation to long-term goals in an environment, given
        either an initial path or a long-term navigation goal.

        Arguments:

        observations: Dict[str, Any]
            a dictionary containing observations from the environment,
            returned by calling the self.task.get_observations() function.
        goal: torch.Tensor
            a tensor representing the long-term  navigation goal of the agent
            the agent will attempt to reach by planning.

        navigation_map: str
            a string that represents a key into the semantic_mappers dict
            and specifies which map to use for navigation.
        update_map: Union[str, List[str]]
            a string representing a key into the semantic_mappers dict that
            controls which semantic maps to update.

        padding: int
            an integer representing the number of voxels around occupied
            space in manhattan distance to consider occupied.
        depth_slice: slice
            a depth slice that specifies which voxel to use when
            determining which areas in the map are blocked by obstacles.
        obstacle_threshold: float
            threshold on the one-norm of the feature vector at every channel
            in the feature map to determine if that voxel is occupied.

        update_navigation_graph_interval: int
            an integer that controls how many navigation steps occur
            between subsequent updates to the navigation graph.

        Yields:

        observations: Dict[str, Any]
            a dictionary containing observations from the environment,
            returned by calling the self.task.get_observations() function.

        """

        observations = self.task.get_observations()
        action = self.get_action(observations, goal, navigation_map,
                                 update_map=update_map, padding=padding,
                                 depth_slice=depth_slice,
                                 obstacle_threshold=obstacle_threshold,
                                 update_navigation_graph=True)

        for time_step in count(start=1):
            yield observations  # yield for visualization

            if self.task.is_done() or action is None:
                break  # navigation finished or environment halted

            """Note, the action_success check can be implemented by taking the 
            difference in rgb values of subsequent frames, and if the norm is 
            below a threshold, the images are the same and the action failed. 
            This is less efficient, but leads to the same performance."""

            # attempt to execute the generated action in the simulator
            step_result = self.task.step(action)  # and handle failed actions
            if not step_result.info["action_success"]:
                self.failed_action(observations, action, navigation_map)

            observations = self.task.get_observations()
            action = self.get_action(observations, goal, navigation_map,
                                     update_map=update_map, padding=padding,
                                     depth_slice=depth_slice,
                                     obstacle_threshold=obstacle_threshold,
                                     update_navigation_graph=time_step %
                                     update_navigation_graph_interval == 0)
