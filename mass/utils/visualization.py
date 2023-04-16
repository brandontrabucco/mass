import cv2
import torch
import numpy as np
from typing import Union


def get_triangle_vertices(y, x, yaw, size):
    """Helper function for drawing the location and orientation of an agent
    in the top down semantic map by visualizing the agent as a triangle,
    which points in the direction where the agent is facing.

    Arguments:

    y: float
        the y coordinate of the triangle given as a float where 0 is the
        center of the image and y increases to the top.
    x: float
        the x coordinate of the triangle given as a float where 0 is the
        center of the image and x increases to the right.
    yaw: float
        the amount the triangle is rotated in the image plane in radians,
        where positive yaw corresponds to a counterclockwise rotation.
    size: int
        the size of the line in image coordinates connecting the center of
        the triangle to the farthest away vertex.

    Returns:

    vertices: np.ndarray
        a numpy array specifying three points that define a triangle in
        image coordinates at the target location with the target yaw.

    """

    # generate three points that define a triangle in the image
    return np.array([(int(x + size / 1.5 * np.cos(yaw + np.pi * 4 / 3)),
                      int(y + size / 1.5 * np.sin(yaw + np.pi * 4 / 3))),
                     (int(x + size * np.cos(yaw)),
                      int(y + size * np.sin(yaw))),
                     (int(x + size / 1.5 * np.cos(yaw - np.pi * 4 / 3)),
                      int(y + size / 1.5 * np.sin(yaw - np.pi * 4 / 3)))])


def draw_agent(image, y, x, yaw, agent_size=10, agent_stroke=2,
               agent_fill_color=(1.0, 0.0, 0.0),
               agent_outline_color=(0.0, 0.0, 0.0)):
    """Helper function to draw the agent in a semantic map by placing a
    triangle at the agent's location in the semantic map pointing in the
    direction the agent is facing, specified by its yaw.

    Arguments:

    image: np.ndarray
        a numpy array with shape [map_height, map_width, 3] that represents a top
        down view of the semantic map that has been built.
    y: float
        the y coordinate of the triangle given as a float in world
        coordinates that must first be converted to image coordinates.
    x: float
        the x coordinate of the triangle given as a float in world
        coordinates that must first be converted to image coordinates.
    yaw: float
        the amount the triangle is rotated in the image plane in radians,
        where positive yaw corresponds to a counterclockwise rotation.

    agent_size: int
        the size in pixels of the arrow depicting the agent that will be
        drawn on the semantic map in this function call.
    agent_stroke: int
        the thickness in pixels of the outline that will be drawn around
        the triangle depicting the agent in the semantic map.
    agent_fill_color: Tuple[float]
        the color used to fill the inside of the triangle drawn on the
        semantic map, given as a tuple of three floats in [0, 1].
    agent_outline_color: Tuple[float]
        the color used to outline the triangle drawn on the
        semantic map, given as a tuple of three floats in [0, 1].

    """

    # define three triangle vertices with the specified location and yaw
    triangle = get_triangle_vertices(y, x, -yaw, agent_size)

    # fill the triangle and draw an outline with the specified thickness
    cv2.drawContours(image, [triangle], 0, agent_fill_color, -1)
    cv2.drawContours(image, [triangle], 0, agent_outline_color, agent_stroke)
    return image  # technically modified in-place and need not be returned


def visualize_path(feature_map, position: torch.Tensor, yaw: torch.Tensor,
                   path: torch.Tensor, depth_slice: slice = None,
                   map_norm_divisor: Union[float, None] = None,
                   path_thickness: int = 1, agent_size: int = 6,
                   agent_stroke: int = 1, path_color=(1.0, 0.0, 0.0),
                   agent_fill_color=(1.0, 0.0, 0.0),
                   agent_outline_color=(1.0, 1.0, 1.0)):
    """Helper function that renders the area in the semantic map that can
    be traversed by the agent and draws the path taken by the set of
    waypoints passed in, drawing lines between waypoints.

    Arguments:

    feature_map: torch.Tensor
        an tensor from a projection layer that contains a grid of voxels
        representing the world, used to visualize occupancy.

    position: torch.Tensor
        the position of the agent in the world coordinate system, where
        the position will be binned to voxels in a semantic map.
    yaw: torch.Tensor
        a tensor representing the yaw in radians of the coordinate,
        starting from the x-axis and turning counter-clockwise.
    path: torch.Tensor
        a tensor representing the path of waypoints the agent can take in
        navigable area from the source to the target location.

    path_thickness: int
        the number pixels thick of the line drawn on the image connecting
        between subsequent waypoints along the path.
    agent_size: int
        the size in pixels of the arrow depicting the agent that will be
        drawn on the semantic map in this function call.
    agent_stroke: int
        the thickness in pixels of the outline that will be drawn around
        the triangle depicting the agent in the semantic map.

    path_color: Tuple[float]
        the color of the line that depicts the path the agent is planning
        to take, and the color of the endpoint on the map.
    agent_fill_color: Tuple[float]
        the color used to fill the inside of the triangle drawn on the
        semantic map, given as a tuple of three floats in [0, 1].
    agent_outline_color: Tuple[float]
        the color used to outline the triangle drawn on the
        semantic map, given as a tuple of three floats in [0, 1].

    Returns:

    image: np.ndarray
        an image that depicts the current semantic map from the top down
        with the path given by the set of waypoints drawn.

    """

    # ensure that all inputs are tensors and on the same device
    kwargs = dict(dtype=torch.float32, device=feature_map.device)
    yaw = torch.as_tensor(yaw, **kwargs)

    # convert the waypoints location from the world coordinate system to
    # the map coordinate system by binning the points
    path = path[..., :2].cpu().numpy()

    # compute a normalized density at each voxel in the feature
    # map and clamp the result between zero and one for visualization
    density = feature_map.norm(dim=-1, keepdim=True)
    density = (density / (density.amax() if map_norm_divisor is None
                          else map_norm_divisor)).clamp(min=0.0, max=1.0)

    # construct an image from the voxel densities by taking a single
    # index, or the maximum over a range of indices
    image = 1.0 - np.tile((density[:, :, depth_slice]
                           if depth_slice is not None else
                           density).amax(dim=2).cpu().numpy(), (1, 1, 3))

    # loop through pairs of subsequent waypoints along the path and
    # draw a line connecting the first waypoint to the second waypoint
    for i in range(path.shape[0] - 1):
        cv2.line(image, path[i], path[i + 1], path_color, path_thickness)

    # convert waypoints from the world coordinates to map coordinates
    # by binning the points, and snap yaw to intervals of pi / 4
    x, y = position[..., :2].cpu().numpy()
    yaw = np.pi / 4 * np.around(yaw.cpu().numpy() / (np.pi / 4))

    # draw the agent on the image with numpy and opencv and the existing
    # drawing functionality in self.visualize_map() for a blank map
    return draw_agent(image, y, x, yaw, agent_size=agent_size,
                      agent_stroke=agent_stroke,
                      agent_fill_color=agent_fill_color,
                      agent_outline_color=agent_outline_color)
