import numpy as np
import torch
import clip
from PIL import Image
from slam_rcnn.nn.base_projection_layer import BaseProjectionLayer
from typing import Dict, Any, Callable


class CLIPProjectionLayer(BaseProjectionLayer):
    """Create a feature projection layer in PyTorch that maintains a voxel
    grid description of the world world, where each voxel grid cell has
    a feature vector associated with it, typically semantic labels.

    Arguments:

    camera_height: int
        the map_height of the image generated by a pinhole camera onboard the
        agent, corresponding to a map_depth and semantic observation.
    camera_width: int
        the map_width of the image generated by a pinhole camera onboard the
        agent, corresponding to a map_depth and semantic observation.
    vertical_fov: float
        the vertical field of view of the onboard camera, measure in
        radians from the bottom of the viewport to the top.

    map_height: int
        the number of grid cells along the 'map_height' axis of the semantic
        map, as rendered using the top down rendering function.
    map_width: int
        the number of grid cells along the 'map_width' axis of the semantic
        map, as rendered using the top down rendering function.
    map_depth: int
        the number of grid cells that are collapsed along the 'up'
        direction by the top down rendering function.
    feature_size: int
        the number of units in each feature vector associated with every
        grid cell, such as the number of image segmentation categories.

    origin_y: float
        the center of the semantic map along the 'map_height' axis of the
        semantic map as viewed from a top-down render of the map.
    origin_x: float
        the center of the semantic map along the 'map_width' axis of the
        semantic map as viewed from a top-down render of the map.
    origin_z: float
        the center of the semantic map along the 'map_depth' axis of the
        semantic map as viewed from a top-down render of the map.
    grid_resolution: float
        the length of a single side of each voxel in the semantic map in
        units of the world coordinate system, which is typically meters.

    interpolation_weight: float
        float representing the interpolation weight used when adding
        new features in the feature map weighted by interpolation_weight.
    initial_feature_map: torch.Tensor
        tensor representing the initial feature map tensor,
        which will be set to zero if the value not specified by default.

    """

    def __init__(self, camera_height: int = 224, camera_width: int = 224,
                 vertical_fov: float = 90.0, map_height: int = 256,
                 map_width: int = 256, map_depth: int = 64,
                 feature_size: int = 1, dtype: torch.dtype = torch.float32,
                 origin_y: float = 0.0, origin_x: float = 0.0,
                 origin_z: float = 0.0, grid_resolution: float = 0.05,
                 interpolation_weight: float = 0.5,
                 initial_feature_map: torch.Tensor = None,
                 map_downsampling_factor: int = 4,
                 image_downsampling_factor: int = 14,
                 clip_model: torch.nn.Module = None,
                 clip_preprocess: Callable[[Image.Image], torch.Tensor] = None):
        """Create a feature projection layer in PyTorch that maintains a voxel
        grid description of the world world, where each voxel grid cell has
        a feature vector associated with it, typically semantic labels.

        Arguments:

        camera_height: int
            the map_height of the image generated by a pinhole camera onboard the
            agent, corresponding to a map_depth and semantic observation.
        camera_width: int
            the map_width of the image generated by a pinhole camera onboard the
            agent, corresponding to a map_depth and semantic observation.
        vertical_fov: float
            the vertical field of view of the onboard camera, measure in
            radians from the bottom of the viewport to the top.

        map_height: int
            the number of grid cells along the 'map_height' axis of the semantic
            map, as rendered using the top down rendering function.
        map_width: int
            the number of grid cells along the 'map_width' axis of the semantic
            map, as rendered using the top down rendering function.
        map_depth: int
            the number of grid cells that are collapsed along the 'up'
            direction by the top down rendering function.
        feature_size: int
            the number of units in each feature vector associated with every
            grid cell, such as the number of image segmentation categories.

        origin_y: float
            the center of the semantic map along the 'map_height' axis of the
            semantic map as viewed from a top-down render of the map.
        origin_x: float
            the center of the semantic map along the 'map_width' axis of the
            semantic map as viewed from a top-down render of the map.
        origin_z: float
            the center of the semantic map along the 'map_depth' axis of the
            semantic map as viewed from a top-down render of the map.
        grid_resolution: float
            the length of a single side of each voxel in the semantic map in
            units of the world coordinate system, which is typically meters.

        interpolation_weight: float
            float representing the interpolation weight used when adding
            new features in the feature map weighted by interpolation_weight.
        initial_feature_map: torch.Tensor
            tensor representing the initial feature map tensor,
            which will be set to zero if the value not specified by default.

        """

        super(CLIPProjectionLayer, self).__init__(
            camera_height=camera_height // image_downsampling_factor,
            camera_width=camera_width // image_downsampling_factor,
            vertical_fov=vertical_fov,
            map_height=map_height // map_downsampling_factor,
            map_width=map_width // map_downsampling_factor,
            map_depth=map_depth // map_downsampling_factor,
            feature_size=feature_size, dtype=dtype,
            origin_y=origin_y, origin_x=origin_x, origin_z=origin_z,
            grid_resolution=grid_resolution * map_downsampling_factor,
            interpolation_weight=interpolation_weight,
            initial_feature_map=initial_feature_map)

        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.map_downsampling_factor = map_downsampling_factor
        self.image_downsampling_factor = image_downsampling_factor

    def update(self, observation: Dict[str, torch.Tensor]):
        """Update the semantic map given a map_depth image and a feature image
        by projecting the features onto voxels in the semantic map using
        a set of rays emanating from a virtual pinhole camera.

        Arguments:

        observation["position"]: torch.Tensor
            the position of the agent in the world coordinate system, where
            the position will be binned to voxels in a semantic map.
        observation["yaw"]: torch.Tensor
            a tensor representing the yaw in radians of the coordinate,
            starting from the x-axis and turning counter-clockwise.
        observation["elevation"]: torch.Tensor
            a tensor representing the elevation in radians about the x-axis,
            with positive corresponding to upwards tilt.

        observation["map_depth"]: torch.FloatTensor
            the length of the corresponding ray in world coordinates before
            hitting a surface, with shape: [height, width, 1].
        observation["features"]: Any
            a feature vector for every pixel in the imaging plane, to be
            scattered on the map, with shape: [height, width, num_classes].

        """

        # ensure all tensors have the appropriate device and dtype
        position = torch.as_tensor(observation[
            "position"], dtype=torch.float32, device=self.data.device)
        yaw = torch.as_tensor(observation[
            "yaw"], dtype=torch.float32, device=self.data.device)
        elevation = torch.as_tensor(observation[
            "elevation"], dtype=torch.float32, device=self.data.device)
        depth = torch.as_tensor(observation[
            "depth"], dtype=torch.float32, device=self.data.device)

        # prepare the image for processing with the clip model
        # by copying the image to the gpu and normalizing it appropriately
        features = self.clip_preprocess(Image.fromarray(
            np.uint8(255.0 * observation["rgb"]))
            .convert('RGB')).unsqueeze(0).to(self.data.device)

        # update the clip feature map using the latest observations
        super(CLIPProjectionLayer, self).update(  # from the environment
            dict(position=position, yaw=yaw, elevation=elevation,
                 depth=depth[self.image_downsampling_factor // 2::
                             self.image_downsampling_factor,
                             self.image_downsampling_factor // 2::
                             self.image_downsampling_factor],
                 features=self.clip_model.encode_image(
                    features).detach().view(1, 1, self.feature_size)))

        return self  # return self for chaining additional functions

    def top_down(self, depth_slice: slice = slice(0, 32)):
        """Render a top-down view of a map of features organized as a three
        dimensional grid of voxels, by taking the zero vector to be empty
        voxels and rendering the top-most non-empty voxel to a pixel in a grid.

        Arguments:

        depth_slice: slice
            an slice that specifies which map_depth components to use
            when rendering a top down visualization of the feature map.

        Returns:

        feature_image: torch.Tensor
            an image with a feature vector associated with every cell in the
            image corresponding to a visible voxel in the original feature map.

        """

        # downsample the provided slice object because
        if depth_slice is not None:  # this map has a lower resolution
            depth_slice = slice(
                depth_slice.start // self.map_downsampling_factor,
                depth_slice.stop // self.map_downsampling_factor,
                depth_slice.step // self.map_downsampling_factor
                if depth_slice.step is not None else depth_slice.step)

        # call top down super method with down-sampled slice
        return super(CLIPProjectionLayer,
                     self).top_down(depth_slice=depth_slice)

    def visualize(self, obs: Dict[str, Any],
                  depth_slice: slice = slice(4, 32)):
        """Helper function that returns a list of images that are used for
        visualizing the contents of the feature map contained in subclasses,
        such as visualizing object categories, or which voxels are obstacles.

        Arguments:

        obs: Dict[str, dict]
            the current observation, as a dict or Tensor, which can be
            used to visualize the current location of the agent in the scene.
        depth_slice: Union[slice, Dict[str, slice]]
            an slice that specifies which map_depth components to use
            when rendering a top down visualization of the feature map.

        Returns:

        image: np.ndarray
            a list of numpy arrays that visualize the contents of this layer,
             such as an image showing semantic categories.

        """

        clip_text_input = obs["clip_text_input"] \
            if "clip_text_input" in obs else "a fridge in a kitchen"

        with torch.no_grad():
            text_features = self.clip_model.encode_text(
                clip.tokenize(clip_text_input).to(self.data.device))[0]

        base_image = self.data.norm(dim=-1).amax(dim=-1, keepdim=True)
        base_image = (base_image / base_image.amax()).cpu().numpy()

        similarity = (self.data * text_features
                      .view(1, 1, 1, self.feature_size)).sum(dim=-1)

        similarity = similarity.view(-1)\
            .softmax(dim=0).view(*self.data.shape[:-1])

        # downsample the provided slice object because
        if depth_slice is not None:  # this map has a lower resolution
            depth_slice = slice(
                depth_slice.start // self.map_downsampling_factor,
                depth_slice.stop // self.map_downsampling_factor,
                depth_slice.step // self.map_downsampling_factor
                if depth_slice.step is not None else depth_slice.step)

        similarity = (similarity[..., depth_slice]
                      if depth_slice is not None else
                      similarity).amax(dim=-1, keepdim=True)

        similarity = (similarity / (similarity.amax() + 1e-9)).cpu().numpy()

        return (base_image * (1.0 - similarity) +
                np.array([[[1.0, 0.0, 0.0]]]) * similarity)
