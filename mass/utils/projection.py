from itertools import product
import torch
import torch.nn.functional as functional


def spherical_to_cartesian(yaw, elevation):
    """Helper function to convert from a spherical coordinate system
    parameterized by a yaw and elevation to the xyz cartesian coordinate
    with a unit radius, where the z-axis points upwards.

    Arguments:

    yaw: torch.Tensor
        a tensor representing the top-down yaw in radians of the coordinate,
        starting from the positive x-axis and turning counter-clockwise.
    elevation: torch.Tensor
        a tensor representing the elevation in radians of the coordinate,
        about the x-axis, with positive corresponding to upwards tilt.

    Returns:

    point: torch.Tensor
        a tensor corresponding to a point specified by the given yaw and
        elevation, in spherical coordinates.

    """

    # zero elevation and zero yaw points along the positive x-axis
    return torch.stack([torch.cos(yaw) * torch.cos(elevation),
                        torch.sin(yaw) * torch.cos(elevation),
                        torch.sin(elevation)], dim=-1)


def project_camera_rays(image_height, image_width,
                        focal_length_y, focal_length_x,
                        dtype=torch.float32, device='cpu'):
    """Generate a ray for each pixel in an image with a particular map_height
    and map_width by setting up a pinhole camera and sending out a ray from
    the camera to the imaging plane at each pixel location

    Arguments:

    image_height: int
        an integer that described the map_height of the imaging plane in
        pixels and determines the number of rays sampled vertically
    image_width: int
        an integer that described the map_width of the imaging plane in
        pixels and determines the number of rays sampled horizontally
    focal_length_y: float
        the focal length of the pinhole camera, which corresponds to
        the distance to the imaging plane in units of y pixels
    focal_length_x: float
        the focal length of the pinhole camera, which corresponds to
        the distance to the imaging plane in units of x pixels

    Returns:

    rays: torch.Tensor
        a tensor that represents the directions of sampled rays in
        the coordinate system of the camera with shape: [height, width, 3]

    """

    # generate pixel locations for every ray in the imaging plane
    # where the returned shape is: [image_height, image_width]
    kwargs = dict(dtype=dtype, device=device)
    y, x = torch.meshgrid(torch.arange(image_height, **kwargs),
                          torch.arange(image_width, **kwargs), indexing='ij')

    # convert pixel coordinates to the camera coordinate system
    # y is negated to conform to OpenGL convention in computer graphics
    rays_y = (y - 0.5 * float(image_height - 1)) / focal_length_y
    rays_x = (x - 0.5 * float(image_width - 1)) / focal_length_x
    return torch.stack([rays_x, -rays_y, -torch.ones_like(rays_x)], dim=-1)


def transform_rays(rays, eye_vector, up_vector):
    """Given a batch of camera orientations, specified with a viewing
    direction and up vector, convert rays from the camera coordinate
    system to the world coordinate system using a rotation matrix.

    Arguments:

    rays: torch.Tensor
        a batch of rays that have been generated in the coordinate system
        of the camera with shape: [batch, map_height, map_width, 3]
    eye_vector: torch.Tensor
        a batch of viewing directions that are represented as three
        vectors in the world coordinate system with shape: [batch, 3]
    up_vector: torch.Tensor
        a batch of up directions in the imaging plane represented as
        three vectors in the world coordinate system with shape: [batch, 3]

    Returns:

    rays: torch.Tensor
        a batch of rays that have been converted to the coordinate system
        of the world with shape: [batch, map_height, map_width, 3]

    """

    # create a rotation matrix that transforms rays from the camera
    # coordinate system to the world coordinate system
    rotation = torch.stack([torch.cross(
        eye_vector, up_vector), up_vector, -eye_vector], dim=-1)

    # transform the rays using the rotation matrix such that rays project
    # out of the camera in world coordinates in the viewing direction
    return (rays.unsqueeze(-2) *
            rotation.unsqueeze(-3).unsqueeze(-3)).sum(dim=-1)


def bin_rays(bins0, bins1, bins2, origin, rays, depth,
             *features, min_ray_depth=0.0, max_ray_depth=10.0):
    """Given a set of rays and bins that specify the location and size of a
    grid of voxels, return the index of which voxel the end of each ray
    falls into, using a map_depth image to compute this point.

    Arguments:

    bins0: torch.FloatTensor
        a 1D tensor whose elements specify the locations of boundaries of a
        set of voxels along the 0th axis of the coordinate system.
    bins1: torch.FloatTensor
        a 1D tensor whose elements specify the locations of boundaries of a
        set of voxels along the 1st axis of the coordinate system.
    bins2: torch.FloatTensor
        a 1D tensor whose elements specify the locations of boundaries of a
        set of voxels along the 2nd axis of the coordinate system.

    origin: torch.FloatTensor
        the origin of the rays in world coordinates, represented as a batch
        of 3-vectors, shaped like [batch_size, 3].
    rays: torch.FloatTensor
        rays projecting outwards from the origin, ending at a point specified
        by the map_depth, shaped like: [batch_size, height, width, 3].
    map_depth: torch.FloatTensor
        the length of the corresponding ray in world coordinates before
        intersecting a surface, shaped like: [batch_size, height, width, 1].

    features: List[torch.Tensor]
        a list of features for every pixel in the image, such as class
        probabilities shaped like: [batch_size, height, width, num_features]

    min_ray_depth: float
        the minimum distance rays can be to the camera focal point, used to
        handle special cases, such as when the distance is zero.
    max_ray_depth: float
        the maximum distance rays can be to the camera focal point, used to
        handle special cases, such as when the distance is infinity.

    Returns:

    ind0: torch.LongTensor
        voxel ids associated with the point cloud according to their position
        along axis 0, shaped like: [batch_size, num_points].
    ind1: torch.LongTensor
        voxel ids associated with the point cloud according to their position
        along axis 1, shaped like: [batch_size, num_points].
    ind2: torch.LongTensor
        voxel ids associated with the point cloud according to their position
        along axis 2, shaped like: [batch_size, num_points].

    ratio0: torch.FloatTensor
        fraction through each bin corresponding to the points in ind0,
        represented as a float tensor in the range [0, 1].
    ratio1: torch.FloatTensor
        fraction through each bin corresponding to the points in ind1,
        represented as a float tensor in the range [0, 1].
    ratio2: torch.FloatTensor
        fraction through each bin corresponding to the points in ind2,
        represented as a float tensor in the range [0, 1].

    binned_features: List[torch.Tensor]
        a list of features for every pixel in the image, such as class
        probabilities, shaped like: [batch_size, num_points, num_features].

    """

    # bin the point cloud according to which voxel points occupy in space
    # the xyz convention must be known by who is using the function
    rays = origin.unsqueeze(-2).unsqueeze(-2) + rays * depth
    ind0 = torch.bucketize(rays[..., 0].contiguous(), bins0, right=True) - 1
    ind1 = torch.bucketize(rays[..., 1].contiguous(), bins1, right=True) - 1
    ind2 = torch.bucketize(rays[..., 2].contiguous(), bins2, right=True) - 1

    # certain rays will be out of bounds of the map or will have a special
    # map_depth value used to signal invalid points, identify them
    criteria = [torch.logical_and(torch.ge(ind0, 0),
                                  torch.lt(ind0, bins0.size(dim=0) - 1)),
                torch.logical_and(torch.ge(ind1, 0),
                                  torch.lt(ind1, bins1.size(dim=0) - 1)),
                torch.logical_and(torch.ge(ind2, 0),
                                  torch.lt(ind2, bins2.size(dim=0) - 1))]

    # merge each of the criteria into a single mask tensor that will
    # have the shape [batch, image_height, image_width]
    criterion = torch.logical_and(torch.ge(depth.squeeze(-1), min_ray_depth),
                                  torch.le(depth.squeeze(-1), max_ray_depth))
    for next_criterion in criteria:
        criterion = torch.logical_and(criterion, next_criterion)

    # indices where the criterion is true and binned points are valid
    indices = torch.nonzero(criterion, as_tuple=True)

    # select the indices and ray coordinates
    ind0, ind1, ind2, rays = ind0[  # that are within the world bounds
        indices], ind1[indices], ind2[indices], rays[indices]

    # lookup left and right bounds for the voxel each point is binned to
    bounds0 = functional.embedding(torch.stack([
        ind0, ind0 + 1], dim=0), bins0.unsqueeze(-1)).squeeze(-1)
    bounds1 = functional.embedding(torch.stack([
        ind1, ind1 + 1], dim=0), bins1.unsqueeze(-1)).squeeze(-1)
    bounds2 = functional.embedding(torch.stack([
        ind2, ind2 + 1], dim=0), bins2.unsqueeze(-1)).squeeze(-1)

    # evaluate the fraction through the bin each point is
    # which is used when aggregating points in the projective map
    ratio0 = (rays[..., 0] - bounds0[0]) / (bounds0[1] - bounds0[0])
    ratio1 = (rays[..., 1] - bounds1[0]) / (bounds1[1] - bounds1[0])
    ratio2 = (rays[..., 2] - bounds2[0]) / (bounds2[1] - bounds2[0])

    # reverse the indices of the bins along the y axis according
    ind1 = bins1.size(dim=0) - 2 - ind1  # to the open gl convention

    # select a subset of the voxel indices and voxel feature predictions
    # in order to remove points that correspond to invalid voxels
    return (ind0, ind1, ind2, ratio0, 1.0 - ratio1, ratio2,
            *[features_i[indices] for features_i in features])


def update_feature_map(ind0, ind1, ind2, ratio0, ratio1, ratio2,
                       features, feature_map, interpolation_weight=1.0):
    """Scatter add feature vectors associated with a point cloud onto a
    voxel feature map by adding the features to the locations of each voxel
    using the voxel ids returned by the bin_rays function.

    Arguments:

    ind0: torch.LongTensor
        voxel ids associated with the point cloud according to their position
        along axis 0, shaped like: [batch_size, num_points].
    ind1: torch.LongTensor
        voxel ids associated with the point cloud according to their position
        along axis 1, shaped like: [batch_size, num_points].
    ind2: torch.LongTensor
        voxel ids associated with the point cloud according to their position
        along axis 2, shaped like: [batch_size, num_points].

    ratio0: torch.FloatTensor
        fraction through each bin corresponding to the points in ind0,
        represented as a float tensor in the range [0, 1].
    ratio1: torch.FloatTensor
        fraction through each bin corresponding to the points in ind1,
        represented as a float tensor in the range [0, 1].
    ratio2: torch.FloatTensor
        fraction through each bin corresponding to the points in ind2,
        represented as a float tensor in the range [0, 1].

    features: torch.Tensor
        tensor of features that will be added to the feature map using
        torch.scatter_add on the map: [batch_size, num_points, feature_dim].
    feature_map: torch.Tensor
        tensor of features organized as a three dimensional grid of
        voxels with shape: [batch_size, height, width, depth, num_features].
    interpolation_weight: float
        float representing the interpolation weight used when adding
        new features in the feature map weighted by interpolation_weight.

    """

    # get the size of the spatial dimensions of the map, used to
    # infer the batch size of the map if present
    size0, size1, size2, num_features = feature_map.shape[-4:]
    feature_map_flat = feature_map.view(
        -1, size0 * size1 * size2, num_features)

    # compute the indices of upper and lower voxels for each
    ind0_lower = torch.where(ratio0 < 0.5,  # point in the point cloud
                             (ind0 - 1).clamp(min=0), ind0)
    ind1_lower = torch.where(ratio1 < 0.5,
                             (ind1 - 1).clamp(min=0), ind1)
    ind2_lower = torch.where(ratio2 < 0.5,
                             (ind2 - 1).clamp(min=0), ind2)
    ind0_upper = torch.where(ratio0 < 0.5, ind0,
                             (ind0 + 1).clamp(max=size0 - 1))
    ind1_upper = torch.where(ratio1 < 0.5, ind1,
                             (ind1 + 1).clamp(max=size1 - 1))
    ind2_upper = torch.where(ratio2 < 0.5, ind2,
                             (ind2 + 1).clamp(max=size2 - 1))

    # concatenate all combinations of voxel indices along three axes
    indices = torch.cat([  # increasing num points by a factor of eight
        (ind0 * size1 + ind1) * size2 + ind2
        for ind0, ind1, ind2 in product((ind0_lower, ind0_upper),
                                        (ind1_lower, ind1_upper),
                                        (ind2_lower, ind2_upper))], dim=-1)

    # expand the indices tensor with an additional axis so that
    indices = indices.unsqueeze(  # indices has the same shape as features
        -1).expand(*(list(indices.shape) + [num_features]))

    # compute the weights of upper and lower voxels for each
    weight0_lower = torch.where(ratio0 < 0.5,  # point in the point cloud
                                0.5 - ratio0, 1.5 - ratio0)
    weight1_lower = torch.where(ratio1 < 0.5,
                                0.5 - ratio1, 1.5 - ratio1)
    weight2_lower = torch.where(ratio2 < 0.5,
                                0.5 - ratio2, 1.5 - ratio2)
    weight0_upper = torch.where(ratio0 < 0.5,
                                ratio0 + 0.5, ratio0 - 0.5)
    weight1_upper = torch.where(ratio1 < 0.5,
                                ratio1 + 0.5, ratio1 - 0.5)
    weight2_upper = torch.where(ratio2 < 0.5,
                                ratio2 + 0.5, ratio2 - 0.5)

    # concatenate all combinations of weights along each axis
    all_weights = 1e-9 + torch.cat([  # increasing num points by eight
        (w0 * w1 * w2).unsqueeze(-1)
        for w0, w1, w2 in product((weight0_lower, weight0_upper),
                                  (weight1_lower, weight1_upper),
                                  (weight2_lower, weight2_upper))], dim=-2)

    # ensure that all tensors have a batch dimension and if one is not
    if len(indices.shape) < 3:  # present, add a new one with only one unit
        indices = indices.unsqueeze(0)
    if len(features.shape) < 3:
        features = features.unsqueeze(0)
    if len(all_weights.shape) < 3:
        all_weights = all_weights.unsqueeze(0)

    # compute the total amount of probability mass used when interpolating
    # new features for every voxel, used for normalization
    weights_sum = torch.zeros_like(feature_map_flat[..., :1])
    weights_sum.scatter_(-2, indices[..., :1],
                         all_weights, reduce="add")

    # for each new point in the point cloud of features, compute
    # what the interpolated feature in the feature map would be in isolation
    interpolated_features = (((1.0 - interpolation_weight * all_weights) *
                             feature_map_flat.gather(-2, indices)) +
                             (interpolation_weight *
                              all_weights * features.tile(1, 8, 1)))

    # zero the features at all observed voxels in the feature map
    # and assign voxels to the wighted average of new interpolated features
    feature_map_flat.scatter_(-2, indices, 0)
    feature_map_flat.scatter_(-2, indices, interpolated_features *
                              all_weights / weights_sum
                              .gather(-2, indices[..., :1]), reduce="add")
