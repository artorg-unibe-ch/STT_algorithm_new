"""
Created on 20210226
@author: Yvan Gugler
{

}
"""
# import
from Pipeline_Modules.lib import *

# define coordinate axes
v_x = np.array([1, 0, 0])
v_y = np.array([0, 1, 0])
v_z = np.array([0, 0, 1])

# identity matrix
identity_matrix = np.eye(3, dtype=int)

# zero vector
zero_vec = np.zeros(3, dtype=int)

# function definitions
def read_image_information(filename):
    """

    :param filename:
    :return:
    """
    reader = sitk.ImageFileReader()  # construct an image reader instance
    reader.SetFileName(filename)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()  # read the image information

    try:
        image_info_dict = {"Ndims": reader.GetDimension(),
                           "Size": reader.GetSize(),
                           "Origin": reader.GetOrigin(),
                           "Spacing": reader.GetSpacing(),
                           "Transformation Matrix": reader.GetDirection(),
                           "PixelIDValue": reader.GetPixelIDValue(),
                           }
    except:
        image_info_dict = {}

    metadata_dict = {}
    try:
        for key in reader.GetMetaDataKeys():  # dictionary with metadata
            value = reader.GetMetaData(key)
            metadata_dict[key] = value
    except:
        print("Error reading Metadata")
    return metadata_dict, image_info_dict


def read_mhd(filename):
    """

    :param filename:
    :return:
    """
    reader = sitk.ImageFileReader()  # construct image reader instance
    reader.SetImageIO("MetaImageIO")  # set image IO to MetaImage
    reader.SetFileName(filename)
    img_sitk = reader.Execute()  # read the file into an sitk image object

    return img_sitk


def read_nifti(filename):
    """

    :param filename:
    :return:
    """
    reader = sitk.ImageFileReader()  # construct image reader instance
    reader.SetImageIO("NiftiImageIO")  # set image IO to NiftiImage
    reader.SetFileName(filename)
    img_sitk = reader.Execute()  # read the file into an sitk image object

    return img_sitk


def read_vtk(filename):
    """

    :param filename:
    :return:
    """
    reader = sitk.ImageFileReader()  # construct image reader instance
    reader.SetImageIO("VTKImageIO")  # set image IO to VTKImageIO
    reader.SetFileName(filename)
    img_sitk = reader.Execute()  # read the file into an sitk image object

    return img_sitk


def sitk2np(img_sitk, pixel_type=np.float32):
    """

    :param img_sitk:
    :param pixel_type:
    :return:
    """
    img_np = sitk.GetArrayFromImage(img_sitk)  # transform to numpy array
    img_np = img_np.transpose(2, 1, 0)  # switch ordering of indexes: SITK is in x, y, z / np switches to z, y, x
    img_np = img_np.astype(pixel_type)
    return img_np


def np2sitk(img_np, origin=(0, 0, 0), spacing=(1, 1, 1), direction=(1, 0, 0, 0, 1, 0, 0, 0, 1), output_pixel_type = sitk.sitkInt64):
    """
    :param img_np:
    :param origin:
    :param spacing:
    :param direction:
    :return:
    """
    img_np = img_np.transpose(2, 1, 0)  # switch ordering of indexes: SITK is in x,y,z / np switches to z, y, x
    img_sitk = sitk.GetImageFromArray(img_np)  # transform to sitk image and set properties
    img_sitk.SetOrigin(origin)
    img_sitk.SetDirection(direction)
    img_sitk.SetSpacing(spacing)
    img_sitk = sitk.Cast(img_sitk, output_pixel_type)
    return img_sitk


def bounding_box_np(img):
    # TODO ask yvan why he uses r and c instead of x and y or width and height..
    """
    Shrinks the image to its bounding box.
    :param img:
    :return:
    """
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))
    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    return img[(rmin):(rmax+1), (cmin):(cmax+1), (zmin):(zmax+1)]


def bounding_box_start_end_np(img):
    """
    :param img:
    :return:
    """
    r = np.any(img, axis=(1, 2))  # row
    c = np.any(img, axis=(0, 2))  # column
    z = np.any(img, axis=(0, 1))  # z-axis
    rmin, rmax = np.where(r)[0][[0, -1]] # creates array with indexes of where TRUE, then chooses first element of
    # array (only has one), thatof first and last element
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    return rmin, rmax+1, cmin, cmax+1, zmin, zmax+1


def add_layers_nb_np(img_np, add_layer):
    """
    :param img_np:
    :param add_layer:
    :return:
    """
    if np.array(add_layer).shape == ():
        add_layer = [add_layer] * 3
    dim_x = int(img_np.shape[0] + 2 * add_layer[0])
    dim_y = int(img_np.shape[1] + 2 * add_layer[1])
    dim_z = int(img_np.shape[2] + 2 * add_layer[2])
    dimension_moving = img_np.shape
    img_bigger_np = np.zeros((dim_x, dim_y, dim_z))
    img_bigger_np[
    int(dim_x / 2 - dimension_moving[0] / 2):int(dim_x / 2 - dimension_moving[0] / 2) + dimension_moving[0],
    int(dim_y / 2 - dimension_moving[1] / 2):int(dim_y / 2 - dimension_moving[1] / 2) + dimension_moving[1],
    int(dim_z / 2 - dimension_moving[2] / 2):int(dim_z / 2 - dimension_moving[2] / 2) + dimension_moving[2]] = img_np
    del img_np
    return img_bigger_np


def normi(vector):
    """

    :param vector:
    :return:
    """
    norm_unit = np.linalg.norm(vector)
    return vector / norm_unit


def linear_regression_3d(data):
    """

    :param data:
    :return:
    """
    datamean = data.mean(axis=0)
    uu, dd, vv = np.linalg.svd(data - datamean)
    return normi(vv[0]), datamean


def disc_perpendicular_to_vector(point, vector, radius=50, nb_points_disc=400):
    """
    Creates a disc perpendicular to a vector.
    :param point:
    :param vector:
    :param radius:
    :param nb_points_disc:
    :return:
    """
    x = 0.; y = 0.; z = 0.
    vector = normi(vector)
    d = -np.dot(point, vector)
    if not vector.any():
        print('Error: Direction vector is zero!')
    elif vector[1] == 0. and vector[2] == 0.:
        x = -d / vector[0]; y = 1.; z = 1.
    elif vector[0] == 0. and vector[2] == 0.:
        x = 1.; y = -d/vector[1]; z = 1.
    elif vector[0] == 0. and vector[1] == 0.:
        x = 1.; y = 1.; z = -d/vector[2]
    elif vector[0] == 0.:
        x = 1.; y = 1.; z = (-d-vector[1])/vector[2]
    elif vector[1] == 0.:
        x = 1.; y = 1.; z = (-d-vector[0])/vector[2]
    elif vector[2] == 0.:
        x = 1.; y = (-d - vector[0]) / vector[1]; z = 1.
    else:
        z = (-vector[0] - vector[1] - d) * 1. / vector[2]

    vx = normi(np.array([x, y, z]) - point)
    vy = normi(np.cross(vx, vector))
    t = np.linspace(0, 2 * np.pi, nb_points_disc)
    circle_list = []
    for r in range(radius):
        circle_list.append(
            np.rint(point + (vx.reshape(3, 1) * (r * np.cos(t))).T + (-vy.reshape(3, 1) * (r * np.sin(t))).T))
    return np.array(circle_list).reshape(np.array(circle_list).shape[0] * np.array(circle_list).shape[1], 3)


def disc_collision_point(image_np, vector, starting_point, radius=60, last_disc=False, nb_points_disc=400):
    """
    Creation of a disc perpendicular to direction vector.
    Move the disc along direction vector till it hits the mask.
    The coordinates of the point(s) where the disc contacts the mask are returned.
    :param image_np: numpy array of masked image
    :param vector: direction vector
    :param starting_point:
    :param radius:
    :param last_disc:
    :param nb_points_disc:
    :return:
    """
    vector = normi(vector)
    same_value = []
    i = 0
    while len(same_value) == 0:
        disc = disc_perpendicular_to_vector(starting_point + vector * i, vector, radius=radius,
                                            nb_points_disc=nb_points_disc)

        disc = disc[np.all(disc > [0, 0, 0], axis=1)]
        disc = disc[np.all(disc < image_np.shape, axis=1)]
        same_value = disc[image_np[disc[:, 0].astype(int), disc[:, 1].astype(
            int), disc[:, 2].astype(int)] == 1, :]
        i += 1
    if last_disc == True:
        return np.mean(same_value, axis=0), disc
    else:
        return np.mean(same_value, axis=0)


def R_from_vector(v1, v2):
    """
    Computes rotation matrix between two vectors:
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d

    :param v1:
    :param v2:
    :return:
    """
    v1 = normi(v1)  # normalize the vectors / was missing before
    v2 = normi(v2)
    v = np.cross(v1, v2)
    v_ = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])

    # angle = angle_between_vectors(v1, v2)
    s = np.linalg.norm(v)
    c = np.dot(v1, v2)
    R = np.eye(3) + v_ + np.dot(v_, v_) * (1 - c) / s ** 2
    return R


def closest_point_between_vector(v1, v2, start_v1, start_v2):
    """

    :param v1:
    :param v2:
    :param start_v1:
    :param start_v2:
    :return:
    """
    # morroworks.com/Content/Docs/Rays%20closest%20point.pdf
    print('    ... closest_point_between_vector')
    c = start_v2 - start_v1
    D = start_v1 + v1 * ((-np.dot(v1, v2) * np.dot(v2, c) + np.dot(v1, c) * np.dot(v2, v2)) / (
            np.dot(v1, v1) * np.dot(v2, v2) - np.dot(v1, v2) * np.dot(v1, v2)))
    E = start_v2 + v2 * ((np.dot(v1, v2) * np.dot(v1, c) - np.dot(v2, c) * np.dot(v1, v1)) / (
            np.dot(v1, v1) * np.dot(v2, v2) - np.dot(v1, v2) * np.dot(v1, v2)))
    return (D + E) / 2.


def closest_distance_between_vector_and_point(vector, vector_start, point):
    """
    http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    :param vector:
    :param vector_start:
    :param point:
    :return:
    """
    return (np.linalg.norm(np.cross(point - vector_start, point - (vector_start + vector * 100)))) / (
        np.linalg.norm((vector_start + vector * 100) - vector_start))


def fit_sphere(coord_list):
    """
    Fit a sphere to point cloud.
    :param coord_list:
    :return:
    """
    # Assemble the A matrix
    spX = np.array(coord_list[:, 0])
    spY = np.array(coord_list[:, 1])
    spZ = np.array(coord_list[:, 2])
    A = np.zeros((len(spX), 4))
    A[:, 0] = spX * 2
    A[:, 1] = spY * 2
    A[:, 2] = spZ * 2
    A[:, 3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX), 1))
    f[:, 0] = (spX * spX) + (spY * spY) + (spZ * spZ)
    C, residules, rank, singval = np.linalg.lstsq(A, f)

    #   solve for the radius
    t = (C[0] * C[0]) + (C[1] * C[1]) + (C[2] * C[2]) + C[3]
    radius = np.sqrt(t)

    return radius, np.array([C[0], C[1], C[2]]).flatten()


def sitk_transform_from_matrix(transformation_matrix):
    """
    Compute sitk transformation object from transformation matrix. (Works only for rigid body movements).
    :param transformation_matrix: 4x4 numpy nd.array that defines a homogenous transformation
    :return:
    """
    rotation = transformation_matrix[0:3, 0:3]
    rotation_listshape = [rotation[0, 0], rotation[0, 1], rotation[0, 2], rotation[1, 0],
                          rotation[1, 1], rotation[1, 2], rotation[2, 0], rotation[2, 1],
                          rotation[2, 2]]

    sitk_transform = sitk.VersorRigid3DTransform()  # Define the transform in SITK
    sitk_transform.SetMatrix(rotation_listshape)
    sitk_transform.SetTranslation(([transformation_matrix[0, 3], transformation_matrix[1, 3],
                                    transformation_matrix[2, 3]]))
    sitk_inverse_transform = sitk_transform.GetInverse()  # inverse of the transform

    return sitk_transform, sitk_inverse_transform


def transform_mask_to_experimental_position(img_mask_np, transformation_matrix, Spacing,
                                            interpolator=sitk.sitkNearestNeighbor):
    """
    Transform the mask using the transformation matrix.
    :param img_mask_np:
    :param transformation_matrix:
    :param interpolator:
    :return:
    """
    img_mask_sitk = np2sitk(img_mask_np, spacing=Spacing)  # sitk image from np array
    sitk_transform, sitk_inverse_transform = sitk_transform_from_matrix(transformation_matrix)  # transform from matrix
    img_mask_transformed = sitk.Resample(img_mask_sitk, sitk_inverse_transform, interpolator)
    img_mask_transformed = sitk2np(img_mask_transformed).astype('int')
    return img_mask_transformed


def transform_point(point, transformation_matrix):
    """
    Use the homogenous transformation matrix to transform a single point or vector from the original to the experimental
    position.
    :param point: np.array (3x1)
    :param transformation_matrix: 4x4 np.array -> represents a homogenous transformation matrix
    :return: transformed point or vector
    """
    point = np.concatenate((point, np.array([1])))
    transformed_point = np.dot(transformation_matrix, point)
    return transformed_point[0:3]


def resampling(img_np, interpolator=sitk.sitkNearestNeighbor, input_spacing=(1, 1, 1), voxel_size=3,
               output_direction=None, output_origin=None, default_pixel_value=0, output_pixel_type=None):
    """
    :param img_np: image as np array
    :param interpolator: interpolation method for resampling (SITK)
    :param input_spacing: 3-tuple containing the input spacing
    :param voxel_size:
    :param output_direction:
    :param output_origin:
    :param default_pixel_value:
    :param output_pixel_type:
    :return:
    """
    img_sitk = np2sitk(img_np, spacing=input_spacing)
    output_spacing = (1, 1, 1)
    output_spacing = tuple([voxel_size * x for x in output_spacing])

    grid_size = [img_sitk.GetWidth(), img_sitk.GetHeight(), img_sitk.GetDepth()]
    output_size = [0, 0, 0]

    for dim in range(0, len(input_spacing)):
        output_size[dim] = int(input_spacing[dim] / output_spacing[dim] * grid_size[dim])

    if output_direction is None:
        output_direction = img_sitk.GetDirection()

    if output_origin is None:
        output_origin = img_sitk.GetOrigin()

    if output_pixel_type is None:
        output_pixel_type = sitk.sitkUInt8

    identity_transform = sitk.TranslationTransform(3)
    identity_transform.SetIdentity()

    img_sitk_coarse = sitk.Resample(img_sitk, output_size, identity_transform, interpolator,
                                    output_origin, output_spacing, output_direction, default_pixel_value,
                                    output_pixel_type)
    img_np_coarse = sitk2np(img_sitk_coarse)
    return img_np_coarse


def sphere_radius_from_volume(vol):
    """
    Compute the radius of a sphere given its volume.
    :param vol:
    :return:
    """
    radius = (3/(4*np.pi) * vol)**(1/3)
    return radius


def create_sphere(shape, radius, position):
    """
    Create a mask of a "generalized sphere/ellipsoid".
    :param shape:
    :param radius: 3-tuple of int or float
    :param position: 3-tuple of int or float
    :return: mask of sphere at position with radius in an array of shape
    """
    # https://stackoverflow.com/questions/46626267/how-to-generate-a-sphere-in-3d-numpy-array/46626448
    # assume shape and position are both a 3-tuple of int or float
    # the units are pixels / voxels (px for short)
    # radius is a int or float in px
    # semisizes = (radius,) * 3
    semisizes = radius

    # generate the grid for the support points
    # centered at the position indicated by position
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]
    # calculate the distance of all points from `position` center
    # scaled by the radius
    arr = np.zeros(shape, dtype=float)
    for x_i, semisize in zip(position, semisizes):
        # this can be generalized for exponent != 2
        # in which case `(x_i / semisize)`
        # would become `np.abs(x_i / semisize)`
        arr += (x_i / semisize) ** 2

    # the inner part of the sphere will have distance below 1
    return arr <= 1.0


def create_sphere_2(shape, radius, position):
    """
    Create a mask of a "generalized sphere/ellipsoid".
    :param shape:
    :param radius: 3-tuple of int or float
    :param position: 3-tuple of int or float
    :return: mask of sphere at position with radius in an array of shape
    """
    sphere_mask = np.zeros(shape, dtype=bool)
    xmin = int(position[0] - (radius + 2))
    xmax = int(position[0] + (radius + 2))
    ymin = int(position[1] - (radius + 2))
    ymax = int(position[1] + (radius + 2))
    zmin = int(position[2] - (radius + 2))
    zmax = int(position[2] + (radius + 2))
    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            for z in range(zmin, zmax):
                vector = np.array([x, y, z]) - position
                if np.linalg.norm(vector) < radius:
                    sphere_mask[x, y, z] = True
    return sphere_mask


def close_holes(binary_img_np):
    """

    :param binary_img_np:
    :return:
    """
    boolean_mask = binary_img_np.astype(bool)
    inv_boolean_mask = ~boolean_mask

    labels_out, N = cc3d.connected_components(inv_boolean_mask.astype('int'), connectivity=6, return_N=True)
    distribution = np.bincount(np.ndarray.flatten(labels_out))

    max_label = np.argmax(distribution[1:])
    holes_mask = np.ones(np.shape(labels_out))
    holes_mask[labels_out == 0] = 0
    holes_mask[labels_out == max_label + 1] = 0

    return (binary_img_np + holes_mask)


def remove_badly_connected_voxels(binary_img_np, con=6):
    """
    Removes small unconnected clusters as well as voxels from an image that are badly connected to the adjacent voxels
    of the binary mask,
    Voxels that are connected only over edges or vertices are considered to be badly connected.
    :param binary_img_np:
    :param con:
    :return:
    """
    labelled_image, N = cc3d.connected_components(binary_img_np, connectivity=con,  return_N=True)
    distribution = np.bincount(np.ndarray.flatten(labelled_image))

    max_label = np.argmax(distribution[1:]) + 1
    binary_mask = binary_img_np * (labelled_image == max_label)
    removed_voxels = binary_img_np - binary_mask

    return binary_mask, removed_voxels


def compute_inertia_tensor(img_mask_np, voxel_size, reference_point=(0, 0, 0)):
    """
    Computation of inertia tensor of a masked, voxelized geometry. The definition of the inertia tensor can be found
    in the lecture Continuum Mechanics, spring semester 2020, at University of Bern - Lecture 6 - slide 220.
    :param img_mask_np:
    :param voxel_size:
    :param reference_point:
    :return:
    """
    dimensions = np.shape(img_mask_np)
    inertia_tensor = np.zeros([3, 3])
    identity = np.identity(3)
    reference_point = np.array(reference_point)
    for x in range(0, dimensions[0]):
        for y in range(0, dimensions[1]):
            for z in range(0, dimensions[2]):
                if img_mask_np[x, y, z] != 0:
                    distance_vector = np.array([x, y, z]) - reference_point
                    density = img_mask_np[x, y, z]
                    inertia_tensor += ((np.dot(distance_vector, distance_vector) * identity -
                                        np.outer(distance_vector, distance_vector)) * density * voxel_size**3)
    return inertia_tensor


def homogenous_transformation_matrix(rotation_matrix, translation):
    """
    Build the four by four matrix of a homogenous (Rotation followed by translation) transformation in Euclidean R3 space.
    Rigid body rotation followed by a translation.
    :param rotation_matrix:
    :param translation:
    :return:
    """
    transformation_matrix = np.zeros([4, 4])
    transformation_matrix[0:3, 0:3] = rotation_matrix
    transformation_matrix[0:3, 3] = translation
    transformation_matrix[3, 3] = 1
    return transformation_matrix


def rotation_arbitrary_axis(axis, angle):
    """
    Rotation around an arbitrary axis through the origin in euclidean R3 space, computed by Rodrigues' rotation formula.
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    :param axis:
    :param angle:
    :return:
    """
    vector = normi(axis)
    k_matrix = np.array([[0, -vector[2], vector[1]], [vector[2], 0, -vector[0]], [-vector[1], vector[0], 0]])
    identity = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    rotation_matrix = identity + np.sin(angle) * k_matrix + (1-np.cos(angle))*np.dot(k_matrix, k_matrix)
    return rotation_matrix


def deg2rad(angle_deg):
    """
    Convert angle in degrees to angle in radians.
    :param angle_deg:
    :return:
    """
    return angle_deg/180*np.pi


def rad2deg(angle_rad):
    """
    Convert angle in radians to angle in degrees.
    :param angle_rad:
    :return:
    """
    return angle_rad/np.pi*180


def distance_between_points(point1, point2):
    """
    Distance between two points.
    :param point1:
    :param point2:
    :return:
    """
    difference_vector = point2 - point1
    return np.linalg.norm(difference_vector)


def xprimary():
    """
    Function used for the definition of the axis direction of the cylinder in generate_cylinder_mask()
    :return:
    """
    primary_direction = np.array([1, 0, 0]); primary_ind = 0
    secondary_direction = np.array([0, 1, 0]); secondary_ind = 1
    tertiary_direction = np.array([0, 0, 1]); tertiary_ind = 2
    return primary_direction, secondary_direction, tertiary_direction, primary_ind, secondary_ind, tertiary_ind


def yprimary():
    """
    Function used for the definition of the axis direction of the cylinder in generate_cylinder_mask()
    :return:
    """
    primary_direction = np.array([0, 1, 0]); primary_ind = 1
    secondary_direction = np.array([0, 0, 1]); secondary_ind = 2
    tertiary_direction = np.array([1, 0, 0]); tertiary_ind = 0
    return primary_direction, secondary_direction, tertiary_direction, primary_ind, secondary_ind, tertiary_ind


def zprimary():
    """
    Function used for the definition of the axis direction of the cylinder in generate_cylinder_mask()
    :return:
    """
    primary_direction = np.array([0, 0, 1]); primary_ind = 2
    secondary_direction = np.array([1, 0, 0]); secondary_ind = 0
    tertiary_direction = np.array([0, 1, 0]); tertiary_ind = 1
    return primary_direction, secondary_direction, tertiary_direction, primary_ind, secondary_ind, tertiary_ind


def xprimary_neg():
    """
    Function used for the definition of the axis direction of the cylinder in generate_cylinder_mask()
    :return:
    """
    primary_direction = np.array([-1, 0, 0]); primary_ind = 0
    secondary_direction = np.array([0, -1, 0]); secondary_ind = 1
    tertiary_direction = np.array([0, 0, 1]); tertiary_ind = 2
    return primary_direction, secondary_direction, tertiary_direction, primary_ind, secondary_ind, tertiary_ind


def yprimary_neg():
    """
    Function used for the definition of the axis direction of the cylinder in generate_cylinder_mask()
    :return:
    """
    primary_direction = np.array([0, -1, 0]); primary_ind = 1
    secondary_direction = np.array([0, 0, -1]); secondary_ind = 2
    tertiary_direction = np.array([1, 0, 0]); tertiary_ind = 0
    return primary_direction, secondary_direction, tertiary_direction, primary_ind, secondary_ind, tertiary_ind


def zprimary_neg():
    """
    Function used for the definition of the axis direction of the cylinder in generate_cylinder_mask()
    :return:
    """
    primary_direction = np.array([0, 0, -1]); primary_ind = 2
    secondary_direction = np.array([-1, 0, 0]); secondary_ind = 0
    tertiary_direction = np.array([0, 1, 0]); tertiary_ind = 1
    return primary_direction, secondary_direction, tertiary_direction, primary_ind, secondary_ind, tertiary_ind


def generate_cylinder_mask(img_dimensions, base_point, direction, radius, height, voxel_size,
                           foreground_pixel_value=1, background_pixel_value=0):
    """
    Create a Cylinder along a coordinate axis with positive or negative direction given its height, radius and "base
    point".
    :param img_dimensions: outer dimensions of the masked image.
    :param base_point: Base_point
    :param direction: Direction to build cylinder from base_point
    :param radius: Cylinder radius in mm
    :param height: Cylinder height in mm
    :param voxel_size: Isotropic voxel size in mm
    :param foreground_pixel_value:
    :param background_pixel_value:
    :return: cylinder_mask_np:
    """
    cylinder_mask_np = background_pixel_value * np.ones(img_dimensions)  # empty mask
    switch_direction = {  # direction of the cylinder axis from base point
        'x': xprimary(),
        'y': yprimary(),
        'z': zprimary(),
        'x_neg': xprimary_neg(),
        'y_neg': yprimary_neg(),
        'z_neg': zprimary_neg()
    }

    primary_direction, secondary_direction, tertiary_direction, primary_ind, secondary_ind, tertiary_ind =\
        switch_direction.get(direction)  # define the direction using the switch statement above

    voxels_in_height = int(height/voxel_size)  # number of voxels in the given height
    voxels_in_radius = int(radius/voxel_size)  # number of voxels in radius
    voxels_height_width = voxels_in_radius + 5

    for vH in range(0, voxels_in_height):  # for each layer of voxels define a circle and look for the voxels inside
        current_circle_center = base_point + vH * primary_direction
        for vD in range(base_point[secondary_ind]-voxels_height_width, base_point[secondary_ind]+voxels_height_width):
            for vL in range(base_point[tertiary_ind]-voxels_height_width, base_point[tertiary_ind]+voxels_height_width):
                position = np.zeros(3)
                position[primary_ind] = current_circle_center[primary_ind]
                position[secondary_ind] = vD
                position[tertiary_ind] = vL
                position = position.astype('int')

                distance = voxel_size * distance_between_points(current_circle_center, position)
                if distance <= radius:  # check if current voxel is inside the circle
                    cylinder_mask_np[position[0], position[1], position[2]] = foreground_pixel_value

    return cylinder_mask_np


def write_MHD(img, filename):
    """

    :param img:
    :param filename:
    :return:
    """
    print('    ... write_MHD')
    if type(img) != np.ndarray:
        pass
    else:
        img = sitk.GetImageFromArray(img.transpose(2, 1, 0))
    minmax = sitk.MinimumMaximumImageFilter()
    minmax.Execute(img)
    if minmax.GetMaximum() == 1:
        """
        sitk.WriteImage(sitk.Cast(img, sitk.sitkInt8),
                        filename.split(filename.split('/')[-1].split('.')[0])[0] + filename.split('/')[-1].split('.')[
                            0] + '.mhd')
        """
        sitk.WriteImage(sitk.Cast(img, sitk.sitkInt8), filename)
    else:
        """
        sitk.WriteImage(sitk.Cast(img, sitk.sitkInt16),
                        filename.split(filename.split('/')[-1].split('.')[0])[0] + filename.split('/')[-1].split('.')[
                            0] + '.mhd')
        """
        sitk.WriteImage(sitk.Cast(img, sitk.sitkInt16), filename)


def mayavi_plot_mesh(image, factor=1, opacity=1, fig=None, color=(0.69163319, 0.58612982, 0.42201347), title=''):
    """

    :param image:
    :param factor:
    :param opacity:
    :param fig:
    :param color:
    :param title:
    :return:
    """
    if type(image) != np.ndarray:
        image = sitk.GetArrayFromImage(image).transpose(2, 1, 0)
    else:
        try:
            if len(image.shape) == 2:
                if image.shape[1] == 3:
                    # coordinate detected...
                    image = coordinate_to_array_3d(image)
        except:
            if len(np.array(image).shape) == 2:
                if np.array(image).shape[1] == 3:
                    # coordinate detected...
                    image = coordinate_to_array_3d(np.array(image))
    if fig is None:
        fig = mlab.figure(bgcolor=(1, 1, 1))
    else:
        pass
    verts, faces, normals, values = measure.marching_cubes(image[::factor, ::factor, ::factor], 0)
    mlab.triangular_mesh(verts[:, 0], verts[:, 1], verts[:, 2], faces, opacity=opacity, color=color)
    mlab.title(title)
    return fig


def coordinate_to_array_3d(coordinate, dim='max_of_coordinate', value=1):
    """

    :param coordinate:
    :param dim:
    :param value:
    :return:
    """
    if type(coordinate) != np.ndarray:
        coordinate = np.array(coordinate)
    print('    ... coordinate_to_array_3d')
    mini = np.min(coordinate)
    if mini < 0:
        print('        ... negative coordinates detected, -> shifted (np.abs) %s' % (mini))
        if np.min(coordinate[:, 0]) < 0:
            coordinate[:, 0] += np.abs(np.min(coordinate[:, 0]))
        if np.min(coordinate[:, 1]) < 0:
            coordinate[:, 1] += np.abs(np.min(coordinate[:, 1]))
        if np.min(coordinate[:, 2]) < 0:
            coordinate[:, 2] += np.abs(np.min(coordinate[:, 2]))
    if dim == 'max_of_coordinate':
        dim = np.array(
            [int(np.max(coordinate[:, 0]) + 1), int(np.max(coordinate[:, 1]) + 1), int(np.max(coordinate[:, 2]) + 1)])
    array_3d = np.zeros((dim[0], dim[1], dim[2]))
    array_3d[coordinate[:, 0].astype(int), coordinate[:, 1].astype(int), coordinate[:, 2].astype(int)] = value
    return array_3d


def mayavi_add_sphere(center, fig=None, radius=3, color=(1, 0, 0)):
    """

    :param center:
    :param fig:
    :param radius:
    :param color:
    :return:
    """
    from tvtk.api import tvtk
    from tvtk.common import configure_input_data
    if fig is None:
        fig = mlab.figure(bgcolor=(1, 1, 1))
    else:
        pass
    sphere = tvtk.SphereSource(center=(center[0], center[1], center[2]), radius=radius)
    sphere_mapper = tvtk.PolyDataMapper()
    configure_input_data(sphere_mapper, sphere.output)
    sphere.update()
    p = tvtk.Property(opacity=0.99, color=color)
    sphere_actor = tvtk.Actor(mapper=sphere_mapper, property=p)
    fig.scene.add_actor(sphere_actor)


def mayavi_add_arrow(vector, point, length=10, fig=None, scaling_factor=1, color=(0, 0, 0)):
    """

    :param vector:
    :param point:
    :param length:
    :param fig:
    :param scaling_factor:
    :param color:
    :return:
    """
    if fig is None:
        fig = mlab.figure()
    else:
        pass
    from tvtk.tools import visual
    visual.set_viewer(fig)
    x1, y1, z1 = point
    x2, y2, z2 = point + np.array(vector) * length
    ar1 = visual.arrow(x=x1, y=y1, z=z1)
    ar1.length_cone = 0.3 * scaling_factor
    arrow_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    ar1.actor.scale = [arrow_length, arrow_length, arrow_length]
    ar1.radius_shaft = 0.01 * scaling_factor
    ar1.radius_cone = ar1.radius_shaft * 3
    ar1.length_cone = ar1.radius_shaft * 12
    ar1.pos = ar1.pos / arrow_length
    ar1.axis = [x2 - x1, y2 - y1, z2 - z1]
    ar1.color = color


def save_mayavi_to_png(filename, fig):
    """

    :param filename:
    :return:
    """
    mlab.savefig(filename, size=(1000, 1000), figure=fig)


def mayavi_close_figure(figure):
    """

    Parameters
    ----------
    figure :

    Returns
    -------

    """
    mlab.close(figure)


def line_line_intersection(start_point1, vector1, start_point2, vector2):
    """
    Computes the intersection between two lines in space if it exists.
    https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection -> more than two lines -> in three dimensions
    Parameters
    ----------
    start_point1 :
    vector1 :
    start_point2 :
    vector2 :

    Returns
    -------
    """
    is_intersecting = False  # bool: intersecting or not
    w = 0   # variable to hold the intersection point

    # construct four planes with a normal vector and a point
    # each line is considered as the intersection of two planes
    nv1 = np.array([vector1[1], -vector1[0], 0])
    nv2 = np.array([0, vector1[2], -vector1[1]])
    nv3 = np.array([vector2[1], -vector2[0], 0])
    nv4 = np.array([0, vector2[2], -vector2[1]])

    b1 = np.dot(nv1, start_point1)
    b2 = np.dot(nv2, start_point1)
    b3 = np.dot(nv3, start_point2)
    b4 = np.dot(nv4, start_point2)

    A = np.array([nv1, nv2, nv3, nv4])
    b = np.array([b1, b2, b3, b4])

    # check the rank of the coefficient matrix (nbr. of linearly ind. column vectors)
    matrix_rank = np.linalg.matrix_rank(A)

    # for a matrix of full rank of 3 an intersection exists
    if matrix_rank == 3:
        # Moore-Penrose generalized inverse
        A_gen_inv = np.dot(np.linalg.inv(np.dot(np.transpose(A), A)), np.transpose(A))
        w = np.dot(A_gen_inv, b) # compute intersection
        is_intersecting = True

    return is_intersecting, w


def distance_between_lines(v1, v2, start_v1, start_v2):
    """

    :param v1:
    :param v2:
    :param start_v1:
    :param start_v2:
    :return:
    """
    # morroworks.com/Content/Docs/Rays%20closest%20point.pdf
    c = start_v2 - start_v1
    D = start_v1 + v1 * ((-np.dot(v1, v2) * np.dot(v2, c) + np.dot(v1, c) * np.dot(v2, v2)) / (
            np.dot(v1, v1) * np.dot(v2, v2) - np.dot(v1, v2) * np.dot(v1, v2)))
    E = start_v2 + v2 * ((np.dot(v1, v2) * np.dot(v1, c) - np.dot(v2, c) * np.dot(v1, v1)) / (
            np.dot(v1, v1) * np.dot(v2, v2) - np.dot(v1, v2) * np.dot(v1, v2)))
    return np.linalg.norm(E-D), (D + E) / 2.


def angle_between_vectors(v1, v2):
    """
    Compute angle between two vectors.

    :param v1:
    :param v2:
    :return:
    """
    return np.arccos(sum((a * b) for a, b in zip(v1, v2)) / (
            np.sqrt(sum((a * b) for a, b in zip(v1, v1))) * np.sqrt(sum((a * b) for a, b in zip(v2, v2)))))


def create_disk_mask(slice_dim, center, radius):
    """

    Parameters
    ----------
    slice_dim :
    center :
    radius :

    Returns
    -------

    """

    x_val = slice_dim[0]
    y_val = slice_dim[1]

    coord = np.ogrid[0:x_val, 0:y_val]

    xs = coord[0] - center[0]
    ys = coord[1] - center[1]

    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    disk_mask = np.square(xx) + np.square(yy)
    disk_mask = disk_mask < radius**2
    return disk_mask


def create_sector_mask(slice_dim, center, angle1, angle2):
    """

    Parameters
    ----------
    slice_dim :
    center :
    angle1 :
    angle2 :

    Returns
    -------

    """
    x_val = slice_dim[0]
    y_val = slice_dim[1]

    coord = np.ogrid[0:x_val, 0:y_val]

    xs = coord[0] - center[0]
    ys = coord[1] - center[1]
    xx, yy = np.meshgrid(xs, ys, indexing='ij')

    sector_mask = np.arccos(np.divide(xx, np.sqrt(np.square(xx) + np.square(yy))))
    sector_mask = rad2deg(sector_mask)
    sector_mask3 = (yy < 0)
    sector_mask[sector_mask3] *= (-1)
    sector_mask[sector_mask3] += 360

    sector_mask1 = sector_mask > angle1
    sector_mask2 = sector_mask < angle2

    if angle1 < angle2:
        sector_mask_tot = sector_mask1 * sector_mask2
    elif angle1 > angle2:
        sector_mask_tot = sector_mask1 + sector_mask2
    else:
        sector_mask_tot = 0 * sector_mask1

    return sector_mask_tot


def create_circular_sector_mask(slice_dim, center, angle1, angle2, radius):
    """

    Parameters
    ----------
    slice_dim :
    center :
    angle1 :
    angle2 :
    radius :

    Returns
    -------

    """

    dd = create_disk_mask(slice_dim, center, radius)
    ss = create_sector_mask(slice_dim, center, angle1, angle2)

    return dd * ss


def create_cylinder_sector_mask(img_dimensions, base_point, direction, radius, height, angle1, angle2,):
    """
    Create a Cylinder along a coordinate axis with positive or negative direction given its height, radius and "base
    point".
    :param img_dimensions: outer dimensions of the masked image.
    :param base_point: Base_point
    :param direction: Direction to build cylinder from base_point
    :param radius: Cylinder radius in mm
    :param height: Cylinder height in mm
    :param angle1:
    :param angle2:
    :return: cylinder_mask_np:
    """
    cylinder_sector_mask_np = np.zeros(img_dimensions)  # empty mask
    switch_direction = {  # direction of the cylinder axis from base point
        'x': xprimary(),
        'y': yprimary(),
        'z': zprimary(),
        'x_neg': xprimary_neg(),
        'y_neg': yprimary_neg(),
        'z_neg': zprimary_neg()
    }

    primary_direction, secondary_direction, tertiary_direction, primary_ind, secondary_ind, tertiary_ind =\
        switch_direction.get(direction)  # define the direction using the switch statement above

    voxels_in_height = int(height)  # number of voxels in the given height

    planar_sector_center = np.array([base_point[secondary_ind], base_point[tertiary_ind]])
    slice_dimension = np.array([img_dimensions[secondary_ind], img_dimensions[tertiary_ind]])

    circular_sector = create_circular_sector_mask(slice_dimension, planar_sector_center, angle1, angle2, radius)
    for vH in range(0, voxels_in_height):  # for each layer of voxels define a circle and look for the voxels inside
        current_slice = base_point + vH * primary_direction
        current_slice = int(current_slice[primary_ind])

        if primary_ind == 0:
            cylinder_sector_mask_np[current_slice, :, :] = circular_sector
        elif primary_ind == 1:
            cylinder_sector_mask_np[:, current_slice, :] = circular_sector
        elif primary_ind == 2:
            cylinder_sector_mask_np[:, :, current_slice] = circular_sector

    return cylinder_sector_mask_np


def create_hollow_cylinder_sector_mask(img_dimensions, base_point, direction, radius, thickness,
                                       height, angle1, angle2):
    """
    Create a Cylinder along a coordinate axis with positive or negative direction given its height, radius and "base
    point".
    :param img_dimensions: outer dimensions of the masked image.
    :param base_point: Base_point
    :param direction: Direction to build cylinder from base_point
    :param radius:
    :param thickness:
    :param height:
    :param angle1:
    :param angle2:
    :return: cylinder_mask_np:
    """
    inner_cylinder_mask = create_cylinder_sector_mask(img_dimensions, base_point, direction, radius, height,
                                                      angle1, angle2)
    outer_cylinder_mask = create_cylinder_sector_mask(img_dimensions, base_point, direction, radius + thickness, height,
                                                      angle1, angle2)
    hollow_cylinder_sector_mask = outer_cylinder_mask - inner_cylinder_mask

    return hollow_cylinder_sector_mask


def binary_dilation(img_mask_np, radius=(20, 20, 20), output_type=sitk.sitkUInt8):
    """

    :param img_mask_np:
    :param radius:
    :param output_type:
    :return:
    :return:
    """
    img_mask_sitk = np2sitk(img_mask_np)
    dilated = sitk.BinaryDilate(img_mask_sitk, radius)
    dilated = sitk.Cast(dilated, output_type)
    dilated = sitk2np(dilated)
    return dilated


def binary_erosion(img_np, radius=(20, 20, 20), output_type=sitk.sitkUInt8):
    """

    :param img_np:
    :param radius:
    :param output_type:
    :return:
    """
    img_sitk = np2sitk(img_np)
    eroded = sitk.BinaryErode(img_sitk, radius)
    eroded = sitk.Cast(eroded, output_type)
    eroded = sitk2np(eroded)
    return eroded


def create_line_mask(start_point, direction, length, shape):
    """

    Parameters
    ----------
    direction :
    length :
    shape :

    Returns
    -------

    """
    mask = np.zeros(shape)
    line = np.ones(length, dtype=int)

    #mask[start_point[0]:start_point[0]+length, start_point[1]] = line
    mask[start_point[0], start_point[1], start_point[2]:start_point[2]+length] = line

    return mask


def create_arbitrarily_oriented_cylinder_mask(mask_shape, center, direction, radius, height):
    """
    Create a mask of a arbitrarily oriented cylinder in space.
    Parameters
    ----------
    mask_shape : 3-list, 3-tuple or nd.array of size 3 containing the outer dimensions of the image
    center : 3-list, 3-tuple or nd.array containing the center coordinates
    direction : 3-list, 3-tuple or nd.array containing the direction of the longitudinal axis
    radius : int or float with the radius
    height : int or float of the height

    Returns: mask: binary mask with 1 for the cylinder and 0 in the background
    -------
    """
    cylinder_mask = np.stack(np.meshgrid(np.arange(0, mask_shape[0]),
                                         np.arange(0, mask_shape[1]),
                                         np.arange(0, mask_shape[2]), indexing='ij'), axis=3)

    # check types of direction and center
    if type(center) == "list" or "tuple":
        center = np.array(center)

    rel_pos = cylinder_mask - np.array(center)
    if type(direction) == "list" or "tuple":
        v_dir = normi(np.array(direction))
    else:
        v_dir = normi(direction)

    # comupte projections on longitudinal and radial direction of cylinder
    rel_pos_proj_ax = np.abs(np.inner(rel_pos, v_dir))
    rel_pos_proj_rad = np.sqrt(np.sum(np.square(rel_pos), axis=3) - np.square(rel_pos_proj_ax))

    # compute masks for voxels within the longitudinal limits and radial limits respectively
    axial = np.where(rel_pos_proj_ax < height/2, 1, 0)
    radial = np.where(rel_pos_proj_rad < radius, 1, 0)

    # multiply the two images to get the cylinder mask
    mask = axial * radial

    return mask


def set_mayavi_offscreen(offscreen_on):
    """

    Parameters
    ----------
    offscreen_on :

    Returns
    -------

    """
    mlab.options.offscreen = offscreen_on
    return offscreen_on


def set_mayavi_orientation_axes():
    mlab.orientation_axes()


def set_mayavi_view(angle1, angle2):
    mlab.view(angle1, angle2)


def resample_volume_sitk(volume, interpolator=sitk.sitkLinear, new_spacing=(1.0, 1.0, 1.0)):
    """

    Parameters
    ----------
    volume :
    interpolator :
    new_spacing :

    Returns
    -------

    """
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]
    old_origin = volume.GetOrigin()
    new_origin = tuple(np.array(old_origin) - 0.5 * np.array(original_spacing) + 0.5 * np.array(new_spacing))
    return sitk.Resample(volume, new_size, sitk.Transform(), interpolator,
                         new_origin, new_spacing, volume.GetDirection(), 0,
                         volume.GetPixelID())


def npy2mhd(in_file, out_file, outfile_pixel_type):
    """

    Parameters
    ----------
    in_file : npy file to read (absolute path)
    out_file : mhd file to write (absolute path)
    outfile_pixel_type : output pixel type (SITK pixel type)

    Returns
    -------

    """
    array = np.load(in_file)  # load npy file
    array_sitk = np2sitk(array, output_pixel_type=outfile_pixel_type)  # transform numpy array to sitk image

    writer = sitk.ImageFileWriter()  # create SITK writer object
    writer.SetFileName(out_file)  # set filename
    writer.Execute(array_sitk)  # write file



