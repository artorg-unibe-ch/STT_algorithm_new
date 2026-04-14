"""
Created on 20210226
@author: Yvan Gugler
{

}
"""
# import modules
from Pipeline_Modules import img_geom_utils
from Pipeline_Modules.mesh_classes import element
from Pipeline_Modules.mesh_classes import node
from Pipeline_Modules.lib import *


def initialize_bone_dict(img_mask_sitk, image_information_dict, meta_data_dict, late=None, scanHeadDown=False,
                         add_layer=100,
                         csv_filename=None):
    """

    :param img_mask_sitk:
    :param image_information_dict:
    :param meta_data_dict:
    :param laterality:
    :param scanHeadDown:
    :param add_layer:
    :param csv_filename:
    :return:
    """
    bmd_array = img_geom_utils.sitk2np(img_mask_sitk)
    if scanHeadDown:
        bmd_array = np.flip(bmd_array, axis=0)
        bmd_array = np.flip(bmd_array, axis=1)

    bmd_array, mask_array, meta_data_dict, axis_flipped = standardize_image(bmd_array, meta_data_dict, add_layer,
                                                                            csv_filename, lat=late)

    bone_dict = {"BMD_Array": bmd_array}
    bone_dict["Mask_Array"] = mask_array
    bone_dict["Spacing"] = image_information_dict["Spacing"]
    bone_dict["Original Size"] = image_information_dict["Size"]
    bone_dict["Laterality"] = meta_data_dict["Laterality"]
    bone_dict["Size After Standardization"] = np.shape(bmd_array)
    bone_dict["Axis Flip"] = axis_flipped

    return bone_dict


def standardize_image(bmd_array, metadata_dict, add_layer=100, csv_filename=None, lat=None):
    """
    :param bmd_array:
    :param metadata_dict:
    :param add_layer:
    :param csv_filename:
    :return:
    """
    # TODO: Ask Yvan if the first if condition is based on the coordinate system might be different defined?
    mask_array = np.copy(bmd_array)
    mask_array[bmd_array != 0] = 1  # make binary mask
    mask_array.astype("uint8")

    im_flipped = False
    if lat is not None:
        laterality = lat
        lat_pca = femur_laterality(mask_array)
        if laterality != lat_pca:
            im_flipped = True
            bmd_array = np.flip(bmd_array, axis=0)
            bmd_array = np.flip(bmd_array, axis=1)
            mask_array = np.flip(mask_array, axis=0)
            mask_array = np.flip(mask_array, axis=1)
    else:
        try:
            laterality = metadata_dict['Laterality']
        except(KeyError, TypeError):
            try:
                laterality = shaper_read_laterality(csv_filename)  # look for csv file and read lat. from there
            except:
                laterality = femur_laterality(mask_array)  # PCA for detection of laterality

    if laterality == 'Left':
        mask_array = switch_laterality(mask_array)  # flip x axis in nd.array (left femur transformed to right femur)
        bmd_array = switch_laterality(bmd_array)

    elif laterality == 'Right':
        pass

    # compute bounding box and add layers
    bmd_array = img_geom_utils.bounding_box_np(bmd_array)
    mask_array = img_geom_utils.bounding_box_np(mask_array)
    bmd_array = img_geom_utils.add_layers_nb_np(bmd_array, add_layer)
    mask_array = img_geom_utils.add_layers_nb_np(mask_array, add_layer)

    # complete dictionary for laterality
    try:
        metadata_dict["Laterality"] = laterality
    except:
        metadata_dict = {"Laterality": laterality}

    return bmd_array, mask_array, metadata_dict, im_flipped


def shaper_read_laterality(file_to_read, delimiter=';'):
    """
    Read the laterality from a 3D-Shaper summary file (in CSV format) and return it as a string: 'Left' or 'Right'.
    :param file_to_read:
    :param delimiter:
    :return:
    """
    csv_file = open(file_to_read, newline='')
    deli = delimiter
    file_reader = csv.reader(csv_file, delimiter=deli)
    first_line = next(file_reader)
    test = first_line[0]

    if test[0:4] == 'Left':
        laterality = 'Left'
        return laterality
    elif test[0:5] == 'Right':
        laterality = 'Right'
        return laterality
    else:
        laterality = None
        print('Error reading Laterality from csv file')
        return laterality


def femur_laterality(img_mask_np):
    """
    Read in a binary mask of the proximal femur. Based on the assumption that an LPS system is used, detect the
    laterality of the femur in question.
    :param img_mask_np: Binary mask of proximal femur
    :return:
    """
    coordinates = np.transpose(np.nonzero(img_mask_np)).astype(int)

    pca = decomposition.PCA(n_components=1)  # we are interested in 1 component only (1 axis)
    pca.fit(coordinates)  # fit the data
    var = pca.components_

    test = var[0][0] * var[0][2]

    if test < 0:
        laterality = 'Left'
    elif test > 0:
        laterality = 'Right'

    return laterality


def switch_laterality(img_mask_np):
    """
    Switch the laterality of a bone image in LPS coordinate system.
    :param img_mask_np: three-dimensional numpy array
    :return: img_mask_np_flipped
    """
    img_mask_np_flipped = np.flip(img_mask_np, axis=0)  # flip x axis in nd.array
    return img_mask_np_flipped


def detect_coordinate_system(femur_mask_np_binary, background_layer, spacing=(1, 1, 1)):
    """
    :param femur_mask_np_binary:
    :param background_layer:
    :param spacing:
    :return:
    """
    # _________________________________________________________________________________________________________________
    # Head detection:
    # Approach a plane from 4 different directions and keep its intersection with the mask
    # 4 points: sup. -> inf. (z-axis) / med. -> lat. (x-axis) / ant. -> post. (y-axis) /
    # post. -> ant. (y-axis) - the last two are inclined 15° away from the y axis to the x axis.
    # initial sphere fit with 4 points
    # Definition of a set of directional vectors to repeat the above procedures and repeat the sphere fit.

    i = femur_mask_np_binary.shape[2] - background_layer + 5
    r_guess = 30  # Normal D_head = 40 to 54mm (https://www.sciencedirect.com/topics/engineering/femoral-head)
    r_guess = int(r_guess / spacing[0])

    # create an extra mask for the detection of the femoral head / set the right half zero, in order to prevent the
    # detection of the top point of the greater trochanter instead of the top point of the femoral head
    head_detection_mask = np.copy(femur_mask_np_binary)
    head_detection_mask[0:(femur_mask_np_binary.shape[0] - int(2.5 * r_guess)), :, :] = 0

    while np.mean(head_detection_mask[:, :, i]) == 0:
        i -= 1

    head_z = np.append(np.mean(np.argwhere(head_detection_mask[:, :, i] == 1), axis=0), i)  # top point of head
    head_z = head_z.astype('int')

    # recreate another head_detection_mask for the next steps of the process
    head_detection_mask = np.zeros(np.shape(femur_mask_np_binary))
    head_detection_mask[head_z[0] - int(1.2 * r_guess):head_z[0] + int(1.2 * r_guess),
    head_z[1] - int(1.2 * r_guess):head_z[1] + int(1.2 * r_guess),
    head_z[2] - int(2.3 * r_guess):head_z[2]] = 1
    head_detection_mask = head_detection_mask * femur_mask_np_binary  # mask containing only the head and surroundings

    v_x_image = np.array([1.01, 0.01, 0.01])  # x and y base vector in image
    v_y_image = np.array([0.01, 1.01, 0.01])
    v_z_image = np.array([0.01, 0.01, 1.01])

    rot_m1 = img_geom_utils.rotation_arbitrary_axis(v_z_image, np.deg2rad((-1.) * 75))
    rot_m2 = img_geom_utils.rotation_arbitrary_axis(v_z_image, np.deg2rad(-(-1.) * 75))

    v_ant_post = np.dot(rot_m1, (-1.) * v_x_image)
    v_post_ant = np.dot(rot_m2, (-1.) * v_x_image)

    """
    # Replaced by lines above to get rid off package transformations
    v_ant_post = np.dot(trans.rotation_matrix(np.deg2rad((-1.) * 75), v_z_image)[:3, :3],
                        (-1.) * v_x_image)
    v_post_ant = np.dot(trans.rotation_matrix(np.deg2rad(-(-1.) * 75), v_z_image)[:3, :3],
                        (-1.) * v_x_image)
    """

    # use the head_detection_mask to detect three more points on the surface of the head
    head_mediolateral = img_geom_utils.disc_collision_point(head_detection_mask, (-1.) * v_x_image,
                                                            head_z - (-1.) * 2.5 * r_guess * v_x_image, radius=60)
    head_anteroposterior, disc_head = img_geom_utils.disc_collision_point(head_detection_mask, v_ant_post,
                                                                          np.append(head_z[:2], head_mediolateral[2])
                                                                          - 2.5 * r_guess * v_ant_post, radius=30,
                                                                          last_disc=True)
    head_posteroanterior = img_geom_utils.disc_collision_point(head_detection_mask, v_post_ant,
                                                               np.append(head_z[:2], head_mediolateral[2])
                                                               - 2.5 * r_guess * v_post_ant, radius=30)
    # initial sphere fit with the four points
    head_radius, head_center = img_geom_utils.fit_sphere(
        np.c_[head_z, head_mediolateral, head_anteroposterior,
              head_posteroanterior].T)

    del head_detection_mask
    print('... initial guess for head detection completed')

    # define a set of directions in polar coordinates for refined sphere fit
    # angles in polar coordinates / phi: Azimuth angle / theta: elevation angle
    # https://en.wikipedia.org/wiki/Spherical_coordinate_system
    v_initial = v_x_image
    testing_angles_phi = np.linspace(-90, 90, 20)
    testing_angles_theta = np.linspace(-30, 120, 10)

    directional_vectors = []  # directional vectors defined by angles above

    for phi in testing_angles_phi:
        # R_phi = trans.rotation_matrix(np.deg2rad(phi), v_z_image)[:3, :3] # to get rid off package transformations
        R_phi = img_geom_utils.rotation_arbitrary_axis(v_z_image, np.deg2rad(phi))
        for theta in testing_angles_theta:
            # R_theta = trans.rotation_matrix(np.deg2rad(theta), np.dot(R_phi, -v_y_image))[:3, :3]
            R_theta = img_geom_utils.rotation_arbitrary_axis(np.dot(R_phi, -v_y_image), np.deg2rad(theta))
            directional_vectors.append(np.dot(np.dot(R_theta, R_phi), v_initial))

    # compute points on head
    points_on_head = []
    for vector in directional_vectors:
        points_on_head.append(
            img_geom_utils.disc_collision_point(femur_mask_np_binary, -vector, head_center + head_radius * 1.2 * vector,
                                                radius=3, nb_points_disc=25))
    # refined sphere fit
    head_radius, head_center_sphere = img_geom_utils.fit_sphere(np.array(points_on_head))

    print('... head detection completed')

    # _________________________________________________________________________________________________________________
    # Neck axis detection:
    # Create a series of spheres (head_radius +10 to +20) and compute intersection with binary mask
    # Compute center of intersection
    # Linear regression of centers to find a straight line
    # Compute a number of disks perpendicular to straight line above (more proximally than spheres above)
    # Center of gravity of each intersecting disk
    # Linear regression

    nb_u = 200  # nb. of points for sphere representation
    nb_v = 200

    inputs = np.linspace(10, 20, 10) + head_radius  # sphere radii
    neck_points_list = []

    for radius in inputs:
        u = np.linspace(0, 2 * np.pi, nb_u)  # angles for polar repr. of sphere
        v = np.linspace(0, np.pi, nb_v)
        x = np.rint(radius * np.outer(np.cos(u), np.sin(v)).reshape(nb_u * nb_v, 1) + head_center[0])  # surface coord.
        y = np.rint(radius * np.outer(np.sin(u), np.sin(v)).reshape(nb_u * nb_v, 1) + head_center[1])
        z = np.rint(radius * np.outer(np.ones(np.size(u)), np.cos(v)).reshape(nb_u * nb_v, 1) + head_center[2])
        sphere = np.c_[x, y, z]

        for i in np.arange(3):
            sphere = sphere[sphere[:, i] < femur_mask_np_binary.shape[i]]  # remove sphere coord. outside the image

        same_value = sphere[
                     femur_mask_np_binary[sphere[:, 0].astype(int), sphere[:, 1].astype(int), sphere[:, 2].astype(int)]
                     == 1, :]  # intersection
        center = np.mean(same_value, axis=0)  # CoM of intersection
        neck_points_list.append(center)

    v_neck, mean = img_geom_utils.linear_regression_3d(
        np.vstack(
            (np.array(head_center.tolist() * 10).reshape(10, 3), np.array(neck_points_list))))  # linear regression

    center_gravity_neck = []
    for i in np.linspace(0, 15, 20):
        disc_neck = img_geom_utils.disc_perpendicular_to_vector(head_center + v_neck * (head_radius * 0.7 + i), -v_neck,
                                                                radius=40)
        same_value = disc_neck[femur_mask_np_binary[disc_neck[:, 0].astype(int), disc_neck[:, 1].astype(
            int), disc_neck[:, 2].astype(int)] == 1, :]
        center = np.mean(same_value, axis=0)
        center_gravity_neck.append(center.tolist())

    v_neck, mean = img_geom_utils.linear_regression_3d(
        np.vstack(
            (np.array(head_center.tolist() * 10).reshape(10, 3), np.array(center_gravity_neck))))  # linear regression

    if np.dot(v_neck, v_x_image) < 1:
        v_neck = -v_neck

    print('  ... femoral neck detected!')

    # __________________________________________________________________________________________________________________
    # Diaphysis axis detection:
    # start from the first layer where we have pixels (given through bounding box) -> i = backround_layer
    # move upwards till the surface growth from slice to slice goes below a certain threshold (30mm²)
    # from there on compute the centers of mass of the next n slices
    # Linear regression to compute the equation of the diaphysis axis

    i = background_layer
    nbr_pixels0 = np.sum(femur_mask_np_binary[:, :, i])  # number of pixels in section i and section i+1
    nbr_pixels = np.sum(femur_mask_np_binary[:, :, i + 1])

    # look for the first slice where the variation of the size is smaller than 30 mm² (old measure)
    # / the total surface is approximately. 700 mm² (d=30mm) - 30mm² correspond to 4.2% of 700mm²
    # while nbr_pixels - nbr_pixels0 > int(30/spacing[0]):
    # new measure: use a relative variation of 5%

    while (nbr_pixels - nbr_pixels0) / nbr_pixels0 > 0.05:
        i += 1
        nbr_pixels0 = nbr_pixels
        nbr_pixels = np.sum(femur_mask_np_binary[:, :, i + 1])
        if i - background_layer >= 12:
            print('Diaphysis axis detection problem. Check the orientation of the proximal femur.')

    # if surface is too small this might be erroneous. Reference Hu_2018: Comparison of Proximal Femoral Geometry
    # and Risk Factors
    # between Femoral Neck Fractures and Femoral Intertrochanteric Fractures in an Elderly Chinese Population.
    # -> mean FSD - 2*STD (women) for surface area calculation.
    if nbr_pixels0 < 415:
        print('Diaphysis axis detection. Very small cross-sectional area!')

    # compute centroids of the next nbr_slices
    nbr_slices = int(10 / spacing[0])
    start_z = i
    end_z = i + nbr_slices
    slice_centroids = []

    mask_shape = np.shape(femur_mask_np_binary)
    face = np.rollaxis(np.indices((mask_shape[0], mask_shape[1])), 0, 2 + 1)

    for ii in range(start_z, end_z):
        present_slice = face[femur_mask_np_binary[:, :, ii] == 1]  # current slice
        slice_centroid = np.mean(present_slice, axis=0)  # slice centroid
        slice_centroid = np.append(slice_centroid, ii)  # 3D coord. of slice centroid
        slice_centroids.append(slice_centroid)

    v_dia, mean_point = img_geom_utils.linear_regression_3d(np.array(slice_centroids))  # linear regression

    if np.dot(v_dia, v_z_image) < 0:  # check if v_dia is directed from inf. to sup.
        v_dia = -v_dia

    # compute the intersection of the regression line with the first slice that was used
    factor = (start_z - mean_point[2]) / v_dia[2]
    d_point = mean_point + factor * v_dia

    # check how the diaphysis axis is directed with respect to the z-axis of the image space
    # if the angle is very large, this may be due to an erroneous detection
    # check the direction if only the lower or upper half of the points is used for the regression
    if np.arccos(np.dot(v_dia, np.array([0, 0, 1]))) / np.pi * 180 > 25:
        v_dia2, mean_point2 = img_geom_utils.linear_regression_3d(np.array(slice_centroids[0:5]))  # linear regression
        v_dia3, mean_point3 = img_geom_utils.linear_regression_3d(np.array(slice_centroids[5:]))  # linear regression

        if np.dot(v_dia2, v_z_image) < 0:  # check if v_dia is directed from inf. to sup.
            v_dia2 = -v_dia2

        if np.dot(v_dia3, v_z_image) < 0:  # check if v_dia is directed from inf. to sup.
            v_dia3 = -v_dia3

        if np.arccos(np.dot(v_dia2, v_dia3)) / np.pi * 180 > 25:
            v_dia = v_dia3
            mean_point = mean_point3
            factor = (start_z + int(nbr_slices / 2) - mean_point[2]) / v_dia[2]
            d_point = mean_point + factor * v_dia

    # __________________________________________________________________________________________________________________
    # transform the image so that v_dia is aligned with the vertical axis --> look for the first complete
    # section.
    rotation_matrix0 = img_geom_utils.identity_matrix
    translation0 = -d_point
    transformation_matrix0 = img_geom_utils.homogenous_transformation_matrix(rotation_matrix0, translation0)

    # align diaphysis with z-axis
    rotation_matrix1 = img_geom_utils.R_from_vector(v_dia, img_geom_utils.v_z)
    translation1 = img_geom_utils.zero_vec
    transformation_matrix1 = img_geom_utils.homogenous_transformation_matrix(rotation_matrix1, translation1)

    #  second rotation - align the projected neck axis with the x - axis
    translation2 = d_point
    transformation_matrix2 = img_geom_utils.homogenous_transformation_matrix(rotation_matrix0, translation2)

    total_transformation_matrix = np.dot(np.dot(transformation_matrix2, transformation_matrix1),
                                         transformation_matrix0)

    img_mask_transformed = img_geom_utils.transform_mask_to_experimental_position(femur_mask_np_binary,
                                                                                  total_transformation_matrix,
                                                                                  (1, 1, 1))

    x1, x2, y1, y2, z1, z2 = img_geom_utils.bounding_box_start_end_np(img_mask_transformed)
    i = z1
    nbr_pixels0 = np.sum(img_mask_transformed[:, :, i])  # number of pixels in section i and section i+1
    nbr_pixels = np.sum(img_mask_transformed[:, :, i + 1])

    new_d_point = img_mask_transformed[:, :, i + 1].nonzero()  # centroid of first complete section by dist. appr.
    new_d_point = np.mean(np.transpose(new_d_point), axis=0)
    new_d_point = np.append(new_d_point, i + 1)

    # search for first slice where relative variation is less than 5%
    while (nbr_pixels - nbr_pixels0) / nbr_pixels0 > 0.05:
        i += 1
        nbr_pixels0 = nbr_pixels
        nbr_pixels = np.sum(img_mask_transformed[:, :, i + 1])
        if i - z2 >= 12:
            print('Diaphysis axis detection problem. Check the orientation of the proximal femur.')

        new_d_point = img_mask_transformed[:, :, i].nonzero()  # centroid of first complete section by dist. appr.
        new_d_point = np.mean(np.transpose(new_d_point), axis=0)
        new_d_point = np.append(new_d_point, i)

    new_d_point_orig_config = img_geom_utils.transform_point(new_d_point,
                                                             np.linalg.inv(total_transformation_matrix))
    # __________________________________________________________________________________________________________________
    # Neck-shaft intersection, corrected v_dia and v_neck
    # diaphysis_base_point = slice_centroids[0]  # not used anymore but still returned!
    intersection_neck_shaft = img_geom_utils.closest_point_between_vector(v_neck, v_dia, head_center,
                                                                          d_point)
    v_dia_corr = img_geom_utils.normi(intersection_neck_shaft - d_point)
    v_neck_corr = img_geom_utils.normi(head_center_sphere - intersection_neck_shaft)
    neck_shaft_angle = 180 - img_geom_utils.rad2deg(np.arccos(np.dot(v_dia_corr, v_neck_corr) / (
            np.linalg.norm(v_neck_corr) * np.linalg.norm(v_dia_corr))))

    head_radius = head_radius[0]
    print("Coordinate system detected.")

    return head_center_sphere, head_radius, v_dia_corr, v_neck_corr, intersection_neck_shaft, \
           neck_shaft_angle, d_point, head_z, head_mediolateral, head_anteroposterior, head_posteroanterior, \
           v_dia, v_neck, new_d_point_orig_config


def detect_landmarks_and_correct_coordinate_system(img_mask, head_center, head_radius, v_dia, v_neck,
                                                   intersection_neck_shaft, d_point, v_neck_in, add_layer=30,
                                                   spacing=(1, 1, 1)):
    """
    Parameters
    ----------
    img_mask :
    head_center :
    head_radius :
    v_dia :
    v_neck :
    intersection_neck_shaft :
    d_point :
    v_neck_in :
    spacing :

    Returns
    -------

    """
    # __________________________________________________________________________________________________________________
    # landmark detection
    # __________________________________________________________________________________________________________________
    # coordinate axes
    v_x_image = np.array([1, 0, 0])
    v_z_image = np.array([0, 0, 1])

    # align the initially evaluated diaphysis axis with the z-axis of the voxel space and the projected neck axis with
    # the x axis
    rot_angle = np.arccos(
        np.dot(v_dia, v_z_image) / (np.linalg.norm(v_dia) * np.linalg.norm(v_z_image)))  # rotation angle
    v_rot = np.cross(v_dia, v_z_image)  # rotation axis

    # transformation to be performed - rotate around the base point of the diaphysis (d_point)
    # from the coordinate system detection --> Translation - Rotation - Translation
    rotation_matrix0 = np.eye(3, dtype=int)
    translation0 = -d_point
    transformation_matrix0 = img_geom_utils.homogenous_transformation_matrix(rotation_matrix0, translation0)

    rotation_matrix1 = img_geom_utils.rotation_arbitrary_axis(v_rot, rot_angle)  # align diaphysis with z-axis
    translation1 = np.zeros(3, dtype=int)
    transformation_matrix1 = img_geom_utils.homogenous_transformation_matrix(rotation_matrix1, translation1)

    # rotate the neck axis and project it on the x-y plane
    v_neck_rot = np.dot(rotation_matrix1, v_neck)
    v_neck_rot_proj = np.array([v_neck_rot[0], v_neck_rot[1], 0])  # projection on x-y-plane
    v_neck_in_rot = np.dot(rotation_matrix1, v_neck_in)  # rotate the initial neck axis

    # compute angle phi in x-y plane of the projected neck axis
    sign = np.sign(np.dot(np.cross(v_x_image, v_neck_rot_proj), v_z_image))
    phi = sign * img_geom_utils.angle_between_vectors(v_x_image, v_neck_rot_proj)

    #  second rotation - align the projected neck axis with the x - axis
    translation2 = np.array([0, 0, 0])
    rotation_matrix2 = img_geom_utils.rotation_arbitrary_axis(v_z_image, -phi)
    transformation_matrix2 = img_geom_utils.homogenous_transformation_matrix(rotation_matrix2, translation2)
    v_neck_in_rot2 = np.dot(rotation_matrix2, v_neck_in_rot)  # apply second rotation to initial v_neck

    # translate back
    translation3 = d_point
    transformation_matrix3 = img_geom_utils.homogenous_transformation_matrix(rotation_matrix0, translation3)

    # total transformation matrix
    total_transformation_matrix = np.dot(transformation_matrix3,
                                         np.dot(np.dot(transformation_matrix2, transformation_matrix1),
                                                transformation_matrix0))

    # transform the mask
    img_mask_transformed = img_geom_utils.transform_mask_to_experimental_position(img_mask,
                                                                                  total_transformation_matrix,
                                                                                  (1, 1, 1))
    # transform/recompute points in new position
    head_center_transformed = img_geom_utils.transform_point(head_center, total_transformation_matrix)
    intersection_neck_shaft_transformed = img_geom_utils.transform_point(intersection_neck_shaft,
                                                                         total_transformation_matrix)
    head_z_transformed = head_center_transformed + v_z_image * head_radius

    # __________________________________________________________________________________________________________________
    # Greater Trochanter - top point - detection:
    # Remove femoral head from mask.
    # Approach a disk perpendicular to diaphysis axis from above
    # Register the top most point

    head_removal = np.rint(
        np.array([head_center_transformed[0] - 1.6 * head_radius, head_center_transformed[0] + 1.6 * head_radius,
                  head_center_transformed[1] - 1.6 * head_radius, head_center_transformed[1] + 1.6 * head_radius,
                  head_center_transformed[2] - 1.6 * head_radius, head_center_transformed[2] +
                  1.6 * head_radius])).astype(int)  # define head region

    # check that the limit indexes for cutting the head are not outside the image dimensions. If so adapt them to the
    # image border
    mask_shape = np.shape(img_mask_transformed)
    if head_removal[1] > mask_shape[0]:
        head_removal[1] = mask_shape[0]

    if head_removal[2] < 0:
        head_removal[2] = 0

    if head_removal[3] > mask_shape[1]:
        head_removal[3] = mask_shape[1]

    if head_removal[5] > mask_shape[2]:
        head_removal = mask_shape[5]

    # copy mask and put head region to zero - detect top point of GT by approaching a disk from superior direction
    img_mask_transformed_copy = np.copy(img_mask_transformed)
    img_mask_transformed_copy[head_removal[0]:head_removal[1], head_removal[2]:head_removal[3],
    head_removal[4]:head_removal[5]] = 0  # set head region to 0

    gt_top_start = np.rint(np.array([intersection_neck_shaft_transformed[0], intersection_neck_shaft_transformed[1],
                                     head_z_transformed[2]])).astype(int)  # starting point
    gt_top, disc_gt = img_geom_utils.disc_collision_point(img_mask_transformed_copy, -v_z_image, gt_top_start,
                                                          radius=int(60 / spacing[0]),
                                                          last_disc=True)  # approach for detection of top point

    del img_mask_transformed_copy

    # __________________________________________________________________________________________________________________
    # Lesser Trochanter Detection
    # __________________________________________________________________________________________________________________
    # approach a hollow cylinder in a direction where the LT can be expected (angle between 20° and 70° from posteriorly
    # from x-axis). Register the initial collision points. From there on compute the contours of the region (nbr_slices
    # above and below). Look at the dispersion of the distances from the diaphysis axis per slice. Keep the slices with
    # the largest dispersion. Keep the largest distances in these slices.

    img_mask_transformed_copy = np.copy(img_mask_transformed)

    # cut the superior part of the mask (above the lesser trochanter) - Unnannuntana_2009 reports the following ratio
    # VO/FHD = 0.95 +-0.10 where VO: vertical offset between superior onset of lesser trochanter and FHD: femoral head
    # diameter as measured in digital photographs.
    # 0.7 = ca. mean(VO/FHD) - 2.5 * std(VO/FHD), which should be fine for most cases
    cut_height = int(head_center_transformed[2] - 0.7 * (2 * head_radius))
    img_mask_transformed_copy[:, :, cut_height:] = 0

    # remove voxels voxels that originate from bad segmentations (long extensions that stretch over several slices)
    bbox_indices = img_geom_utils.bounding_box_start_end_np(img_mask_transformed_copy)
    zmin = bbox_indices[4]
    zmax = bbox_indices[5]
    del bbox_indices

    rem_voxels = np.zeros(np.shape(img_mask_transformed))

    for slice in range(zmin, zmax):
        img_mask_transformed_copy[:, :, slice], removed_pixels = \
            img_geom_utils.remove_badly_connected_voxels(img_mask_transformed_copy[:, :, slice], con=4)
        rem_voxels[:, :, slice] = removed_pixels

    img_mask_transformed = (img_mask_transformed - rem_voxels).astype(int)

    # define initial hollow cylinder for approach
    cyl_height = int((cut_height - d_point[2]))
    cyl_radius = int(40 / spacing[0])
    sect_angle1 = 20
    sect_angle2 = 70
    thickness = int(5 / spacing[0])

    intersect = 0

    # approach hollow cylinder and register initial collision point (mask_intersection)
    while intersect == 0:
        hollow_cyl_sector = img_geom_utils.create_hollow_cylinder_sector_mask(mask_shape, d_point, "z", cyl_radius,
                                                                              thickness, cyl_height, sect_angle1,
                                                                              sect_angle2)
        mask_intersection = img_mask_transformed_copy * hollow_cyl_sector
        test = np.sum(mask_intersection)  # test if there is an intersection

        if test != 0:
            intersect = 1

        cyl_radius -= 1

    non_zero_coordinates = np.nonzero(mask_intersection)
    cog_non_zero_coordinates = np.array(
        [np.mean(non_zero_coordinates[0]), np.mean(non_zero_coordinates[1]), np.mean(non_zero_coordinates[2])])

    # vector from diaphysis axis to LT
    v_lt_init = img_geom_utils.normi(cog_non_zero_coordinates - np.array([d_point[0],
                                                                          d_point[1],
                                                                          cog_non_zero_coordinates[2]]))
    neck_lt_angle = img_geom_utils.angle_between_vectors(v_x_image, v_lt_init)  # angle between diaphysis and proj. neck

    nb_contours = int(40 / spacing[0])  # nbr of contours (40 seems a good choice in most cases)
    startSlice = int(cog_non_zero_coordinates[2] - int(nb_contours / 2))
    if startSlice < int(d_point[2]):
        nb_contours -= 2 * (int(d_point[2]) - startSlice) + 2
        startSlice = int(d_point[2]) + 1

    contour_mask = np.zeros(np.shape(img_mask_transformed))  # array to store the contours
    distance_mask = np.zeros(np.shape(img_mask_transformed))  # array to store the distances
    distance_dispersion = []  # store dispersions of distances

    for slice in range(startSlice, startSlice + nb_contours):

        transverse_slice = img_mask_transformed[:, :, slice]
        contours = measure.find_contours(transverse_slice, 0)  # compute contours in given slice

        nbr_voxels_in_contour = 0
        for contour in contours:  # keep the largest contour
            if nbr_voxels_in_contour < np.shape(contour)[0]:
                nbr_voxels_in_contour = np.shape(contour)[0]
                con = contour

        # the next part may be more complicated than necessary
        # compute distance of every voxel in the contour from the diaphysis axis and store in mask
        contour = np.column_stack((con, slice * np.ones(nbr_voxels_in_contour))).astype(int)
        for ind in range(0, nbr_voxels_in_contour):
            contour_mask[contour[ind][0], contour[ind][1], slice] = 1

            v_dia_axis_lt = np.array(
                [contour[ind][0] - d_point[0], contour[ind][1] - d_point[1]])
            distance_mask[contour[ind][0], contour[ind][1], slice] = np.linalg.norm(v_dia_axis_lt)

        # create a sector centered on the LT (angle computed above) and keep only the distances in this part
        circular_sector = img_geom_utils.create_circular_sector_mask(np.shape(distance_mask[:, :, slice]), np.array(
            [d_point[0], d_point[1], slice]), img_geom_utils.rad2deg(neck_lt_angle) - 30,
                                                                     img_geom_utils.rad2deg(neck_lt_angle) + 30,
                                                                     int(cyl_radius + 10 / spacing[0]))

        distance_mask[:, :, slice] = distance_mask[:, :, slice] * circular_sector

        distances = distance_mask[:, :, slice]  # distances of the voxels in the contour
        distances = distances[distances.nonzero()]  # non-zero distances
        std_distances = np.std(distances)  # standard deviation of these distances as measure of dispersion
        distance_dispersion.append(std_distances)  # add the dispersion of distances in the LT region

    lt_mask = np.copy(distance_mask)  # copy the mask of distances and reduce it further to get an estimate of the LT
    distance_dispersion = np.array(distance_dispersion)

    # consider the dispersion of distances array as a signal / look for the peaks in the signal.
    # The dispersion of the distances acts like a measure of curvature
    # Keep the most prominent peak in the signal and compute its prominence use the peak_dispersion - prominence_of_peak
    # as threshold to determine the upper and lower limits of the LT
    peaks, peak_properties = signal.find_peaks(distance_dispersion)  # peaks
    prominences = signal.peak_prominences(distance_dispersion, peaks)  # compute the prominence of the peaks
    most_prominent_peak_index = np.argmax(prominences[0])
    prominence = prominences[0][most_prominent_peak_index]

    left_base = prominences[1][most_prominent_peak_index]  # lower limit
    right_base = prominences[2][most_prominent_peak_index]  # upper limit
    center = peaks[most_prominent_peak_index]

    lt_peak = startSlice + center
    threshold = distance_dispersion[center] - prominence
    distance_dispersion_peak_region = np.where(distance_dispersion >= threshold, 1, 0)
    distance_dispersion_peak_region[:left_base] = 0
    distance_dispersion_peak_region[right_base + 1:] = 0

    # reduce the LT mask to the slices that are within the prominence
    # within these slices keep only the part that is larger than the mean distance of the contour within the slice
    for slice in range(startSlice, startSlice + nb_contours):
        lt_mask[:, :, slice] = lt_mask[:, :, slice] * distance_dispersion_peak_region[slice - startSlice]
        if distance_dispersion_peak_region[slice - startSlice] == 1:
            mean_dist_in_slice = np.sum(lt_mask[:, :, slice]) / np.count_nonzero(lt_mask[:, :, slice])
            lt_mask[:, :, slice] = np.where(lt_mask[:, :, slice] > mean_dist_in_slice, 1, 0)

    #  keep just the largest cluster of voxels
    lt_mask, removed_voxels = img_geom_utils.remove_badly_connected_voxels(lt_mask.astype(int), con=18)
    xmin, xmax, ymin, yman, zmin, zmax = img_geom_utils.bounding_box_start_end_np(lt_mask)
    V_proximal_femur = np.sum(img_mask_transformed_copy[:, :, zmin:]) * (spacing[0] * spacing[1] * spacing[2])
    del removed_voxels, xmin, xmax, ymin, yman, zmin, zmax

    # __________________________________________________________________________________________________________________
    # Define a single point for the lesser Trochanter
    # __________________________________________________________________________________________________________________
    # in the slice with the largest dispersion compute the COG of the contour of the LT.
    # compute the angles between the y-axis (0,1) and the direction vectors between the COG and each voxel
    # keep the voxel, which corresponds to the median of angles
    lt_peak0 = lt_peak
    lt_peak = lt_mask[:, :, lt_peak]  # keep the slice with the largest dispersion
    coordinates = lt_peak.nonzero()  # coordinates of the voxels in the contour
    cog_lt_peak = np.mean(coordinates, axis=1)  # center of mass of these coordinates
    dir_vecs = np.transpose(coordinates) - cog_lt_peak  # direction vector from COG to each of these voxels
    dir_vecs, dir_vec_norms = sklearn.preprocessing.normalize(dir_vecs, return_norm=True)  # normalize the direction vectors
    angle_correction1 = np.where(dir_vecs[:, 1] > 0, -1, 1) * np.where(dir_vecs[:, 0] > 0, 1, -1)
    angle_correction2 = np.where(dir_vecs[:, 0] > 0, 2 * np.pi, 0) * np.where(dir_vecs[:, 1] < 0, 1, 0)
    angle_correction3 = np.where(dir_vecs[:, 1] < 0, -1, 1) * np.where(dir_vecs[:, 0] > 0, 1, -1)

    angles = np.arccos(np.dot(dir_vecs, np.array([0, 1])))  # compute between direction vectors and [0,1] direction
    angles = angles * angle_correction1 \
             * angle_correction3 + angle_correction2  # apply correction (numpy returns values between 0 and pi)

    sort_angles = np.argsort(angles)
    len = np.shape(sort_angles)
    dir_vec_norms = dir_vec_norms[sort_angles]

    if len[0] % 2 == 1:
        med_angle_index = len[0] // 2  # look for the median of the angles - that's where we have the LT peak

    else:  # for an even amount of angles
        med_angle_ind1 = len[0] // 2 - 1
        med_angle_ind2 = len[0] // 2

        if dir_vec_norms[med_angle_ind1] < dir_vec_norms[med_angle_ind2]:
            med_angle_index = med_angle_ind1
        else:
            med_angle_index = med_angle_ind2

    med_angle_index = sort_angles[med_angle_index]
    lt_peak = np.transpose(coordinates)[med_angle_index, :]
    lt_peak = np.append(lt_peak, lt_peak0)  # peak point of the lesser trochanter

    cog_lt_peak = np.append(cog_lt_peak, lt_peak0)

    # __________________________________________________________________________________________________________________
    # Define a single point for the lesser Trochanter (alternative)
    # ! some more comments needed
    # __________________________________________________________________________________________________________________
    #
    r1, r2, c1, c2, lt_base0, lt_top0 = img_geom_utils.bounding_box_start_end_np(lt_mask)

    if lt_top0 - lt_base0 <= 2:
        lt_peak_alternative = lt_peak
        cog_lt_mid = cog_lt_peak
    else:
        lt_top = lt_mask[:, :, lt_top0 - 1]
        coordinates_top = lt_top.nonzero()
        lt_base = lt_mask[:, :, lt_base0]
        coordinates_base = lt_base.nonzero()

        cog_lt_top = np.append(np.mean(coordinates_top, axis=1), lt_top0 - 1)
        cog_lt_base = np.append(np.mean(coordinates_base, axis=1), lt_base0)

        v1 = cog_lt_top - cog_lt_base
        v2 = cog_lt_top - cog_lt_peak

        if not np.any(cog_lt_top - cog_lt_peak):
            lt_top = lt_mask[:, :, lt_top0 - 2]
            coordinates_top = lt_top.nonzero()
            cog_lt_top = np.append(np.mean(coordinates_top, axis=1), lt_top0 - 2)
            v2 = -cog_lt_top + cog_lt_peak

        if not np.any(cog_lt_base - cog_lt_peak):
            lt_base = lt_mask[:, :, lt_base0 + 1]
            coordinates_base = lt_base.nonzero()
            cog_lt_base = np.append(np.mean(coordinates_base, axis=1), lt_base0 + 1)
            v1 = cog_lt_top - cog_lt_base

        mid_point = 0.5 * (cog_lt_top + cog_lt_base)
        n_vec = np.cross(v1, v2)

        disk = img_geom_utils.disc_perpendicular_to_vector(mid_point, n_vec, radius=int(20 / spacing[0]),
                                                           nb_points_disc=int(200 / spacing[0])).astype(int)

        disk = disk[img_mask_transformed[disk[:, 0].astype(int), disk[:, 1].astype(
            int), disk[:, 2].astype(int)] == 1, :]

        intersect_mask = np.zeros(np.shape(img_mask_transformed))
        intersect_mask[disk[:, 0], disk[:, 1], disk[:, 2]] = 1
        intersect_mask = img_geom_utils.binary_dilation(intersect_mask,
                                                        radius=(int(1 / spacing[0]), int(1 / spacing[0]), int(1 / spacing[0])))

        intersect_mask = np.multiply(lt_mask, intersect_mask)

        lt_peak_candidates = np.transpose(intersect_mask.nonzero())
        nbr_candidates = np.shape(lt_peak_candidates)[0]

        lt_peak_alternative = 0
        dist = 0

        for cand in range(0, nbr_candidates):
            dist_test = img_geom_utils.closest_distance_between_vector_and_point(v1, cog_lt_base,
                                                                                 lt_peak_candidates[cand])
            if dist_test > dist:
                dist = dist_test
                lt_peak_alternative = lt_peak_candidates[cand]

        slice = lt_peak_alternative[2]
        lt_mid_slice = lt_mask[:, :, slice]
        coord_mid = lt_mid_slice.nonzero()
        cog_lt_mid = np.append(np.mean(coord_mid, axis=1), slice)  # center of mass of the slice where lt_peak_alt. is

    # __________________________________________________________________________________________________________________
    # Redefine the Diaphysis axis
    # __________________________________________________________________________________________________________________

    # length distal of LT available for redefinition of diaphysis axis
    distal_length = ((left_base + startSlice) - d_point[2]).astype(int)  # given by base point of diaphysis
    head_r_mean = (52.09 / 2) / spacing[0]  # Unnanuntana_2010
    alt_distal_length = int(head_radius / head_r_mean * 15 / spacing[0])  # scaled to head diameter

    distal_cut_height = min(distal_length, alt_distal_length)  # keep the minimum of the two

    # define the slices for redefinition of the diaphysis based on the foregoing
    start_z = left_base + startSlice - distal_cut_height
    end_z = left_base + startSlice
    slice_centroids = []
    slice_areas = []

    face = np.rollaxis(np.indices((mask_shape[0], mask_shape[1])), 0, 2 + 1)

    # compute slice centroids of slices below LT in the zone defined before
    for ii in range(start_z, end_z):
        present_slice = face[img_mask_transformed[:, :, ii] == 1]  # current slice
        slice_centroid = np.mean(present_slice, axis=0)  # slice centroid
        slice_area = np.sum(present_slice)  # slice area
        slice_areas.append(slice_area)
        slice_centroid = np.append(slice_centroid, ii)  # 3D coord. of slice centroid
        slice_centroids.append(slice_centroid)

    eccentricities = []
    orientations = []
    areas = []
    half_nbr_slices = int(10 / spacing[0])

    if half_nbr_slices > right_base:
        half_nbr_slices = right_base - 1

    # compute eccentricities, orientations and areas of slices around the upper border of the LT
    for ii in range(startSlice + right_base - half_nbr_slices, startSlice + right_base + half_nbr_slices):
        slice2 = img_mask_transformed[:, :, ii]
        props = measure.regionprops(slice2)
        for prop in props:
            areas.append(prop.area)
            eccentricities.append(prop.eccentricity)
            orientations.append(prop.orientation)

    ind = np.argmin(eccentricities)  # index of section with minimal eccentricity

    # if ind < half_nbr_slices it is below the upper border of the LT, so still in the LT region, which should rather
    # be more eccentric. In this case look at the eccentricities in this part and weight them with the distance
    # dispersion from above --> take the minimum of that
    if ind < half_nbr_slices:
        eccs = eccentricities[0:half_nbr_slices]
        disps = distance_dispersion[(right_base - half_nbr_slices):right_base]
        disp_weighted_eccentricities = np.array(eccs) * np.array(disps)
        ind = np.argmin(disp_weighted_eccentricities)

    # compute slice centroid of the slice with the minimal eccentricity
    present_slice = face[
        img_mask_transformed[:, :, startSlice + right_base - half_nbr_slices + ind] == 1]  # current slice
    slice_centroid_3 = np.mean(present_slice, axis=0)  # slice centroid
    slice_centroid_3 = np.append(slice_centroid_3, startSlice + right_base - half_nbr_slices + ind)

    slice_centroids.append(slice_centroid_3)

    # compute slice area ratios and keep only from where on the variation of area is less than 5%
    slice_area_ratios = np.array(slice_areas[:-1]) / np.array(slice_areas[1:])
    slice_area_ratios2 = slice_area_ratios > 0.95

    if np.any(slice_area_ratios2):
        i = np.argmax(slice_area_ratios2)  # first slice where the condition of 5% is satisfied

        # ransac fitting for axis computation
        model_robust, inliers = measure.ransac(np.array(slice_centroids[i:]), measure.LineModelND, min_samples=2,
                                               residual_threshold=0.2, max_trials=1000)

        # two candidates for diaphysis axis: cand1 through linear regression over all centroids
        # cand2: remove outliers through ransac fitting
        v_dia_new_candidate1, m_point = img_geom_utils.linear_regression_3d(
            np.array(slice_centroids[i:]))
        v_dia_new_candidate2 = model_robust.params[1]

        # compute distances from initial neck axis and closest point
        d1, p1 = img_geom_utils.distance_between_lines(v_dia_new_candidate1, v_neck_in_rot2, slice_centroids[i],
                                                       head_center_transformed)
        d2, p2 = img_geom_utils.distance_between_lines(v_dia_new_candidate2, v_neck_in_rot2, model_robust.params[0],
                                                       head_center_transformed)

        # keep the closer one
        if d1 < d2:
            v_dia_new = v_dia_new_candidate1
            db_point = slice_centroids[i]
            v_dia_new_final = img_geom_utils.normi(p1 - db_point)
            intersection_neck_shaft_new = p1
            dia_flag = 1

        else:
            v_dia_new = v_dia_new_candidate2
            db_point = model_robust.params[0]
            x = (slice_centroids[i][2] - db_point[2]) / v_dia_new[2]
            db_point = db_point + x * v_dia_new
            v_dia_new_final = img_geom_utils.normi(p2 - db_point)
            intersection_neck_shaft_new = p2
            dia_flag = 2

    else:
        # if the 5% criterion cannot be satisfied keep either the initial diaphysis axis or try if it yields
        # a better result by using the least eccentric slice above the LT
        v_dia_new_candidate1 = v_z_image
        v_dia_new_candidate2 = img_geom_utils.normi(slice_centroid_3 - d_point)

        d1, p1 = img_geom_utils.distance_between_lines(v_dia_new_candidate1, v_neck_in_rot2, d_point,
                                                       head_center_transformed)
        d2, p2 = img_geom_utils.distance_between_lines(v_dia_new_candidate2, v_neck_in_rot2, d_point,
                                                       head_center_transformed)

        db_point = d_point
        if d1 < d2:
            v_dia_new_final = img_geom_utils.normi(p1 - db_point)
            intersection_neck_shaft_new = p1
            dia_flag = 3

        else:
            v_dia_new_final = img_geom_utils.normi(p2 - db_point)
            intersection_neck_shaft_new = p2
            dia_flag = 4

    # update the neck_lt_angle
    factor = (lt_peak[2] - db_point[2]) / v_dia_new_final[2]
    lt_section_center = db_point + factor * v_dia_new_final
    v_lt = img_geom_utils.normi(lt_peak - lt_section_center)
    neck_lt_angle = img_geom_utils.angle_between_vectors(v_x_image, v_lt)

    # __________________________________________________________________________________________________________________
    # Define the corrected neck axis and neck_shaft_intersection in original image voxel space
    # __________________________________________________________________________________________________________________
    inv_total_transformation_matrix = np.linalg.inv(total_transformation_matrix)

    intersection_neck_shaft_new_final = img_geom_utils.transform_point(intersection_neck_shaft_new,
                                                                       inv_total_transformation_matrix)
    db_point = img_geom_utils.transform_point(db_point, inv_total_transformation_matrix)

    lt_peak = img_geom_utils.transform_point(lt_peak, inv_total_transformation_matrix)
    cog_lt_peak = img_geom_utils.transform_point(cog_lt_peak, inv_total_transformation_matrix)
    lt_peak_alternative = img_geom_utils.transform_point(lt_peak_alternative, inv_total_transformation_matrix)
    cog_lt_mid = img_geom_utils.transform_point(cog_lt_mid, inv_total_transformation_matrix)

    v_neck_new_final = img_geom_utils.normi(head_center - intersection_neck_shaft_new_final)
    v_dia_new_final = img_geom_utils.normi(intersection_neck_shaft_new_final - db_point)

    new_neck_shaft_angle = 180 - img_geom_utils.rad2deg(
        img_geom_utils.angle_between_vectors(v_neck_new_final, v_dia_new_final))

    # __________________________________________________________________________________________________________________
    # Transform the computed masks back to the voxel space of the initial array
    # __________________________________________________________________________________________________________________
    # seperate masks for LT and GT medial and lateral faces
    # dilate these "shell-like" masks and transform them to the original image space. There intersect them with a newly
    # generated contour mask (nearest neighbor interpol. for the transformations).

    lt_mask_orig_orientation = img_geom_utils.binary_dilation(lt_mask, radius=(1, 1, 1))
    lt_mask_orig_orientation = \
        img_geom_utils.transform_mask_to_experimental_position(lt_mask_orig_orientation,
                                                               inv_total_transformation_matrix,
                                                               (1, 1, 1), interpolator=sitk.sitkNearestNeighbor)

    start_point = img_geom_utils.transform_point(lt_peak, inv_total_transformation_matrix)
    start_point = int(start_point[2] - int(center - left_base))

    x1, x2, y1, y2, z1, z2 = img_geom_utils.bounding_box_start_end_np(lt_mask_orig_orientation)
    del x1, x2, y1, y2

    if start_point > z1:
        start_point = z1 + 2
        nb_contours = z2 - z1

    if start_point <= add_layer:
        start_point = add_layer + 1

    img_mask_copy = np.copy(img_mask)
    contour_mask = np.zeros(np.shape(img_mask_copy))  # array to store the contours

    # contouring of the original image in the LT zone
    # for slice in range(start_point, start_point + (right_base - left_base)):
    for slice in range(start_point, start_point + nb_contours):
        transverse_slice = img_mask_copy[:, :, slice]
        contours = measure.find_contours(transverse_slice, 0)  # compute contours in given slice
        nbr_voxels_in_contour = 0
        for contour in contours:  # keep the largest contour
            if nbr_voxels_in_contour < np.shape(contour)[0]:
                nbr_voxels_in_contour = np.shape(contour)[0]
                con = contour

        # form a mask of the contours computed above
        contour = np.column_stack((con, slice * np.ones(nbr_voxels_in_contour))).astype(int)
        for ind in range(0, nbr_voxels_in_contour):
            contour_mask[contour[ind][0], contour[ind][1], slice] = 1

    lt_mask_orig_orientation = lt_mask_orig_orientation * contour_mask

    # compute the mask of the greater trochanter in the original orientation
    gt_top_orig_orientation = img_geom_utils.transform_point(gt_top, inv_total_transformation_matrix)
    gt_sphere = img_geom_utils.create_sphere_2(mask_shape, 4, gt_top_orig_orientation)
    gt_mask_orig_orientation = np.multiply(gt_sphere, img_mask_copy)

    gt_mask_medial_orig_orientation = gt_mask_orig_orientation
    gt_mask_lateral_orig_orientation = gt_mask_orig_orientation

    # __________________________________________________________________________________________________________________
    # Compute the distal length
    # __________________________________________________________________________________________________________________
    if dia_flag == 1 or dia_flag == 2:
        # transform the image so that v_dia_new_final is aligned with the vertical axis --> look for the first complete
        # section.
        rotation_matrix0 = img_geom_utils.identity_matrix
        translation0 = -intersection_neck_shaft_new_final
        transformation_matrix0 = img_geom_utils.homogenous_transformation_matrix(rotation_matrix0, translation0)

        # align diaphysis with z-axis
        rotation_matrix1 = img_geom_utils.R_from_vector(v_dia_new_final, img_geom_utils.v_z)
        translation1 = img_geom_utils.zero_vec
        transformation_matrix1 = img_geom_utils.homogenous_transformation_matrix(rotation_matrix1, translation1)

        #  second rotation - align the projected neck axis with the x - axis
        translation2 = intersection_neck_shaft_new_final
        transformation_matrix2 = img_geom_utils.homogenous_transformation_matrix(rotation_matrix0, translation2)

        total_transformation_matrix = np.dot(np.dot(transformation_matrix2, transformation_matrix1),
                                             transformation_matrix0)

        img_mask_transformed = img_geom_utils.transform_mask_to_experimental_position(img_mask,
                                                                                      total_transformation_matrix,
                                                                                      (1, 1, 1))
        # lt_peak_transformed = img_geom_utils.transform_point(lt_peak, total_transformation_matrix)
        # cog_lt_peak_transformed = img_geom_utils.transform_point(cog_lt_peak, total_transformation_matrix)
        # lt_peak_alt_transformed = img_geom_utils.transform_point(lt_peak_alternative, total_transformation_matrix)
        x1, x2, y1, y2, z1, z2 = img_geom_utils.bounding_box_start_end_np(img_mask_transformed)

        i = z1
        nbr_pixels0 = np.sum(img_mask_transformed[:, :, i])  # number of pixels in section i and section i+1
        nbr_pixels = np.sum(img_mask_transformed[:, :, i + 1])

        # search for first slice where relative variation is less than 5%
        while (nbr_pixels - nbr_pixels0) / nbr_pixels0 > 0.05:
            i += 1
            nbr_pixels0 = nbr_pixels
            nbr_pixels = np.sum(img_mask_transformed[:, :, i + 1])
            if i - z2 >= 12:
                print('Diaphysis axis detection problem. Check the orientation of the proximal femur.')

        new_base_point = img_mask_transformed[:, :, i].nonzero()  # centroid of first complete section by dist. appr.
        new_base_point = np.mean(np.transpose(new_base_point), axis=0)
        new_base_point = np.append(new_base_point, i)

        new_base_point_orig_config = img_geom_utils.transform_point(new_base_point,
                                                                    np.linalg.inv(total_transformation_matrix))
    else:
        new_base_point_orig_config = db_point

    # compute the lengths
    lt_base_dir_vec = lt_peak - new_base_point_orig_config
    distal_length_along_vdia_new_final = np.dot(lt_base_dir_vec, v_dia_new_final)

    lt_db_dir_vec = lt_peak - db_point
    db_lt_dist = np.dot(lt_db_dir_vec, v_dia_new_final)
    lt_peak_projected_on_axis = db_point + db_lt_dist * v_dia_new_final

    print("Coordinate system detection corrected - LT detected - GT detected.")

    return v_dia_new_final, v_neck_new_final, intersection_neck_shaft_new_final, \
           db_point, lt_mask_orig_orientation, gt_mask_medial_orig_orientation, gt_mask_lateral_orig_orientation, \
           lt_peak, lt_peak_alternative, cog_lt_peak, cog_lt_mid, new_neck_shaft_angle, neck_lt_angle, dia_flag, \
           distal_length_along_vdia_new_final, new_base_point_orig_config, lt_peak_projected_on_axis, V_proximal_femur


def distally_extend_femur(img_mask, lt_pos, dia_base_point, intersect_neck_shaft, v_dia, spacing=(1, 1, 1),
                          target_distal_length=50):
    """
    Parameters
    ----------
    img_mask : np.ndarray with ndim = 3
    lt_pos : np.ndarray of size (3,) of float
    dia_base_point : np.ndarray of size (3,) of float
    intersect_neck_shaft : np.ndarray of size (3,) of float
    v_dia : np.ndarray of size (3,) of float
    spacing : 3-tuple of int or float
    target_distal_length : float or int: target length from lesser trochanter in [mm]

    Returns
    -------
    """

    v_z = np.array([0, 0, 1])

    base_lt_vec = lt_pos - dia_base_point  # vector from base point to LT
    distal_length = np.dot(base_lt_vec, v_z) * spacing[0]  # distance along diaphysis axis

    if distal_length >= target_distal_length:  # already long enough
        img_mask_extended = img_mask
        overlap = None
        shaft_mask_to_add_orig_orient = None
        extension_flag = False

    else:
        # transform img_mask to align v_dia with [0,0,1]
        rot_matrix = img_geom_utils.R_from_vector(v_dia, v_z)
        trans_vec = -intersect_neck_shaft

        rot_matrix2 = np.eye(3)
        trans_vec2 = intersect_neck_shaft

        trans_matrix = img_geom_utils.homogenous_transformation_matrix(rot_matrix, trans_vec)
        trans_matrix2 = img_geom_utils.homogenous_transformation_matrix(rot_matrix2, trans_vec2)

        trans_matrix_tot = np.dot(trans_matrix2, trans_matrix)

        img_mask_trans = img_geom_utils.transform_mask_to_experimental_position(img_mask, trans_matrix_tot, spacing,
                                                                                interpolator=sitk.sitkLinear)
        dia_base_point_trans = img_geom_utils.transform_point(dia_base_point, trans_matrix_tot)

        first_complete_slice = math.ceil(dia_base_point_trans[2]) + 1  # go one slice further than the base point
        add_length = math.ceil((target_distal_length - distal_length) * 1 / spacing[0]) + 1

        shaft_mask_to_add = np.zeros(np.shape(img_mask_trans))

        # these lines are not so clear ...
        start_slice = first_complete_slice - add_length + 1
        extrusion_height = add_length + 1

        if start_slice < 0:
            start_slice = 0
            extrusion_height = first_complete_slice + 2

        # build a mask by extruding the first "complete" section over a height that is equal to target - actual length
        shaft_mask_to_add[:, :, start_slice: first_complete_slice + 2] = \
            np.swapaxes(np.swapaxes(np.tile(
                img_mask_trans[:, :, first_complete_slice], (extrusion_height, 1, 1)), 0, 2), 0, 1)

        shaft_mask_to_add_binary = np.where(shaft_mask_to_add != 0, 1, 0)  # binarize the mask

        # erode the topmost elements
        struct_element = img_geom_utils.create_line_mask([1, 1, 1], [0, 0, 2], 3, [3, 3, 4])
        shaft_mask_to_add_binary_eroded = morphology.binary_erosion(shaft_mask_to_add_binary, struct_element).astype(
            int)

        # transform the BMD and the binary mask
        inv_trans_matrix_tot = np.linalg.inv(trans_matrix_tot)
        shaft_mask_to_add_orig_orient = img_geom_utils.transform_mask_to_experimental_position(
            shaft_mask_to_add, inv_trans_matrix_tot, spacing, interpolator=sitk.sitkLinear)

        shaft_mask_to_add_binary_eroded_orig_orient = img_geom_utils.transform_mask_to_experimental_position(
            shaft_mask_to_add_binary_eroded, inv_trans_matrix_tot, spacing, interpolator=sitk.sitkNearestNeighbor)

        # erode the top layer
        shaft_mask_to_add_orig_orient = shaft_mask_to_add_orig_orient * shaft_mask_to_add_binary_eroded_orig_orient
        shaft_mask_to_add_bin = np.copy(shaft_mask_to_add_orig_orient)
        shaft_mask_to_add_bin = np.where(shaft_mask_to_add_bin != 0, 1, 0)
        img_mask_bin = np.where(img_mask != 0, 1, 0)

        overlap = np.array(shaft_mask_to_add_bin * img_mask_bin)  # compute the overlap between extension and image
        overlap = np.array(overlap, dtype=bool)
        inv_overlap = np.invert(overlap).astype(int)  # invert it

        img_mask_extended = img_mask * inv_overlap + shaft_mask_to_add_orig_orient
        extension_flag = True

    return img_mask_extended, overlap, shaft_mask_to_add_orig_orient, extension_flag


def compute_transform_to_side_fall_position(neck_axis, diaphysis_axis, diaphysis_target_angle=30,
                                            rotation_target_angle=0, rotation_center=np.array([0, 0, 0])):
    """
    Compute the 4x4 transformation matrix, the SITK transform and its inverse for transforming the proximal femur
    to the desired experimental position.
    :param neck_axis: np.array() with the direction of the neck axis (unit vector).
    :param diaphysis_axis: np.array() with the direction of the diaphysis axis (unit vector).
    :param diaphysis_target_angle: Target inclination of the diaphysis angle with respect to the ground.
    :param rotation_target_angle: Internal rotation of the proximal femur defined as a rotation around the diaphysis
    axis.
    :param rotation_center: rotation Center for the Transformation.
    :return: transformation_matrix_tot, inverse_transformation_matrix
    """
    diaphysis_target_angle = img_geom_utils.deg2rad(diaphysis_target_angle)
    rotation_target_angle = img_geom_utils.deg2rad(rotation_target_angle)

    v_dia_target = np.array([np.sin(diaphysis_target_angle), 0, np.cos(diaphysis_target_angle)])  # dia. target dir.

    rotation_0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # translate to rotation center
    translation_0 = -rotation_center
    transformation_matrix_0 = img_geom_utils.homogenous_transformation_matrix(rotation_0, translation_0)

    xz_normal = np.array([0., 1., 0.])  # First rotation: make congruent diaphysis-neck plane and xz_plane
    normal = np.cross(diaphysis_axis, neck_axis)
    rotation_1 = img_geom_utils.R_from_vector(normal, xz_normal)
    translation_1 = np.zeros(3)
    transformation_matrix_1 = img_geom_utils.homogenous_transformation_matrix(rotation_1, translation_1)

    # Second rotation: rotate around y-axis (antero-posterior) to adjust the inclination of the diaphysis
    v_dia_trans1 = np.dot(rotation_1, diaphysis_axis)  # diaphysis axis after first rotation
    rotation_2 = img_geom_utils.R_from_vector(v_dia_trans1, v_dia_target)
    translation_2 = np.zeros(3)
    transformation_matrix_2 = img_geom_utils.homogenous_transformation_matrix(rotation_2, translation_2)

    # Third Rotation: rotate around diaphysis axis
    rotation_3 = img_geom_utils.rotation_arbitrary_axis(v_dia_target, rotation_target_angle)
    translation_3 = np.zeros(3)
    transformation_matrix_3 = img_geom_utils.homogenous_transformation_matrix(rotation_3, translation_3)

    rotation_4 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Translate back to origin
    translation_4 = rotation_center
    transformation_matrix_4 = img_geom_utils.homogenous_transformation_matrix(rotation_4, translation_4)

    # Total transform: compose the transformation matrices
    transformation_matrix_tot = np.dot(transformation_matrix_4, np.dot(transformation_matrix_3,
                                                                       np.dot(transformation_matrix_2,
                                                                              np.dot(transformation_matrix_1,
                                                                                     transformation_matrix_0))))
    inverse_transformation_matrix = np.linalg.inv(transformation_matrix_tot)  # matrix of inverse transform
    return transformation_matrix_tot, inverse_transformation_matrix


def compute_transform_to_stance_position(neck_axis, diaphysis_axis, diaphysis_target_angle=20,
                                         rotation_center=np.array([0, 0, 0])):
    """
    Compute 4x4 transformation matrix, the SITK transform and its inverse for transforming the proximal femur $
    to the desired experimental stance position.
    :param neck_axis: np.array
    :param diaphysis_axis: np.array
    :param diaphysis_target_angle: Angle in degrees
    :param rotation_center: Angle in degrees
    :return:
    """
    diaphysis_target_angle = img_geom_utils.deg2rad(diaphysis_target_angle)
    v_dia_target = np.array([np.sin(diaphysis_target_angle), 0, np.cos(diaphysis_target_angle)])  # target dir. of dia.
    rotation_0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # translate to rotationCenter
    translation_0 = -rotation_center
    transformation_matrix_0 = img_geom_utils.homogenous_transformation_matrix(rotation_0, translation_0)

    xz_normal = np.array([0., 1., 0.])  # First rotation: make congruent diaphysis-neck plane and xz_plane
    normal = np.cross(diaphysis_axis, neck_axis)
    rotation_1 = img_geom_utils.R_from_vector(normal, xz_normal)
    translation_1 = np.zeros(3)
    transformation_matrix_1 = img_geom_utils.homogenous_transformation_matrix(rotation_1, translation_1)

    # Second rotation: rotate around y-axis (antero-posterior) to adjust the inclination of the diaphysis
    v_dia_trans1 = np.dot(rotation_1, diaphysis_axis)  # diaphysis axis after first rotation
    rotation_2 = img_geom_utils.R_from_vector(v_dia_trans1, v_dia_target)
    translation_2 = np.zeros(3)
    transformation_matrix_2 = img_geom_utils.homogenous_transformation_matrix(rotation_2, translation_2)

    # Translate back to origin
    rotation_3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    translation_3 = rotation_center
    transformation_matrix_3 = img_geom_utils.homogenous_transformation_matrix(rotation_3, translation_3)

    # Total transform: compose the transformation matrices
    transformation_matrix_tot = np.dot(transformation_matrix_3,
                                       np.dot(transformation_matrix_2,
                                              np.dot(transformation_matrix_1, transformation_matrix_0)))
    inverse_transformation_matrix = np.linalg.inv(transformation_matrix_tot)  # matrix of inverse transform
    return transformation_matrix_tot, inverse_transformation_matrix


def bmd2bvtv(bmd_array_np):
    """
    Calibration equation according to Dall'Ara 2012.
    :param bmd_array_np: np-array containing the BMD values.
    :return: bvtv_array_np: np-array-containing the BVTV values.
    """
    bvtv_mask_np = np.copy(bmd_array_np)
    bvtv_array_np = np.copy(bmd_array_np)

    bvtv_array_np = np.where(bvtv_array_np <= -100, -100, bvtv_array_np)  # set BMD to -100 if smaller
    bvtv_mask_np = np.where(bvtv_mask_np <= -100, 1, bvtv_mask_np)

    bvtv_array_np = np.where(bvtv_array_np > 1064, 100, bvtv_array_np)
    bvtv_mask_np = np.where(bvtv_mask_np > 1064, 0, bvtv_mask_np)
    bvtv_mask_np[bvtv_mask_np != 0] = 1
    bvtv_mask_np = bvtv_mask_np.astype('int')

    bvtv_array_np = np.where(bvtv_mask_np == 1, 0.093 * bvtv_array_np + 1.077, bvtv_array_np)
    bvtv_array_np = np.where(bvtv_array_np >= 100, 100, bvtv_array_np)
    bvtv_array_np = np.where((bvtv_array_np < 1) & (bvtv_array_np > 0) | (bvtv_array_np < 0), 1, bvtv_array_np)
    return 1 / 100 * bvtv_array_np


def bvtv2bmd(bvtv_array_np):
    """
    Inverse of the calibration equation according to Dall'Ara 2012.
    :param bvtv_array_np: np-array containing the BVTV values.
    :return: bmd_array_np: np-array-containing the BMD values.
    """
    bmd_mask_np = np.copy(bvtv_array_np)
    bmd_array_np = 100 * np.copy(bvtv_array_np)
    bmd_mask_np[bmd_array_np != 0] = 1
    bmd_array_np = np.where(bmd_mask_np == 1, 1 / 0.093 * (bmd_array_np - 1.077), bmd_array_np)
    return bmd_array_np


def compute_bmc_and_volume(bmd_array, voxel_size, bvtv_array=None):
    """
    :param bmd_array:
    :param voxel_size:
    :param bvtv_array:
    :return:
    """
    dimensions = np.shape(bmd_array)
    if bvtv_array is None:
        bvtv_array = bmd2bvtv(bmd_array)

    total_volume = 0
    bmc = 0
    bone_vol = 0
    for x in range(0, dimensions[0]):
        for y in range(0, dimensions[1]):
            for z in range(0, dimensions[2]):
                if bmd_array[x, y, z] != 0:
                    total_volume += 1
                    bmc += bmd_array[x, y, z]
                    bone_vol += bvtv_array[x, y, z]
    mean_bmd = bmc / total_volume
    bmc *= voxel_size ** 3 / 1000
    total_volume *= voxel_size ** 3
    bone_vol *= voxel_size ** 3
    return mean_bmd, bmc, bone_vol, total_volume


def material_mapping(mask_array, bmd_array, trans_matrix, inv_trans_matrix, inp_spacing, radius=3, vox_size=3,
                     fabric_on=False):
    """
    Computes the material mapping from the initial proximal femur image to the image in experimental position with
    the voxel size that corresponds to the intended element size for the FE simulation.
    :param mask_array: np.array binary containing the masked bone
    :param bmd_array: np.array containing the bmd values in the initial image
    :param trans_matrix: transformation matrix (4x4) for the rigid transformation between initial and exp. positions
    :param inv_trans_matrix: inverse transformation matrix
    :param vox_size: isotropic voxel size
    :param inp_spacing: tuple of the input spacings
    :param fabric_on: bool: Fabric information to map
    :return: coarse_struct_array
    """
    # transform initial mask to exp position, dilate and resample at voxel size that is aimed at for the simulation
    rotated_mask = np.copy(mask_array)
    rotated_mask = img_geom_utils.transform_mask_to_experimental_position(rotated_mask, trans_matrix, inp_spacing)

    initial_coarse_mask = img_geom_utils.resampling(rotated_mask, interpolator=sitk.sitkNearestNeighbor,
                                                    input_spacing=inp_spacing, voxel_size=vox_size,
                                                    output_direction=None,
                                                    output_origin=(0, 0, 0), default_pixel_value=0,
                                                    output_pixel_type=sitk.sitkFloat64)

    initial_coarse_mask = img_geom_utils.close_holes(initial_coarse_mask)

    # create structured arrays that contain mask, BVTV information and fabric information
    dimensions = np.shape(initial_coarse_mask)
    if fabric_on:
        coarse_struct_array = np.zeros(dimensions,
                                       dtype={'names': ('Mask', 'BVTV', 'BMD', 'eigenvalues', 'eigenvectors'),
                                              'formats': ('u1', 'f4', 'f4', '3f4', '(3,3)f4')})
    else:
        coarse_struct_array = np.zeros(dimensions,
                                       dtype={'names': ('Mask', 'BVTV', 'BMD'), 'formats': ('u1', 'f4', 'f4')})

    elements_to_map = np.argwhere(initial_coarse_mask)  # determine the elements to map to from the initial mask
    mask_shape = np.shape(mask_array)
    # map BVTV information for isotropic case
    if not fabric_on:
        for element in elements_to_map:
            r = element[0]
            s = element[1]
            t = element[2]
            pos = np.array([r, s, t]) * vox_size
            pos = np.append(pos, 1)
            pos = np.dot(inv_trans_matrix, pos)
            sphere_mask = img_geom_utils.create_sphere_2(mask_shape, radius / inp_spacing[0],
                                                         1 / inp_spacing[0] * pos[0:3])
            local_bmd = np.mean(bmd_array[sphere_mask])
            local_bvtv = bmd2bvtv(local_bmd)
            if local_bvtv < 0.0 or (0.0 < local_bvtv < 0.01):
                coarse_struct_array[r, s, t] = (1, 0.01, local_bmd)
            elif local_bvtv >= 0.01:
                coarse_struct_array[r, s, t] = (1, local_bvtv, local_bmd)
            else:
                coarse_struct_array[r, s, t] = (0, 0.0, local_bmd)
    elif fabric_on:  # map BVTV and fabric information in the anisotropic case (to be done)
        pass
    return coarse_struct_array


def embed_side_fall(img_mask_np, voxel_size, head_radius=None):
    """
    Creates a mask for the embedding (like in the experiment) of the proximal femur in side fall position (two caps).
    :param img_mask_np: binary mask of the proximal femur
    :param voxel_size: size of the voxel/element in the final simulation
    :return: img_mask_embedded: np.array of integers (1: bone, 2: inner cylinder, 3: outer cylinder of embedding)
    """
    nx = img_mask_np.shape[0]
    ny = img_mask_np.shape[1]
    nz = img_mask_np.shape[2]

    d_guess = 65  # Normal D_head = 40 to 54mm (https://www.sciencedirect.com/topics/engineering/femoral-head)
    d_guess = int(d_guess / voxel_size)
    # nz_max = nz - d_guess

    thickness = 10.0  # thickness of embedding in mm (9.0mm in release v0.0.1)
    voxels_in_thickness = int(thickness / voxel_size)  # number of voxels in thickness

    # Embedding of the femoral head
    img_mask_np_top = np.copy(img_mask_np)
    img_mask_np_top[img_mask_np != 0] = 1

    bbox_indices = img_geom_utils.bounding_box_start_end_np(img_mask_np_top)
    nz_max = bbox_indices[-1] - d_guess  # added in v0.0.5

    img_mask_np_top[:, :, 0:nz_max] = 0  # set lower half zero -> avoid a detection of the most medial shaft point

    # detect first layer of voxel space that contains bone, when approaching in medio-lateral direction
    j = nx - 1
    while not img_mask_np_top[j, :, :].any() == 1:
        j -= 1

    # checking if a certain minimal number of voxels is found in the first layer, if not the embedding depth is
    # increased by 1, in order to increase the contact area. This should avoid failure of the head in the inferior part
    # of the contact. Added for version v0.0.2. / modified v0.0.3
    voxels_in_thickness_head = voxels_in_thickness

    if head_radius is None:
        head_radius = 23

    surface = (head_radius ** 2 - (head_radius - voxel_size) ** 2) * math.pi
    threshold = surface / (voxel_size ** 2)

    if thickness % voxel_size >= 0.5:
        threshold *= 0.65
    else:
        threshold *= 0.5

    threshold = math.ceil(threshold)

    if np.sum(img_mask_np_top[j, :, :]) <= threshold:
        voxels_in_thickness_head = voxels_in_thickness + 1

    j = j - voxels_in_thickness_head + 1

    # centering the embedding (use first embedded and first non-embedded slice to center it)
    # use function round() --> version v.0.0.1
    # slice_centroid = np.mean(np.argwhere(img_mask_np_top[j - 1, :, :] != 0), axis=0)
    # slice_centroid2 = np.mean(np.argwhere(img_mask_np_top[j, :, :] != 0), axis=0)
    # slice_centroid = np.round(0.5*(slice_centroid + slice_centroid2)).astype(int)
    # base_point = np.array([j, slice_centroid[0], math.round(slice_centroid[1])], dtype=int)

    # centering the embedding (non-embedded slice to center it)
    # use function ceil() for the component in z-direction) --> this has the tendence to move the embedding cap a bit
    # superiorely. --> v0.0.2 / v0.0.3
    # slice_centroid = np.mean(np.argwhere(img_mask_np_top[j - 1, :, :] != 0), axis=0)
    # base_point = np.array([j, slice_centroid[0], math.ceil(slice_centroid[1])], dtype=int)

    # v0.0.4
    # slice_centroid = np.mean(np.argwhere(img_mask_np_top[j, :, :] != 0), axis=0)
    # base_point = np.array([j, slice_centroid[0], slice_centroid[1]], dtype=int)

    # v0.0.6
    slice_centroid = np.mean(np.argwhere(img_mask_np_top[j, :, :] != 0), axis=0)
    base_point = np.array([j, round(slice_centroid[0]), math.ceil(slice_centroid[1])], dtype=int)

    del img_mask_np_top  # delete the mask of only the top part again

    foreground_pixel_value_outer_cyl = 3
    foreground_pixel_value_inner_cyl = 2
    img_dims = [nx, ny, nz]

    # creation of the cylinders
    img_mask_np_bin = np.copy(img_mask_np)
    img_mask_np_bin[img_mask_np != 0] = 1
    outer_cylinder_mask = img_geom_utils.generate_cylinder_mask(img_dims, base_point, 'x', 27, 27,
                                                                voxel_size, foreground_pixel_value_outer_cyl)
    inner_cylinder_mask = img_geom_utils.generate_cylinder_mask(img_dims, base_point, 'x', 24, 18,
                                                                voxel_size, foreground_pixel_value_inner_cyl)
    img_mask_embedded = np.add(img_mask_np_bin, np.add(inner_cylinder_mask, outer_cylinder_mask))

    # added for v0.0.7
    if voxels_in_thickness_head > voxels_in_thickness:
        outer_inner_cylinder_difference = outer_cylinder_mask + inner_cylinder_mask
        outer_inner_cylinder_difference = outer_inner_cylinder_difference[j, :, :]
        outer_inner_cylinder_difference = np.where(outer_inner_cylinder_difference == foreground_pixel_value_outer_cyl,
                                                   1, 0)
        outer_cyl_ring = np.zeros([nx, ny, nz])
        outer_cyl_ring[j - 1, :, :] = outer_inner_cylinder_difference

        if np.sum(outer_cyl_ring * img_mask_np_bin) != 0:
            base_point = np.array([j + 1, round(slice_centroid[0]), math.ceil(slice_centroid[1])], dtype=int)
            outer_cylinder_mask = img_geom_utils.generate_cylinder_mask(img_dims, base_point, 'x', 27, 27,
                                                                        voxel_size, foreground_pixel_value_outer_cyl)
            inner_cylinder_mask = img_geom_utils.generate_cylinder_mask(img_dims, base_point, 'x', 24, 18,
                                                                        voxel_size, foreground_pixel_value_inner_cyl)
            img_mask_embedded = np.add(img_mask_np_bin, np.add(inner_cylinder_mask, outer_cylinder_mask))

    # Embedding of the region of the Greater trochanter (gt).
    r = 0
    while not img_mask_np_bin[r, :, :].any() == 1:  # most lateral point of proximal femur (gt in contact with ground).
        r += 1

    r = r + voxels_in_thickness - 1

    # centering the embedding (use first embedded and first non-embedded slice to center it)
    slice_centroid_gt = np.mean(np.argwhere(img_mask_np_bin[r, :, :] != 0), axis=0)
    slice_centroid_gt2 = np.mean(np.argwhere(img_mask_np_bin[r + 1, :, :] != 0), axis=0)

    slice_centroid_gt = np.round(0.5 * (slice_centroid_gt + slice_centroid_gt2)).astype(int)
    base_point_gt = np.array([r, slice_centroid_gt[0], slice_centroid_gt[1]], dtype=int)

    # creation of the cylinders
    outer_cylinder_mask_gt = img_geom_utils.generate_cylinder_mask(img_dims, base_point_gt, 'x_neg', 30, 27,
                                                                   voxel_size, foreground_pixel_value_outer_cyl)
    inner_cylinder_mask_gt = img_geom_utils.generate_cylinder_mask(img_dims, base_point_gt, 'x_neg', 27, 18,
                                                                   voxel_size, foreground_pixel_value_inner_cyl)
    img_mask_embedded = np.add(img_mask_embedded, np.add(inner_cylinder_mask_gt, outer_cylinder_mask_gt))
    img_mask_embedded_final = np.zeros(np.shape(img_mask_embedded))

    for a in range(0, nx):
        for b in range(0, ny):
            for c in range(0, nz):
                if int(img_mask_embedded[a, b, c]) == \
                        foreground_pixel_value_inner_cyl + foreground_pixel_value_outer_cyl + 1:
                    img_mask_embedded_final[a, b, c] = img_mask_np_bin[a, b, c]
                elif int(img_mask_embedded[a, b, c]) == \
                        foreground_pixel_value_inner_cyl + foreground_pixel_value_outer_cyl:
                    img_mask_embedded_final[a, b, c] = foreground_pixel_value_inner_cyl
                elif int(img_mask_embedded[a, b, c]) == foreground_pixel_value_outer_cyl:
                    img_mask_embedded_final[a, b, c] = foreground_pixel_value_outer_cyl
                else:
                    img_mask_embedded_final[a, b, c] = img_mask_np_bin[a, b, c]

    return img_mask_embedded_final, voxels_in_thickness


def embed_stance(img_mask_np, voxel_size, head_radius=None):
    """
    Creates as mask for the embedding (like in the experiment) of the proximal femur in stance position (one cap).
    :param img_mask_np: binary mask of the proximal femur
    :param voxel_size: size of the voxel/element in the final simulation
    :return: img_mask_embedded: np.array of integers (1: bone, 2: inner cylinder, 3: outer cylinder of embedding)
    """
    nx = img_mask_np.shape[0]
    ny = img_mask_np.shape[1]
    nz = img_mask_np.shape[2]

    thickness = 10.0  # thickness of embedding in mm (9.0mm up to release v0.0.8)
    voxels_in_thickness = int(thickness / voxel_size)  # number of voxels in thickness
    voxels_in_thickness_head = voxels_in_thickness

    # Embedding of the femoral head - detect the top point
    i = nz - 1

    while not img_mask_np[:, :, i].any() == 1:
        i -= 1

    # checking if a certain minimal number of voxels is found in the first layer, if not the embedding depth is
    # increased by 1, in order to increase the contact area. This should avoid failure of the head in the inferior part
    # of the contact. Added for version v0.0.2. / modified v0.0.3
    voxels_in_thickness_head = voxels_in_thickness

    if head_radius is None:
        head_radius = 23

    surface = (head_radius ** 2 - (head_radius - voxel_size) ** 2) * math.pi
    threshold = surface / (voxel_size ** 2)

    if thickness % voxel_size >= 0.5:
        threshold *= 0.65
    else:
        threshold *= 0.5

    threshold = math.ceil(threshold)

    if np.sum(img_mask_np[:, :, i]) <= threshold:
        voxels_in_thickness_head = voxels_in_thickness + 1

    # from the top point find the most inferior slice that will be in the embedding and compute the centroid of the
    # cross-sectional area at this slice
    i = i + 1 - voxels_in_thickness_head

    # centering the embedding (use first embedded and first non-embedded slice to center it)
    slice_centroid = np.mean(np.argwhere(img_mask_np[:, :, i] != 0), axis=0)
    slice_centroid2 = np.mean(np.argwhere(img_mask_np[:, :, i - 1] != 0), axis=0)
    slice_centroid = np.round(0.5 * (slice_centroid + slice_centroid2)).astype(int)
    base_point = np.array([slice_centroid[0], slice_centroid[1], i]).astype('int')

    # set the foreground pixel values for the different parts of the mask
    foreground_pixel_value_outer_cyl = 3
    foreground_pixel_value_inner_cyl = 2
    img_dims = [nx, ny, nz]

    # generate cylinder masks for the two embedding materials
    outer_cylinder_mask = img_geom_utils.generate_cylinder_mask(img_dims, base_point, 'z', 27, 27,
                                                                voxel_size, foreground_pixel_value_outer_cyl)
    inner_cylinder_mask = img_geom_utils.generate_cylinder_mask(img_dims, base_point, 'z', 24, 18,
                                                                voxel_size, foreground_pixel_value_inner_cyl)
    img_mask_embedded = np.add(img_mask_np, np.add(inner_cylinder_mask, outer_cylinder_mask))

    #  look for areas of bone, embedding material 1 and embedding material 2
    for a in range(0, nx):
        for b in range(0, ny):
            for c in range(0, nz):
                if int(img_mask_embedded[a, b, c]) == \
                        foreground_pixel_value_inner_cyl + foreground_pixel_value_outer_cyl + 1:
                    img_mask_embedded[a, b, c] = 1
                elif int(img_mask_embedded[a, b, c]) == \
                        foreground_pixel_value_inner_cyl + foreground_pixel_value_outer_cyl:
                    img_mask_embedded[a, b, c] = foreground_pixel_value_inner_cyl
    return img_mask_embedded, voxels_in_thickness


def crop_femur(img_mask_np):
    """
    Needs to be modified!
    Distal cropping of the femur after orienting, material mapping and embedding. The femur is cut at the first cross-
    section of the diaphysis (coming from the distal end) that is complete.
    :param img_mask_np: np.array: mask of the embedded proximal femur.
    :return: crop_limits: a list of containing the limit indices of the box that has to be cut out in the image arrays.
    """

    # compute the bounding box start and end indices, which is sufficient for all directions except the z-direction.
    crop_limits = list(img_geom_utils.bounding_box_start_end_np(img_mask_np))
    img_mask_np_cp = img_geom_utils.bounding_box_np(img_mask_np)
    surfaces = []
    ratios = []
    nbr_voxels = 25

    # search the first complete cross-section coming from the distal side
    for i in range(0, nbr_voxels):
        surf = np.sum(img_mask_np_cp[:, :, i])
        surfaces.append(surf)

    for j in range(0, nbr_voxels - 1):
        ratios.append(surfaces[j] / surfaces[j + 1])

    # define the cut_off slice
    cut_off = np.min(np.argwhere(np.array(ratios) > 0.9))
    crop_limits[4] += cut_off

    return crop_limits


def crop_femur_new(img_mask_np, fe_el_size, perpendicular_cut=False, cut_location=None, v_dia=None):
    """

    Parameters
    ----------
    img_mask_np :
    fe_el_size :
    perpendicular_cut :
    cut_location :
    v_dia :

    Returns
    -------

    needs to be commented properly!

    """

    if perpendicular_cut is False and cut_location is None:

        # compute the bounding box start and end indices, which is sufficient for all directions except the z-direction.
        crop_limits = list(img_geom_utils.bounding_box_start_end_np(img_mask_np))
        img_mask_np_cp = img_geom_utils.bounding_box_np(img_mask_np)

        surfaces = []
        ratios = []
        nbr_voxels = 25
        # search the first complete cross-section coming from the distal side
        for i in range(0, nbr_voxels):
            surf = np.sum(img_mask_np_cp[:, :, i])
            surfaces.append(surf)

        for j in range(0, nbr_voxels - 1):
            ratios.append(surfaces[j] / surfaces[j + 1])

        # define the cut_off slice
        cut_off = np.min(np.argwhere(np.array(ratios) > 0.9))
        crop_limits[4] += cut_off
        cut_off_mask = np.copy(img_mask_np)
        cut_off_mask[:, :, crop_limits[4]:] = 0
        remaining_mask = (np.logical_not(cut_off_mask) * img_mask_np).astype(int)
        crop_limits = list(img_geom_utils.bounding_box_start_end_np(remaining_mask))
        boundary_elements = None  # mask with boundary elements, only applicable if perpendicular cut is used

    elif perpendicular_cut is False and cut_location is not None:
        crop_limits = list(img_geom_utils.bounding_box_start_end_np(img_mask_np))
        crop_limits[4] = int(cut_location[2])
        cut_off_mask = np.copy(img_mask_np)
        cut_off_mask[:, :, crop_limits[4]:] = 0
        remaining_mask = (np.logical_not(cut_off_mask) * img_mask_np).astype(int)
        crop_limits = list(img_geom_utils.bounding_box_start_end_np(remaining_mask))
        boundary_elements = None

    else:
        cos_angle = np.dot(img_geom_utils.v_z, v_dia)
        # height = 50 / fe_el_size  # mask elements to cut (cylinder mask with arbitrary orientation)
        height = math.ceil(cut_location[2] / cos_angle)
        radius = 40 / fe_el_size
        cyl_center = cut_location - height / 2 * v_dia
        cut_off_mask = img_geom_utils.create_arbitrarily_oriented_cylinder_mask(np.shape(img_mask_np), cyl_center,
                                                                                -v_dia, radius, height)
        cut_off_mask = cut_off_mask * img_mask_np  # part that is cut off
        remaining_mask = (np.logical_not(cut_off_mask) * img_mask_np).astype(int)  # part of the mask that remains
        crop_limits = list(img_geom_utils.bounding_box_start_end_np(remaining_mask))

        # mask elements at boundary after cut (for BC definition)
        struct_element = img_geom_utils.create_line_mask([1, 1, 1], [0, 0, 2], 2, [3, 3, 3])
        cut_off_mask_dilated = morphology.binary_dilation(cut_off_mask, struct_element).astype(
            int)

        boundary_elements = cut_off_mask_dilated * remaining_mask

    return crop_limits, remaining_mask, cut_off_mask, boundary_elements


def bmc_correction(input_bmd_mask, fe_bmd_mask, in_spacing, fe_spacing, cut_off_mask, transformation_matrix):
    """

    Parameters
    ----------
    input_bmd_mask : masked input bmd image
    fe_bmd_mask : FE bmd mask
    in_spacing : (3,) tuple/list of int or float
    fe_spacing : (3,) tuple/list of int or float
    cut_off_mask : mask in fe resolution that was removed
    transformation_matrix :

    Returns
    -------
    """
    shape_original_img = np.shape(input_bmd_mask)  # shape of original mask
    transformed_cut_off_mask = np.zeros(shape_original_img)  # create mask for the mask that is transformed back

    # resample the cut_off_mask from the fe voxel space to the input image space
    transformed_cut_off_mask_0 = img_geom_utils.resampling(cut_off_mask, input_spacing=fe_spacing,
                                                           voxel_size=in_spacing[0])
    # transform the mask to align it with orientation of input mask
    transformed_cut_off_mask_0 = img_geom_utils.transform_mask_to_experimental_position(
        transformed_cut_off_mask_0,
        transformation_matrix, in_spacing,
        interpolator=sitk.sitkNearestNeighbor)

    # dilate it laterally to be sure to cut off everything (no dilation in pos. z direction (cut surface))
    struct_el_width = 3 * math.ceil(fe_spacing[0] / in_spacing[0])
    struct_element = np.ones([struct_el_width, struct_el_width, 3], dtype=int)
    struct_element[:, :, 2] = np.zeros([struct_el_width, struct_el_width], dtype=int)
    transformed_cut_off_mask_0 = morphology.binary_dilation(transformed_cut_off_mask_0, struct_element).astype(
        int)

    # shape of fe image may be different from original mask as in and out spacing may not be int. multiples
    shape_fe_img = np.shape(transformed_cut_off_mask_0)
    transformed_cut_off_mask[0:shape_fe_img[0], 0:shape_fe_img[1], 0:shape_fe_img[2]] = \
        transformed_cut_off_mask_0.astype(bool)

    transformed_cut_off_mask = np.logical_not(transformed_cut_off_mask).astype(int)
    remaining_input_mask = transformed_cut_off_mask * input_bmd_mask

    # remove loose voxels
    remaining_input_mask_binary = np.copy(remaining_input_mask)
    remaining_input_mask_binary[remaining_input_mask_binary != 0] = 1
    remaining_input_mask_binary = remaining_input_mask_binary.astype(int)
    remaining_input_mask_binary, removed_voxels = \
        img_geom_utils.remove_badly_connected_voxels(remaining_input_mask_binary)

    remaining_input_mask = remaining_input_mask * remaining_input_mask_binary
    del remaining_input_mask_binary, removed_voxels

    # compute BMC (input and output) correction factor
    input_bmc = np.sum(remaining_input_mask) * (in_spacing[0] * in_spacing[1] * in_spacing[2])
    fe_bmc = np.sum(fe_bmd_mask) * (fe_spacing[0] * fe_spacing[1] * fe_spacing[2])

    bmc_correction_factor = input_bmc / fe_bmc

    return bmc_correction_factor, remaining_input_mask


def create_voxelmesh(mesh_file_name, voxelmodel_mask, voxelmodel_bvtv, voxel_size, flags, boundary_els=None):
    """

    :param mesh_file_name:
    :param model_array:
    :param voxel_size:
    :return:
    """
    print('Generating voxel mesh.. ')

    # dimensions of the arrays that are used
    dimensions = np.shape(voxelmodel_mask)
    nx = dimensions[0]
    ny = dimensions[1]
    nz = dimensions[2]

    # Create an array for the nodes
    mesh_nodes = np.zeros((nx + 1, ny + 1, nz + 1), np.float32)
    boundary_nodes = np.zeros((nx + 1, ny + 1, nz + 1)).astype(int)
    # arbitrary_factor = 1.0 / 8.0
    arbitrary_factor = 2

    # Loop over voxel values and distribute values (!=0) to mesh nodes, where the nodes are part of an element
    for x in range(0, nx):
        for y in range(0, ny):
            for z in range(0, nz):
                mesh_nodes[x, y, z] = mesh_nodes[x, y, z] + voxelmodel_mask[x, y, z] * arbitrary_factor
                mesh_nodes[x, y, z + 1] = mesh_nodes[x, y, z + 1] + voxelmodel_mask[x, y, z] * arbitrary_factor
                mesh_nodes[x, y + 1, z] = mesh_nodes[x, y + 1, z] + voxelmodel_mask[x, y, z] * arbitrary_factor
                mesh_nodes[x, y + 1, z + 1] = mesh_nodes[x, y + 1, z + 1] + voxelmodel_mask[x, y, z] * arbitrary_factor
                mesh_nodes[x + 1, y, z] = mesh_nodes[x + 1, y, z] + voxelmodel_mask[x, y, z] * arbitrary_factor
                mesh_nodes[x + 1, y, z + 1] = mesh_nodes[x + 1, y, z + 1] + voxelmodel_mask[x, y, z] * arbitrary_factor
                mesh_nodes[x + 1, y + 1, z] = mesh_nodes[x + 1, y + 1, z] + voxelmodel_mask[x, y, z] * arbitrary_factor
                mesh_nodes[x + 1, y + 1, z + 1] = mesh_nodes[x + 1, y + 1, z + 1] + voxelmodel_mask[
                    x, y, z] * arbitrary_factor

                if boundary_els is not None:
                    boundary_nodes[x, y, z] = boundary_nodes[x, y, z] + boundary_els[x, y, z] * \
                                              arbitrary_factor
                    boundary_nodes[x + 1, y, z] = boundary_nodes[x + 1, y, z] + boundary_els[x, y, z] * \
                                                  arbitrary_factor
                    boundary_nodes[x, y + 1, z] = boundary_nodes[x, y + 1, z] + boundary_els[x, y, z] * \
                                                  arbitrary_factor
                    boundary_nodes[x + 1, y + 1, z] = boundary_nodes[x + 1, y + 1, z] + boundary_els[x, y, z] * \
                                                      arbitrary_factor

    # -----------------------
    # Identify the nodes of the FE mesh (mesh_nodes != 0). Write out the nodes and create node sets for BCs.
    # -----------------------
    nodes = {}
    node_sets = {"Lateral": [],
                 "Medial": [],
                 "Superior": [],
                 "Inferior": []}

    nodal_indexes = np.argwhere(mesh_nodes != 0)
    countnode = 0

    sup = int(nz)
    inf = int(0)
    med = int(nx)
    lat = int(0)

    for index_set in nodal_indexes:
        countnode += 1
        # create node object and add to dictionary containing all the nodes
        nodes[countnode] = node(countnode, index_set[0] * voxel_size, index_set[1] * voxel_size,
                                index_set[2] * voxel_size)
        # set the element in mesh_node array to node nbr. for element definition
        mesh_nodes[index_set[0], index_set[1], index_set[2]] = countnode
        # write specific nodes to node sets
        if int(index_set[2]) == sup:
            node_sets["Superior"].append(countnode)

        if boundary_els is not None:
            if boundary_nodes[index_set[0], index_set[1], index_set[2]] != 0:
                node_sets["Inferior"].append(countnode)
        else:
            if int(index_set[2]) == inf:
                node_sets["Inferior"].append(countnode)

        if int(index_set[0]) == lat:
            node_sets["Lateral"].append(countnode)

        # medial nodes should be on the height of the head and not at the distal end of the femur
        if int(index_set[0]) == med:
            if int(index_set[2]) >= 10:
                node_sets["Medial"].append(countnode)

    # if medial node set is still empty, look one layer lower
    med_reduced = med
    while not node_sets["Medial"]:
        countnode = 0
        med_reduced -= 1
        for index_set in nodal_indexes:
            countnode += 1

            if int(index_set[0]) == med_reduced:
                if int(index_set[2]) >= 10:
                    node_sets["Medial"].append(countnode)

    # -----------------------
    # Write out elements (voxel_model != 0). Create the elements, add nodes (node numbering!!!) and create element sets.
    # -----------------------
    elems = {}
    elem_sets = {"PU": [],
                 "Steel": [],
                 "Bone": []}

    el_indexes = np.argwhere(voxelmodel_mask != 0)
    countel = 0

    for index_set in el_indexes:
        countel += 1
        x = index_set[0]
        y = index_set[1]
        z = index_set[2]
        # node numbering according to Abaqus convention for C3D8 elements
        el_nodes = [mesh_nodes[x, y, z], mesh_nodes[x + 1, y, z], mesh_nodes[x + 1, y + 1, z],
                    mesh_nodes[x, y + 1, z], mesh_nodes[x, y, z + 1], mesh_nodes[x + 1, y, z + 1],
                    mesh_nodes[x + 1, y + 1, z + 1], mesh_nodes[x, y + 1, z + 1]]
        elems[countel] = element(countel, el_nodes, "C3D8")
        # create the sets of elements for the different material types in the mesh
        if voxelmodel_mask[x, y, z] == 1:
            elem_sets["Bone"].append(countel)
            elems[countel].set_mat(voxelmodel_bvtv[x, y, z])
        elif voxelmodel_mask[x, y, z] == 4:
            elem_sets["PU"].append(countel)
        elif voxelmodel_mask[x, y, z] == 9:
            elem_sets["Steel"].append(countel)

    # Ouput Abaqus mesh input file
    f = open(mesh_file_name, 'w')
    f.write('**************************************************************************\n')
    f.write('** Voxel mesh\n')
    f.write('**\n')
    #
    # Write nodes and nodesets
    f.write('*NODE\n')
    for no in nodes:
        node_id = nodes[no].get_id()
        coordinates = nodes[no].get_coord()
        f.write('%12d,%12.3f,%12.3f,%12.3f\n' % (node_id, coordinates[0], coordinates[1], coordinates[2]))

    f.write('**\n')
    f.write('*NSET, NSET=SUP_NODES\n')
    counter = 0
    for no in node_sets['Superior']:
        counter += 1
        if counter % 16 == 0:
            f.write('\n')
        f.write('%d,' % no)

    f.write('\n')
    f.write('**\n')
    f.write('*NSET, NSET=INF_NODES\n')
    counter = 0
    for no in node_sets['Inferior']:
        counter += 1
        if counter % 16 == 0:
            f.write('\n')
        f.write('%d,' % no)

    f.write('\n')
    f.write('**\n')
    f.write('*NSET, NSET=LAT_NODES\n')
    counter = 0
    for no in node_sets['Lateral']:
        counter += 1
        if counter % 16 == 0:
            f.write('\n')
        f.write('%d,' % no)

    f.write('\n')
    f.write('**\n')
    f.write('*NSET, NSET=MED_NODES\n')
    counter = 0
    for no in node_sets['Medial']:
        counter += 1
        if counter % 16 == 0:
            f.write('\n')
        f.write('%d,' % no)
    f.write('\n')
    f.write('*SURFACE, NAME=REF_SURFACE_FALL, TYPE=NODE\n')
    f.write('MED_NODES\n')
    f.write('\n')
    f.write('*SURFACE, NAME=REF_SURFACE_STANCE, TYPE=NODE\n')
    f.write('SUP_NODES\n')
    f.write('**\n')

    # Write elements and materials to input file
    print(' ... Write elements and materials')
    # Loop over mesh and write out PU elements
    countelset = 1
    f.write('**\n')
    f.write('*ELEMENT, TYPE=C3D8, ELSET=SET%s\n' % (str(countelset)))
    for elem_no in elem_sets["PU"]:
        el = elems[elem_no]
        elem_nodes = el.get_nodes()
        f.write('%d,%d,%d,%d,%d,%d,%d,%d,%d\n' % (elem_no, elem_nodes[0], elem_nodes[1], elem_nodes[2], elem_nodes[3],
                                                  elem_nodes[4], elem_nodes[5], elem_nodes[6], elem_nodes[7]))

    f.write('**\n')
    f.write('*SOLID SECTION, ELSET=SET%s, MATERIAL=MATSET%s\n' % (str(countelset), str(countelset)))
    f.write('      1.\n')
    f.write('*MATERIAL, NAME=MATSET%s\n' % (str(countelset)))
    f.write('*ELASTIC, TYPE=ISO\n')
    f.write('1360, 0.3\n')
    f.write('**\n')

    # Loop over mesh and write out Steel elements
    countelset += 1
    f.write('**\n')
    f.write('*ELEMENT, TYPE=C3D8, ELSET=SET%s\n' % (str(countelset)))
    for elem_no in elem_sets["Steel"]:
        el = elems[elem_no]
        elem_nodes = el.get_nodes()
        f.write('%d,%d,%d,%d,%d,%d,%d,%d,%d\n' % (elem_no, elem_nodes[0], elem_nodes[1], elem_nodes[2], elem_nodes[3],
                                                  elem_nodes[4], elem_nodes[5], elem_nodes[6], elem_nodes[7]))

    f.write('**\n')
    f.write('*SOLID SECTION, ELSET=SET%s, MATERIAL=MATSET%s\n' % (str(countelset), str(countelset)))
    f.write('      1.\n')
    f.write('*MATERIAL, NAME=MATSET%s\n' % (str(countelset)))
    f.write('*ELASTIC, TYPE=ISO\n')
    f.write('210000,0.3\n')
    f.write('**\n')
    #
    # Write out elements that represent bone
    for elem_no in elem_sets["Bone"]:
        countelset += 1
        f.write('**\n')
        f.write('*ELEMENT, TYPE=C3D8, ELSET=SET%s\n' % (str(countelset)))
        el = elems[elem_no]
        elem_nodes = el.get_nodes()
        bvtv = el.get_mat()
        f.write('%d,%d,%d,%d,%d,%d,%d,%d,%d\n' % (elem_no, elem_nodes[0], elem_nodes[1], elem_nodes[2], elem_nodes[3],
                                                  elem_nodes[4], elem_nodes[5], elem_nodes[6], elem_nodes[7]))
        f.write('**\n')
        f.write('*SOLID SECTION, ELSET=SET%s, MATERIAL=MATSET%s, ORIENTATION=ORIENT%s\n' % (
            str(countelset), str(countelset), str(countelset)))
        f.write('1.\n')
        f.write('*MATERIAL,NAME=MATSET%s\n' % str(countelset))
        f.write('*USER MATERIAL,CONSTANTS=8,UNSYMM,TYPE=MECHANICAL\n')
        # To use UMAT quadric file (UMAT_QUADRIC_PRIMAL_rincon_hardening.f)
        f.write('0, %f, 1., 1., 1., %d, %d, %d\n' % (bvtv, flags["DENSFL"], flags["VISCFL"], flags["PYFL"]))
        f.write('*DEPVAR\n')
        f.write('22,\n')
        f.write('*ORIENTATION,NAME=ORIENT%s\n' % (str(countelset)))
        f.write('1.,0.,0., 0.,1.,0.\n')
        f.write('1, 0.\n')

    f.close()

    nbr_bone_elems = len(elem_sets["Bone"])
    return countel, nbr_bone_elems, countnode


def create_maininput(input_file_name, mesh_file, config, disp, reference_node):
    """

    :param input_file_name:
    :param mesh_file:
    :param config:
    :param disp:
    :param reference_node:
    :return:
    """
    print('\n Generating ABAQUS main input file.. ')
    f = open(input_file_name, 'w')
    f.write('*HEADING\n')
    f.write('*****************************************\n')
    f.write('*****************************************\n')
    f.write('** Mesh\n')
    # Define fabric model
    f.write('*INCLUDE,input=%s\n' % mesh_file)
    #
    f.write('**\n')
    f.write('*****************************************\n')
    f.write('*****************************************\n')
    #
    ref_x = reference_node[0]
    ref_y = reference_node[1]
    ref_z = reference_node[2]

    # boundary conditions and loading
    # stance configuration
    if config["exp_config_image"] == "stance":
        f.write('**nodes on the bottom fixed\n')
        f.write('*BOUNDARY, TYPE=DISPLACEMENT\n')
        #
        f.write('INF_NODES, 1, 3, 0\n')
        #
        f.write('**creation of driving node\n')
        f.write('*NODE\n')
        f.write('10000000, %s, %s, %s\n' % (ref_x, ref_y, ref_z))
        f.write('*NSET, NSET=DRIVING_NODE\n')
        f.write('10000000\n')
        f.write('**link the nodes on the top to the driving node\n')
        f.write('*COUPLING, CONSTRAINT NAME=DRIVING_DISPLACEMENT, REF NODE=DRIVING_NODE,\n')
        f.write('SURFACE=REF_SURFACE_STANCE\n')
        f.write('*KINEMATIC\n')
        f.write('**\n')
        f.write('*****************************************\n')
        f.write('*****************************************\n')
        f.write('**loading: free to rotate and translate in plane, compression\n')
        f.write('*STEP,AMPLITUDE=RAMP,UNSYMM=YES,INC=1000,NLGEOM=YES\n')
        f.write('*STATIC\n')
        f.write('0.05, 1., 5e-04, 0.05\n')
        f.write('*BOUNDARY, TYPE=DISPLACEMENT\n')
        f.write('DRIVING_NODE, 3, 3, %s\n' % disp)
        f.write('**writing the output in the main odb and in the history\n')
        f.write('*OUTPUT,FIELD\n')
        if config["save_field_output"] is not None:
            f.write('*ELEMENT OUTPUT, POSITION=CENTROIDAL\n')
        else:
            f.write('*ELEMENT OUTPUT\n')
        f.write('S,\n')
        f.write('E,\n')
        f.write('SDV,\n')
        f.write('*NODE OUTPUT\n')
        f.write('U, \n')
        f.write('RF,\n')
        f.write('*OUTPUT, HISTORY\n')
        f.write('*NODE OUTPUT, NSET=DRIVING_NODE\n')
        f.write('U, \n')
        f.write('RF,\n')
        f.write('*NODE PRINT, NSET=DRIVING_NODE, FREQUENCY=1, SUMMARY=NO \n')
        f.write('U,\n')
        f.write('RF\n')
        f.write('*END STEP\n')

    # fall configuration
    elif config["exp_config_image"] == "fall":
        f.write('**nodes on the bottom fixed\n')
        f.write('*BOUNDARY, TYPE=DISPLACEMENT\n')
        f.write('INF_NODES, 1, 3, 0\n')
        f.write('LAT_NODES, 1, 1, 0\n')
        f.write('**Creation of driving node\n')
        f.write('*NODE\n')
        f.write('10000000, %s, %s, %s\n' % (ref_x, ref_y, ref_z))
        f.write('*NSET, NSET=DRIVING_NODE\n')
        f.write('10000000\n')
        f.write('**link the nodes on the top to the driving node\n')
        f.write('*COUPLING, CONSTRAINT NAME=DRIVING_DISPLACEMENT, REF NODE=DRIVING_NODE,\n')
        f.write('SURFACE=REF_SURFACE_FALL\n')
        f.write('*KINEMATIC\n')
        f.write('**\n')
        f.write('*****************************************\n')
        f.write('*****************************************\n')
        f.write('**loading: free to rotate and translate in plane, compression\n')
        f.write('*STEP,AMPLITUDE=RAMP,UNSYMM=YES,INC=1000,NLGEOM=YES\n')
        f.write('*STATIC\n')
        f.write('0.05, 1., 5e-04, 0.05\n')
        f.write('*BOUNDARY, TYPE=DISPLACEMENT\n')
        f.write('DRIVING_NODE, 1, 1, %s \n' % disp)
        f.write('**writing the output in the main odb and in the history\n')
        f.write('*OUTPUT,FIELD\n')
        if config["save_field_output"] is not None:
            f.write('*ELEMENT OUTPUT, POSITION=CENTROIDAL\n')
        else:
            f.write('*ELEMENT OUTPUT\n')
        f.write('S,  \n')
        f.write('E,\n')
        f.write('SDV,\n')
        f.write('*NODE OUTPUT\n')
        f.write('U,\n')
        f.write('RF, \n')
        f.write('*OUTPUT, HISTORY\n')
        f.write('*NODE OUTPUT, NSET=DRIVING_NODE\n')
        f.write('U, \n')
        f.write('RF, \n')
        f.write('*NODE PRINT, NSET=DRIVING_NODE, FREQUENCY=1, SUMMARY=NO \n')
        f.write('U,\n')
        f.write('RF\n')
        f.write('*END STEP\n')
    f.close()


def modify_voxel_mesh(old_mesh, new_mesh, new_flags):
    """
    Creates an abaqus mesh file as input and writes it with other flags for the material definition.
    Parameters
    ----------
    old_mesh :
    new_mesh :
    new_flags :

    Returns
    -------

    """
    # read the existing mesh file
    with open(old_mesh, "r") as infile:
        lines = infile.readlines()

    f = open(new_mesh, 'w')  # create new mesh file

    # rewrite the file
    i = 0
    while i < len(lines):
        if lines[i].find("CONSTANTS=8") == -1:  # rewrite the lines as they are
            f.write(lines[i])
            i += 1
        else:
            f.write(lines[i] + "\n")

            dens = str(new_flags["DENSFL"])
            visc = str(new_flags["VISCFL"])
            py = str(new_flags["PYFL"])

            # lines that contain the values of the three flags, rewrite with changed values
            lines[i + 1] = lines[i + 1][:-8] + (dens + ", " + visc + ", " + py + "\n")
            f.write(lines[i + 1])
            i += 2

    f.close()


def modify_maininput_file(old_input_file, new_input_file, new_mesh_file):
    """
    Rewrites an input file to a new file with reference to another mesh file.
    Parameters
    ----------
    input_file :
    new_input_file :

    Returns
    -------

    """
    with open(old_input_file, "r") as infile:
        lines = infile.readlines()

    f = open(new_input_file, 'w')  # create new input file
    i = 0
    while i < len(lines):  # write new input file line by line
        if lines[i].find("*INCLUDE,input=") == -1:
            f.write(lines[i] + "\n")
        else:
            f.write("*INCLUDE,input=" + new_mesh_file + "\n")  # write line with new mesh file

        i += 1
    f.close()


def compute_cam_angles(v_neck):
    cam_angle1 = np.arccos(np.dot(np.array([1, 0]), img_geom_utils.normi(v_neck[0:2]))) / np.pi * 180
    cam_angle2 = np.arccos(np.dot(np.array([1, 0]), img_geom_utils.normi(
        np.array([v_neck[1], -v_neck[0]])))) / np.pi * 180

    return cam_angle1, cam_angle2
