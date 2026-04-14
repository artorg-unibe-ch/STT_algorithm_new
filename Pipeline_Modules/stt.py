"""
Created on 11.08.2022
@author: Paula Cameron, Christina Wapp
{
Contains functions necessary to compute the soft tissue thickness
and muscle thickness from the coordinates of the greater trochanter
Updated March 2026
}
"""
import cc3d
import numpy as np

from Pipeline_Modules.lib import *
from skimage import morphology
from Pipeline_Modules import img_geom_utils, femur_pre
from Pipeline_Modules import femur_pre


def detect_stt_landmarks(img_seg_np, split_x, out_spacing, side):
    """
    Detects the three major landmarks of the femur defining its coordinate system using the segmented femur mask
    :param img_seg_np: segmented image in np array form
    :param split_x: index of middle of the img_seg_np
    :param out_spacing: image property
    :param side: left or right femur
    :return intersec_orig:
    :return v_fea: vector along fall configuration fea test direction
    :return head_center_orig:
    :return intersection_neck_shaft_new_final_orig:
    :return stt_status: 1 = landmarks detected, 0 = no landmarks detected
    """
    stt_status = 1
    diaphysis_angle = 10

    std_layer = int(50 / out_spacing[0])

    # standardize resampled image: flip left femur to right femur, add isotropic "padding" layer
    bmd_array, mask_array, metadata_dict, im_flipped = femur_pre.standardize_image(img_seg_np,
                                                                                   None, add_layer=std_layer)

    # compute bounding boxes
    bbox_in = img_geom_utils.bounding_box_start_end_np(img_seg_np)
    bbox_out = img_geom_utils.bounding_box_start_end_np(mask_array)

    del bmd_array, im_flipped

    # main translation
    cs_translation = np.array([bbox_in[0], bbox_in[2], bbox_in[4]]) - np.array(
        [bbox_out[0], bbox_out[2], bbox_out[4]])

    # translate origin to 0,0,0
    cs_translation_0 = (-1) * np.array([bbox_out[0], bbox_out[2], bbox_out[4]])

    # size of bounding box out
    bbox_extent = np.array(
        [bbox_out[1] - bbox_out[0], bbox_out[3] - bbox_out[2], bbox_out[5] - bbox_out[4]])

    # absolute origin of input bounding box
    cs_translation_1 = np.array([bbox_in[0], bbox_in[2], bbox_in[4]])

    # __________________________________________________________________________________________________________________
    # Detect coordinate system of the proximal femur
    # _________________________________________________________________________________________________________________
    head_center, head_radius, v_dia, v_neck, intersection_neck_shaft, \
        neck_shaft_angle, d_point, \
        head_z, head_ml, head_ap, head_pa, v_dia_in, v_neck_in, new_d_point \
        = femur_pre.detect_coordinate_system(mask_array, std_layer, out_spacing)

    # __________________________________________________________________________________________________________________
    # Detect LT and GT and correct coordinate axes
    # __________________________________________________________________________________________________________________
    try:
        v_dia_new_final, v_neck_new_final, intersection_neck_shaft_new_final, db_point, lt_mask_orig_orientation, \
            gt_mask_medial_orig_orientation, gt_mask_lateral_orig_orientation, lt_peak, lt_peak_alt, cog_lt_peak, cog_lt_mid, \
            new_neck_shaft_angle, neck_lt_angle, dia_flag, dl, new_base_point_orig_config, lt_peak_axis, V_prox_femur = \
            femur_pre.detect_landmarks_and_correct_coordinate_system(mask_array, head_center, head_radius,
                                                                     v_dia, v_neck, intersection_neck_shaft, d_point,
                                                                     v_neck_in, add_layer=std_layer,
                                                                     spacing=out_spacing)
    except: #  set stt = 0 in case not running, calculate
        stt_status = 0
        print(f"Except clause: status = {stt_status}")
        if side == 0:  # right femur
            return 0, 0, cs_translation + head_center, 0, stt_status
        else:  # left femur
            head_center_orig = np.zeros(3)
            head_center_orig[1:] = (head_center + cs_translation_0 + cs_translation_1)[1:]
            head_center_orig[0] = bbox_extent[0] - (head_center[0] + cs_translation_0[0]) + cs_translation_1[0]
            head_center_orig[0] += split_x
            return 0, 0, head_center_orig, 0, stt_status

    # keep v_dia_new_final, v_neck_new_final
    del v_dia, v_neck, intersection_neck_shaft, neck_shaft_angle, d_point, head_z, head_ml, head_ap, head_pa, v_dia_in, v_neck_in, \
        new_d_point, lt_peak_alt, cog_lt_peak, cog_lt_mid, new_neck_shaft_angle, neck_lt_angle, dia_flag, dl, new_base_point_orig_config, \
        lt_peak_axis, V_prox_femur

    v_antero_post = np.cross(v_dia_new_final, v_neck_new_final)  # third axis (orthogonal on v_dia and v_neck)

    rotation_angle = 90 + diaphysis_angle
    rot_matrix = img_geom_utils.rotation_arbitrary_axis(v_antero_post, img_geom_utils.deg2rad(
        rotation_angle))  # rotation around v_antero_post by 100°
    v_dia_transformed = np.dot(rot_matrix, v_dia_new_final)

    rotation_angle = 270 + diaphysis_angle
    rot_matrix = img_geom_utils.rotation_arbitrary_axis(v_antero_post, img_geom_utils.deg2rad(
        rotation_angle))  # rotation around v_antero_post by 100°
    v_fea = np.dot(rot_matrix, v_dia_new_final)

    start_distance = 50 / out_spacing[0]
    intersec = img_geom_utils.disc_collision_point(mask_array, v_dia_transformed,
                                                   intersection_neck_shaft_new_final - start_distance * v_dia_transformed)

    # __________________________________________________________________________________________________________________
    # Change of coordinate system - femur 1
    # __________________________________________________________________________________________________________________

    # apply coordinate system translation
    if side == 0:
        print()
        print("RIGHT")
        head_center_orig = cs_translation + head_center
        intersection_neck_shaft_new_final_orig = cs_translation + intersection_neck_shaft_new_final
        intersec_orig = cs_translation + intersec

    # __________________________________________________________________________________________________________________
    # Change of coordinate system - femur 2
    # __________________________________________________________________________________________________________________

    # apply coordinate system translation for components in y and z direction
    # translation plus symmetry (left femur transformed to right femur) for the x component
    if side == 1:
        print()
        print("LEFT")
        head_center_orig = np.zeros(3)
        head_center_orig[1:] = (head_center + cs_translation_0 + cs_translation_1)[1:]
        head_center_orig[0] = bbox_extent[0] - (head_center[0] + cs_translation_0[0]) + cs_translation_1[0]

        intersection_neck_shaft_new_final_orig = np.zeros(3)
        intersection_neck_shaft_new_final_orig[1:] = (intersection_neck_shaft_new_final + cs_translation_0
                                                      + cs_translation_1)[1:]
        intersection_neck_shaft_new_final_orig[0] = bbox_extent[0] - (
                    intersection_neck_shaft_new_final[0] + cs_translation_0[0]) + cs_translation_1[0]

        intersec_orig = np.zeros(3)
        intersec_orig[1:] = (intersec + cs_translation_0 + cs_translation_1)[1:]
        intersec_orig[0] = bbox_extent[0] - (intersec[0] + cs_translation_0[0]) + cs_translation_1[0]

        v_fea[0] = -v_fea[0]

        head_center_orig[0] += split_x
        intersection_neck_shaft_new_final_orig[0] += split_x
        intersec_orig[0] += split_x

    return intersec_orig, v_fea, head_center_orig, intersection_neck_shaft_new_final_orig, stt_status


def hu_threshold_calibration(calibration_status, calibration_equation):
    """
    Function to adjust HU threshold if images were calibrated.

    :param calibration_status: 'calibrated' or 'uncalibrated'
    :param calibration_equation: dictionary with slope and intercept of calibration equation
    :return thresholds_dict: dictionary with adjusted hu thresholds according to calibration
    """
    if calibration_status == 'uncalibrated':
        thresholds_dict = {'adipose_low': -190,
                           'adipose_high': -30,
                           'muscle_low': -29,
                           'muscle_high': 151,
                           'musclebone_low': 151,
                           'musclebone_high': 400,
                           'bone_low': 400,
                           'air_high': -300}

    if calibration_status == 'calibrated':
        thresholds_dict = {'adipose_low': -190 * calibration_equation['slope'] + calibration_equation['intercept'],
                           'adipose_high': -30 * calibration_equation['slope'] + calibration_equation['intercept'],
                           'muscle_low': -29 * calibration_equation['slope'] + calibration_equation['intercept'],
                           'muscle_high': 151 * calibration_equation['slope'] + calibration_equation['intercept'],
                           'musclebone_low': 151 * calibration_equation['slope'] + calibration_equation['intercept'],
                           'musclebone_high': 400 * calibration_equation['slope'] + calibration_equation['intercept'],
                           'bone_low': 400 * calibration_equation['slope'] + calibration_equation['intercept'],
                           'air_high': -300 * calibration_equation['slope'] + calibration_equation['intercept']}

    return thresholds_dict


def get_stt(right_gt, left_gt, img, pix_dim, v_fea_1, v_fea_2, size_mm_y, size_mm_z, status,
            hu_thresholds):
    """
    This function measures the soft tissue thickness on both sides, defined as the distance from the greater trochanter
    to the soft tissue air boundary along the v_fea vector.

    :param right_gt: coordinates of greater trochanter right
    :param left_gt: coordinates of greater trochanter left
    :param img: numpy array of hip CT scan (not resampled)
    :param pix_dim: numpy array of pixel dimensions (mm per pixel in each direction)
    :param v_fea_1: fea vector right
    :param v_fea_2: fea vector left
    :param size_mm_y: size of plane around GT checking soft tissue air boundary in mm
    :param size_mm_z: size of plane around GT checking soft tissue air boundary in mm
    :param status: status whether coordinate system of GT was found (0 = no, 1 = yes)
    :param hu_thresholds: HU threshold values for different tissue types
    :return d_r: STT right
    :return d_l: STT left
    :return right: array with [GT, min_point, fea_point]
    :return left: array with [GT, min_point, fea_point]
    :return verts: vertices from measure_marching_cubes
    """

    # create binary mask with -700HU as threshold to seperate air/soft tissue and bone
    binary = (img > hu_thresholds['air_high']).astype(int)

    binary_bool = binary.astype(bool)
    binary_bool = morphology.remove_small_holes(binary_bool, 40000)
    # closed = morphology.binary_closing(binary_bool, morphology.ball(5))
    closed = binary_bool.astype(int)

    # keep only largest object in image (the body)
    closed_labeled = measure.label(closed)
    labels, counts = np.unique(closed_labeled, return_counts=True)
    sorted_indices = np.argsort(counts)
    closed = np.zeros(np.shape(closed))
    closed[closed_labeled == sorted_indices[-2]] = 1  # largest element is background, second largest body
    del closed_labeled

    # create triangular mesh
    verts, faces, normals, values = measure.marching_cubes(closed, spacing=pix_dim)

    larger_flag = np.zeros(2)
    n_stab_1 = np.zeros(2)

    # -- right --
    if status[0] == 1:  # status defines whether coordinate system was detected
        side = 'right'
        d_r, right, larger_flag[0], n_stab_points, n_stab_1[0], stab_1_lp_r = find_tissue_air_surf(verts, right_gt, v_fea_1, size_mm_y, size_mm_z, side, pix_dim)

    # set values that are required for function output to na if coordinate system was not detected
    else:
        d_r = np.nan
        right = np.nan
        larger_flag[0] = np.nan
        stab_1_lp_r = np.nan

    # -- left --
    if status[1] == 1:
        side = 'left'
        d_l, left, larger_flag[1], n_stab_points, n_stab_1[1], stab_1_lp_l = find_tissue_air_surf(verts, left_gt, v_fea_2, size_mm_y, size_mm_z, side, pix_dim)

    # set values that are required for function output to na if coordinate system was not detected
    else:
        d_l = np.nan
        left = np.nan
        larger_flag[1] = np.nan
        stab_1_lp_l = np.nan

    return d_r, d_l, right, left, verts, faces, status, larger_flag, n_stab_points, n_stab_1, stab_1_lp_r, stab_1_lp_l


def find_tissue_air_surf(vertices, gt, v_fea, size_mm_y, size_mm_z, side, pix_dim):
    """
    This function finds the tissue air boundary along the v_fea vector or the minimal distance, defined as the
    minimum distance within 0 to 20 degrees anterior of FEA vector. It also checks whether the soft tissue air boundary
    (STAB) is present or whether the object is larger than the scanning region. This is done in a plane around the GT
    with a 1% criterion (if 1 % of the voxels in this plane have no STAB, the object is larger than the scanning region).


    :param vertices: vertices from measure_marching_cubes
    :param gt: point of greater trochanter [x,y,z]
    :param v_fea: vector from fea
    :param size_mm_y: size of plane around GT checking soft tissue air boundary in mm
    :param size_mm_z: size of plane around GT checking soft tissue air boundary in mm
    :param side: body side
    :param pix_dim: pixel dimensions of ct image
    :return d: soft tissue thickness
    :return gt: point of greater trochanter
    :return min_point: point on STAB defining minimum distance from GT to STAB within 0 to 20 degrees anterior of FEA vector
    :return fea_point: point of soft tissue air boundary along v_fea
    :return larger_flag: body is larger than scanning region
    :return n_stab_points: number of voxels in plane checking presence pf STAB
    :return n_stab_1: number of voxels without STAB found
    :return stab_1_lp: coordinates of voxels without STAB found (from this point a line could be plotted to illustrate it)
    """

    size_elements_y = round(size_mm_y/pix_dim[1])
    size_elements_z = round(size_mm_z/pix_dim[2])

    gt_x = int(gt[0]) # fix x using x coordinate of GT
    gt_start_y = int(gt[1] - size_mm_y/2)  # center GT within y and z
    gt_start_z = int(gt[2] - size_mm_z/2)

    stab_1_lp = []  # array to safe points where no STAB (larger = 1) is present
    indices = []  # array to safe indices of STAB is present
    larger_flag_all = np.zeros((size_elements_z, size_elements_y), dtype=int)  # array for every voxel in the plane whether STAB is detected

    # loop through every voxel of the plane to check whether STAB exists at every voxel
    for i in range(size_elements_y):
        current_y = gt_start_y + i * pix_dim[1]

        for j in range(size_elements_z):
            int_verts = np.array(vertices).astype(int)

            current_z = gt_start_z + j * pix_dim[2]

            # coronal indices
            cor_indices = np.where(int_verts[:, 1] == int(current_y))[0]  # select vertices where vertex_y == gt_y
            int_verts = int_verts[cor_indices]

            # axial (depth) indices
            ax_indices = np.where(int_verts[:, 2] == int(current_z))[0]
            int_verts = int_verts[ax_indices]

            base_indices = cor_indices[ax_indices]

            if side == 'right':
                # axial indices right (x decreasing)
                side_indices = np.where(int_verts[:, 0] <= int(gt_x)+50)[0]  # shift point by 5cm in case some points of STAB lay more medial than gt[x]

            if side == 'left':
                # axial indices left (x increasing)
                side_indices = np.where(int_verts[:, 0] >= int(gt_x)-50)[0]  # shift point by 5cm in case some points of STAB lay more medial than gt[x]

            int_verts = int_verts[side_indices]
            indices = base_indices[side_indices]

            # if there is no point on soft tissue-air boundary directly lateral to GT,
            # then the subject is larger than the scanning region
            if len(indices) == 0:
                larger_flag_all[j, i] = int(1)
                stab_1_lp.append([gt_x, current_y, current_z])  # track which line points are not covered by mesh for plotting which points are

    stab_1_lp = np.array(stab_1_lp)
    n_stab_points = size_elements_y * size_elements_z  # total number of elements in the plane
    n_stab_1 = np.sum(larger_flag_all == 1)  # number of points without STAB

    larger_flag = 0
    if n_stab_1 >= n_stab_points/100:  # 1% criteria
        larger_flag = 1
        print()
        print(f"subject {side} larger than scanning region")
        d = np.nan
        fea_point = [np.nan, np.nan, np.nan]
        min_point = [np.nan, np.nan, np.nan]

    else:
        # set maximum distance to half of image size in pixels
        # among the set of points on soft tissue-air boundary,
        # find the point that is at a minimum distance from the GT
        # for subjects larger than scanning region d_hor will not be modified
        d_hor = 256
        hor_point = []

        for p in vertices[indices]:
            tmp_d = np.linalg.norm(gt - p)

            if tmp_d < d_hor:
                d_hor = tmp_d
                hor_point = p

        point_tree = spatial.cKDTree(vertices)
        closest_points = []

        # Find the sphere of points on the soft tissue-air boundary
        # Using the distance found in the previous step as the sphere radius
        d = int(d_hor) + 1

        while (len(closest_points) == 0) and (d < 1000):
            d += 1
            closest_points = point_tree.query_ball_point(gt, d)

        min_point = []
        abs_min_point = []
        fea_point = []

        d_abs_min = d
        min_chi_fea = 90

        for j in closest_points:
            point = vertices[j]
            d_min = np.linalg.norm(gt - point)

            v_stt = img_geom_utils.normi(point - gt)

            v_fea_sag_cor = img_geom_utils.normi([v_fea[0], v_fea[1]])
            v_fea_sag_ax = img_geom_utils.normi([v_fea[0], v_fea[2]])

            v_stt_sag_cor = img_geom_utils.normi([v_stt[0], v_stt[1]])
            v_stt_sag_ax = img_geom_utils.normi([v_stt[0], v_stt[2]])

            theta_fea = np.rad2deg(
                math.acos((v_fea_sag_cor[0] * v_stt_sag_cor[0] + v_fea_sag_cor[1] * v_stt_sag_cor[1])))
            phi_fea = np.rad2deg(math.acos(v_fea_sag_ax[0] * v_stt_sag_ax[0] + v_fea_sag_ax[1] * v_stt_sag_ax[1]))
            chi_fea = np.rad2deg(math.acos(v_fea[0] * v_stt[0] + v_fea[1] * v_stt[1] + v_fea[2] * v_stt[2]))

            if chi_fea <= min_chi_fea:
                min_chi_fea = chi_fea
                fea_point = point

            if d_min <= d and (np.abs(theta_fea) <= 20.0 and np.abs(phi_fea) <= 20.0) and v_fea[1] > v_stt[1]:
                d = d_min
                min_point = point

            if d_min <= d_abs_min:
                d_abs_min = d_min
                abs_min_point = point

        if len(min_point) == 0:
            if len(hor_point) == 0:
                min_point = abs_min_point
                d = d_abs_min
                print("no horizontal point & no min point within 20 degrees and anterior to v_fea")
            else:
                print("no min point")
                min_point = hor_point
                d = d_hor

    return d, np.array([gt, min_point, fea_point]), larger_flag, n_stab_points, n_stab_1, stab_1_lp


def get_muscle_point(point_1, point_2, blur_img, pix_dim, hu_thresholds):
    """
    This function assesses tissue type along the line of the GT to the soft tissue air boundary.
    Three different methods are measured:
    (1) Old line method: find the point where muscle chagnes to adipose tissue; assume that the rest then is adipose
    (2) New line method: assess every change along the line and calculate in the end the total percentage of all increments
    together
    (3) Parallel line methods: new line method with six lines around the line from the GT.
    The values of the seven lines are averaged

    :param point_1: coordinates of GT
    :param point_2: coordinates FEA point
    :param blur_img: gaussian filtered image, or normal image
    :param pix_dim: pixel dimension of CT scan
    :param hu_thresholds: HU thresholds for different tissue types
    :return d_m: calculate distance muscle (greater trochanter to transition) of old line method
    :return point_3: transition point muscle fat old line method
    :return med_filterd_hu: array with hu intensities among line of old line method
    :return col_arr: array of the six line methods with colors given tissue type for plotting
    :return dist_methods: array with the outputs of all different methods. detailed description can be found in the dictionary of
    the clinical soft tissue extraction function
    """

    p1_pix = np.divide(point_1, pix_dim)  # from mm to pixels to be able to work with nd.array
    p2_pix = np.divide(point_2, pix_dim)

    # --- Old line method (OLM)
    # Along single line from GT to surface, with increments calculated as percentage after the first change from
    # muscle to fat
    length = 200
    line = np.linspace(p1_pix, p2_pix, num=length, dtype=int)

    x = np.linspace(0, 1, length)
    y = []

    for index, point in enumerate(line):
        y.append(blur_img[point[0], point[1], point[2]])

    med_filtered_hu = signal.medfilt(y, kernel_size=7)

    # find point_3 defining transition point between muscle and fat
    neg = 0
    p3_pix = [0, 0, 0]
    for index, intensity in enumerate(med_filtered_hu):
        if neg == 0 and intensity < hu_thresholds['muscle_low']:
            neg = 1
            p3_pix = line[index]

    point_3 = np.multiply(p3_pix, pix_dim)  # convert from pixel back to physical space
    d_m = np.linalg.norm(point_1 - point_3)  # calculate distance muscle (greater trochanter to transition)

    dist_methods = []

    # --- New line method with small incremental additions
    transition_points, muscle_start = get_transition_point(point_1, point_2, blur_img, pix_dim, hu_thresholds)
    d_adipose = 0
    d_muscle = 0

    if len(transition_points) != 0:
        d = np.linalg.norm(np.array(transition_points[0][0:3]) - point_1)

        if muscle_start == 0:
            d_adipose += d
        else:
            d_muscle += d

        for idx in range(0, len(transition_points) - 1):

            d = np.linalg.norm(np.array(transition_points[idx][0:3]) - np.array(transition_points[idx + 1][0:3]))

            if transition_points[idx][3] == 1:
                d_adipose += d
            else:
                d_muscle += d

        d = np.linalg.norm(np.array(transition_points[-1][0:3]) - point_2)

        if transition_points[-1][3] == 1:
            d_adipose += d
        else:
            d_muscle += d

    dist_methods.append(d_muscle)
    dist_methods.append(d_adipose)

    # --- Parallel lines method (6 lines)
    cylinder_r = 5.0
    cylinder_r_mm = 5.0
    height = np.linalg.norm(p2_pix.reshape(3, 1) - p1_pix.reshape(3, 1))  # soft tissue thickness
    e = p2_pix.reshape(3, 1) - p1_pix.reshape(3, 1)  # vector between GT and STAB in pixel space
    e_mm = point_2.reshape(3, 1) - point_1.reshape(3, 1)  # vector between GT and STAB in physical space

    b_x = p1_pix[2] + 1
    b_y = p1_pix[1] + 1
    b_z = (-e[0] * b_x - e[1] * b_y) / e[2]  # perpendicular plane: e_0*x + e_1*y + e_2*z = 0
    b = np.array([b_x, b_y, b_z[0]])
    b = b.reshape(3, 1)
    b_unit = b / np.linalg.norm(b)

    b_x_mm = point_1[2] + 1
    b_y_mm = point_1[1] + 1
    b_z_mm = (-e_mm[0] * b_x_mm - e_mm[1] * b_y_mm) / e_mm[2]
    b_mm = np.array([b_x_mm, b_y_mm, b_z_mm[0]])
    b_mm = b_mm.reshape(3, 1)
    b_unit_mm = b_mm / np.linalg.norm(b_mm)

    a_mm = np.cross(e_mm, b_mm, axis=0)
    a_unit_mm = a_mm / np.linalg.norm(a_mm)

    a = np.cross(e, b, axis=0)
    a_unit = a / np.linalg.norm(a)

    d_soft_array = []
    d_muscle_array = []
    d_adipose_array = []
    d_SAT_array = []
    d_soft = 0
    d_adipose = 0
    d_muscle = 0
    d_SAT = 0
    d_muscle_prop = 0
    d_adipose_prop = 0

    # Middle line from GT to STAB
    new_cyl_points = []
    col_arr = []
    out_img = 0
    x_i = p1_pix.reshape(3, 1)

    line_i = []
    length_i = 0
    t = 0

    # check whether GT is in image and if yes, create new point with increment size t along e
    if int(x_i[0][0]) < blur_img.shape[0] and int(x_i[1][0]) < blur_img.shape[1] and int(
            x_i[2][0]) < blur_img.shape[2]:
        line_i.append(x_i)
        t = 1 / int(height)  # 1/n
        new_point = x_i + t * e

        # check whether new point is still in image
        if int(new_point[0][0]) < blur_img.shape[0] and int(new_point[1][0]) < blur_img.shape[1] and int(
                new_point[2][0]) < blur_img.shape[2]:
            line_i.append(new_point)
            length_i = np.linalg.norm(new_point - x_i)
        else:
            length_i = 0
            out_img = 1
    else:
        out_img = 1

    # as long as length_i is smaller than height (STT), and not outside of image, add points along es
    while length_i < height and out_img == 0:  # 100 is arbitrary
        t += 1 / int(height)
        new_point = x_i + t * e

        # always checking whether still inside the image
        if int(new_point[0][0]) < blur_img.shape[0] and int(new_point[1][0]) < blur_img.shape[1] and int(
                new_point[2][0]) < blur_img.shape[2]:
            line_i.append(new_point)
            length_i = np.linalg.norm(new_point - x_i)
        else:
            out_img = 1

    hu_array = []
    for j in range(0, len(line_i)):
        point = line_i[j].astype(int)
        hu = blur_img[point[0], point[1], point[2]][0]
        hu_array.append(hu)

        if hu_thresholds['adipose_low'] <= hu <= hu_thresholds['adipose_high']:  # adipose
            col_arr.append('skyblue')
        elif hu_thresholds['muscle_low'] <= hu <= hu_thresholds['muscle_high']:  # muscle
            col_arr.append('orange')
        elif hu_thresholds['musclebone_low'] < hu < hu_thresholds['musclebone_high']:  # between muscle and bone
            col_arr.append('#009e73')
        elif hu >= hu_thresholds['bone_low']:  # bone
            col_arr.append('blue')
        elif hu <= hu_thresholds['air_high']:  # air
            col_arr.append('black')
        else:  # other
            col_arr.append('#cc79a7')

        # calculate back to physical scale in mm
        new_cyl_points.append(np.multiply(pix_dim, line_i[j].reshape(1, 3))[0])

    # Plots
    """plt.figure()
    plt.title("Middle line")
    plt.axhline(y=-29, c='red')
    plt.axhline(y=-190, c='red')
    plt.axhline(y=151, c='red')
    plt.axhline(y=400, c='red')
    plt.plot(np.linspace(0, len(hu_array), len(hu_array)), hu_array)
    plt.show()"""
    # Plots - end

    # calculate percentage of tissue types
    if len(line_i) > 0:
        point_1_line = np.multiply(pix_dim, line_i[0].reshape(1, 3))[0]
        point_2_line = np.multiply(pix_dim, line_i[-1].reshape(1, 3))[0]

        d_soft, d_muscle, d_adipose, d_SAT = add_increments(point_1_line, point_2_line, blur_img, pix_dim, hu_thresholds)
        d_muscle_prop = d_muscle / d_soft
        d_adipose_prop = d_adipose / d_soft
        d_SAT_prop = d_SAT / d_soft

        d_soft_array.append(d_soft)
        d_muscle_array.append(d_muscle_prop)
        d_adipose_array.append(d_adipose_prop)
        d_SAT_array.append(d_SAT_prop)
    else:
        d_soft = np.nan
        d_muscle_prop = np.nan
        d_adipose_prop = np.nan
        d_SAT_prop = np.nan

    print("*******************************")
    print(f"Middle line")
    print(f"TSTT: {d_soft}")
    print(f"Muscle thickness: {d_muscle}, {d_muscle_prop}")
    print(f"Adipose thickness: {d_adipose}, {d_adipose_prop}")
    print(f"SAT thickness: {d_SAT}, {d_SAT_prop}")
    print()

    # parallel lines around GT
    N = 6  # Number of equidistant parallel lines
    for i in range(0, N):
        out_img = 0
        x_i = p1_pix.reshape(3, 1) + cylinder_r * np.cos(2 * np.pi * i / N) * b_unit + cylinder_r * np.sin(
            2 * np.pi * i / N) * a_unit

        x_i_mm = point_1.reshape(3, 1) + cylinder_r_mm * np.cos(2 * np.pi * i / N) * b_unit_mm + cylinder_r_mm * np.sin(
            2 * np.pi * i / N) * a_unit_mm

        x_i = np.divide(x_i_mm.reshape(1, 3), pix_dim)
        x_i = x_i.reshape(3, 1)

        line_i = []
        length_i = 0
        t = 0

        if int(x_i[0][0]) < blur_img.shape[0] and int(x_i[1][0]) < blur_img.shape[1] and int(
                x_i[2][0]) < blur_img.shape[2]:
            line_i.append(x_i)
            t = 1 / int(height)  # 1/n
            new_point = x_i + t * e

            if int(new_point[0][0]) < blur_img.shape[0] and int(new_point[1][0]) < blur_img.shape[1] and int(
                    new_point[2][0]) < blur_img.shape[2]:
                line_i.append(new_point)
                length_i = np.linalg.norm(new_point - x_i)
            else:
                length_i = 0
                out_img = 1
        else:
            out_img = 1

        while length_i < height and out_img == 0:  # 100 is arbitrary
            t += 1 / int(height)
            new_point = x_i + t * e

            if int(new_point[0][0]) < blur_img.shape[0] and int(new_point[1][0]) < blur_img.shape[1] and int(
                    new_point[2][0]) < blur_img.shape[2]:
                line_i.append(new_point)
                length_i = np.linalg.norm(new_point - x_i)
            else:
                out_img = 1

        hu_array = []
        for j in range(0, len(line_i)):
            point = line_i[j].astype(int)
            hu = blur_img[point[0], point[1], point[2]][0]
            hu_array.append(hu)

            if hu_thresholds['adipose_low'] <= hu <= hu_thresholds['adipose_high']:  # adipose
                col_arr.append('skyblue')
            elif hu_thresholds['muscle_low'] <= hu <= hu_thresholds['muscle_high']:  # muscle
                col_arr.append('orange')
            elif hu_thresholds['musclebone_low'] < hu < hu_thresholds['musclebone_high']:  # between muscle and bone
                col_arr.append('#009e73')
            elif hu >= hu_thresholds['bone_low']:  # bone
                col_arr.append('blue')
            elif hu <= hu_thresholds['air_high']:  # air
                col_arr.append('black')
            else:  # other
                col_arr.append('#cc79a7')

            new_cyl_points.append(np.multiply(pix_dim, line_i[j].reshape(1, 3))[0])

        # Plots
        # plt.figure()
        # plt.title("Line " + str(i))
        # plt.axhline(y=-29, c='red')
        # plt.axhline(y=-190, c='red')
        # plt.axhline(y=151, c='red')
        # plt.axhline(y=400, c='red')
        # plt.plot(np.linspace(0, len(hu_array), len(hu_array)), hu_array)
        # plt.show()
        # Plots - end


        # average of all lines
        if len(line_i) > 0:
            point_1_line = np.multiply(pix_dim, line_i[0].reshape(1, 3))[0]
            point_2_line = np.multiply(pix_dim, line_i[-1].reshape(1, 3))[0]

            d_soft, d_muscle, d_adipose, d_SAT = add_increments(point_1_line, point_2_line, blur_img, pix_dim, hu_thresholds)
            d_muscle_prop = d_muscle / d_soft
            d_adipose_prop = d_adipose / d_soft
            d_SAT_prop = d_SAT / d_soft

            d_soft_array.append(d_soft)
            d_muscle_array.append(d_muscle_prop)
            d_adipose_array.append(d_adipose_prop)
            d_SAT_array.append(d_SAT_prop)
        else:
            d_soft = np.nan
            d_muscle_prop = np.nan
            d_adipose_prop = np.nan
            d_SAT_prop = np.nan

        print("*******************************")
        print(f"Index {i}")
        print(f"TSTT: {d_soft}")
        print(f"Muscle thickness: {d_muscle}, {d_muscle_prop}")
        print(f"Adipose thickness: {d_adipose}, {d_adipose_prop}")
        print(f"SAT thickness: {d_SAT}, {d_SAT_prop}")
        print()

    print(f"Mean muscle proportion: {np.mean(d_muscle_array)}")
    print(f"Mean adipose proportion: {np.mean(d_adipose_array)}")
    print(f"Mean SAT proportion: {np.mean(d_SAT_array)}")
    print()

    dist_methods.append(np.mean(d_muscle_array))
    dist_methods.append(np.mean(d_adipose_array))
    dist_methods.append(np.mean(d_SAT_array))

    return d_m, np.array(point_3, dtype=object), med_filtered_hu, new_cyl_points, col_arr, dist_methods


def add_increments(point_1, point_2, blur_img, pix_dim, hu_thresholds):
    """
    This function finds the increments of different tissue types along the line between two points
    Parameters
    ----------
    point_1: e.g. GT
    point_2: e.g. STAB
    blur_img: CT image
    pix_dim: pixel dimension of CT image
    hu_thresholds: HU thresholds for different tissues

    Returns
    -------
    d_soft: total soft tissue thickness
    d_muscle: muscle thickness
    d_adipose: adipose thickness
    d_SAT: subcutaneous adipose tissue thickness (last increment of adipose tissue below skin)
    """

    transition_points, muscle_start = get_transition_point(point_1, point_2, blur_img, pix_dim, hu_thresholds)
    d_adipose = []
    d_muscle = []

    print(f"Pixel dimensions: {pix_dim}")

    if len(transition_points) != 0:
        # from GT to first transition point
        d = np.linalg.norm(np.array(transition_points[0][0:3]) - point_1)

        if d <= np.min(pix_dim):
            print(f"too small: {d}")
        elif muscle_start == 0:
            d_adipose.append(d)
            print(f"small adipose: d = {d}")
        elif muscle_start == 1:
            d_muscle.append(d)
            print(f"small muscle: d = {d}")

        # between the next transition points
        for idx in range(0, len(transition_points) - 1):
            d = np.linalg.norm(np.array(transition_points[idx][0:3]) - np.array(transition_points[idx + 1][0:3]))

            if d <= np.min(pix_dim):
                print(f"too small: {d}")
            elif transition_points[idx][3] == 1:
                d_adipose.append(d)
                print(f"small adipose: d = {d}")
            elif transition_points[idx][3] == 0:
                d_muscle.append(d)
                print(f"small muscle: d = {d}")

        # distance of last transition point to soft tissue air boundary
        d = np.linalg.norm(np.array(transition_points[-1][0:3]) - point_2)

        if d <= np.min(pix_dim):
            print(f"too small: {d}")
        elif transition_points[-1][3] == 1:
            d_adipose.append(d)
            print(f"small adipose: d = {d}")
        elif transition_points[-1][3] == 0:
            d_muscle.append(d)
            print(f"small muscle: d = {d}")

    else:
        d = np.linalg.norm(np.array(point_2 - point_1))

        if d <= np.min(pix_dim):
            print(f"too small: {d}")
        elif muscle_start == 1:
            d_muscle.append(d)
            print(f"small muscle: d = {d}")
        elif muscle_start == 0:
            d_adipose.append(d)
            print(f"small adipose: d = {d}")

    d_muscle = np.sum(d_muscle)

    if len(d_adipose) > 0:
        # subcutaneous adipose tissue is the last increment of adipose tissue
        d_SAT = d_adipose[-1]
    else:
        d_SAT = 0
    d_adipose = np.sum(d_adipose)
    d_soft = d_muscle + d_adipose

    return d_soft, d_muscle, d_adipose, d_SAT


def get_transition_point(point_1, point_2, blur_img, pix_dim, hu_thresholds):
    """
    Find the muscle-adipose tissue interface point

    :param point_1:
    :param point_2:
    :param blur_img:
    :param pix_dim:
    :return:
    """

    point_1 = np.divide(point_1, pix_dim)
    point_2 = np.divide(point_2, pix_dim)

    length = 10000
    line = np.linspace(point_1, point_2, num=length, dtype=int)

    x = np.linspace(0, 1, length)
    y = []

    for index, point in enumerate(line):
        y.append(blur_img[point[0], point[1], point[2]])

    med_filtered_hu = signal.medfilt(y, kernel_size=7)

    adipose = 0
    muscle = 0
    other = 0
    transition_points = []
    muscle_start = 0

    for index, intensity in enumerate(med_filtered_hu):

        if index == 0:  # defines tissue at starting point (closest to GT)
            if hu_thresholds['adipose_low'] <= intensity <= hu_thresholds['adipose_high']:  # bandwidth adipose
                muscle_start = 0
                adipose = 1
            elif hu_thresholds['muscle_low'] <= intensity <= hu_thresholds['muscle_high']:  # bandwidth muscle
                muscle_start = 1
                muscle = 1
            else:
                muscle_start = 2  # other tissue/air
                other = 2

        else:
            # transition point to adipose tissue if it wasnt before
            if hu_thresholds['adipose_low'] <= intensity <= hu_thresholds['adipose_high'] and adipose == 0:
                adipose = 1
                muscle = 0
                other = 0
                p3_pix = line[index]
                point_3 = np.multiply(p3_pix, pix_dim)
                transition_points.append(np.array([point_3[0], point_3[1], point_3[2], adipose]))
            # transition point to muscle tissue if it wasnt muscle before
            elif hu_thresholds['muscle_low'] <= intensity <= hu_thresholds['muscle_high'] and muscle == 0:
                adipose = 0
                muscle = 1
                other = 0
                p3_pix = line[index]
                point_3 = np.multiply(p3_pix, pix_dim)
                transition_points.append(np.array([point_3[0], point_3[1], point_3[2], adipose]))
            # transition point to other tissues if it wasnt other before
            elif (intensity > hu_thresholds['musclebone_low'] or intensity < hu_thresholds['adipose_low']) and other == 0:
                other = 2
                adipose = 0
                muscle = 0
                p3_pix = line[index]
                point_3 = np.multiply(p3_pix, pix_dim)
                transition_points.append(np.array([point_3[0], point_3[1], point_3[2], other]))
    return transition_points, muscle_start


def plot_hu_along_stt(hu_units):
    """
    Plot the Hounsfield units (HU) along the path of the soft tissue thickness

    :param hu_units:
    """

    neg = 0
    thresh = 0
    x = np.linspace(0, 1, 200)

    for index, intensity in enumerate(hu_units):
        if neg == 0 and intensity < -29:
            neg = 1
            thresh = x[index]

    plt.figure()
    plt.plot(x, hu_units, 'bo')
    plt.axis([0, 1, -200, 200])
    plt.axvline(x=thresh, c='green')
    plt.axhline(y=-29, c='red')
    plt.legend(('Median filtered HU along path', 'Muscle-to-fat transition point', 'Muscle vs. fat threshold value'))
    plt.xlabel('Normalized distance along TSTT path [-]', fontsize=25)
    plt.ylabel('Hounsfield units', fontsize=25)
    plt.show()


def pv_add_mesh(plotter, verts, faces, color=(30,144,255), opacity=0.5):
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
    mesh = pv.PolyData(verts, faces_pv)
    plotter.add_mesh(mesh, color=color, opacity=opacity)


def pv_add_sphere(plotter, center, radius=3, color='gold'):
    sphere = pv.Sphere(radius=radius, center=center)
    plotter.add_mesh(sphere, color=color)


def pv_add_line(plotter, p0, p1, color=(255, 0, 0), width=3):

    p0 = np.asarray(p0, dtype=float).reshape(3)
    p1 = np.asarray(p1, dtype=float).reshape(3)

    line = pv.Line(p0, p1)
    plotter.add_mesh(line, color=color, line_width=width)


def pv_add_plane_around_gt(plotter, gt, pix_dim, size_y_mm, size_z_mm, color=(255,215,0), opacity=0.2):
    """
    Plot a plane centered at gt with size in voxels based on pix_dim.

    Parameters
    ----------
    gt : array-like
        Ground truth coordinates [x, y, z]
    pix_dim : array-like
        Voxel dimensions [dx, dy, dz] for each axis
    """

    # Compute plane size in world units
    size_y_mm
    size_z_mm

    # Voxel spacing (mm)
    dy = pix_dim[1]
    dz = pix_dim[2]

    # Resolution = number of voxels
    res_y = int(np.round(size_y_mm / dy))
    res_z = int(np.round(size_z_mm / dz))

    # Plane center coordinates
    gt_x = gt[0]  # along x axis
    gt_y = gt[1]  # y coordinate
    gt_z = gt[2]  # z coordinate

    center = (gt_x, gt_y, gt_z)

    # Create the plane
    plane = pv.Plane(
        center=center,
        direction=(1, 0, 0),  # plane normal along x-axis, so i = z and j = y (before i=x, then 90° shifted around y)
        i_size=size_z_mm,  # along y
        j_size=size_y_mm,  # along z
        i_resolution=res_z,
        j_resolution=res_y
    )

    # Plot
    plotter.add_mesh(plane, color=color, show_edges=True, opacity=opacity)


def get_angles(point_array):
    """
    Measure the angles

    :param point_array:
    :return theta:
    :return phi:
    """
    p1 = point_array[0]
    p2 = point_array[1]

    theta = np.math.atan((p2[1] - p1[1]) / (p2[0] - p1[0]))
    theta = np.rad2deg(theta)
    phi = np.math.atan((p2[2] - p1[2]) / (p2[0] - p1[0]))
    phi = np.rad2deg(phi)

    return theta, phi


def find_ct_scans_affirm_ct(ct_dir, seg_dir):
    """
    Matches segmentation files (patientID.nii.gz)
    with CT files (patientID_0000.nii.gz).

    Returns:
        List of [patientID, ct_path, seg_path]
    """

    # 1) Index all CT files once
    ct_index = {}
    for root, _, files in os.walk(ct_dir):
        for fname in files:
            if fname.endswith("_0000.nii.gz"):
                patient_id = fname.replace("_0000.nii.gz", "")
                ct_index[patient_id] = os.path.join(root, fname)

    # 2) Match segmentations to CTs
    pairs = []
    for root, _, files in os.walk(seg_dir):
        for fname in files:
            if fname.endswith(".nii.gz") and not fname.endswith("_0000.nii.gz"):
                patient_id = fname.replace(".nii.gz", "")
                seg_path = os.path.join(root, fname)

                ct_path = ct_index.get(patient_id)
                if ct_path is None:
                    print(f"WARNING: No CT found for {patient_id}")
                    continue
                patient_id_numberonly = int(patient_id.replace("P", ""))
                print(patient_id_numberonly)
                pairs.append([patient_id_numberonly, ct_path, seg_path])

    return pairs


def find_ct_scans_forensic(ct_dir, seg_dir):
    """
    Matches segmentation files (patientID.nii.gz)
    with CT files (patientID_0000.nii.gz).

    Returns:
        List of [patientID, ct_path, seg_path]
    """

    # 1) Index all CT files once
    ct_index = {}
    for root, _, files in os.walk(ct_dir):
        for fname in files:
            if fname.endswith("_hip_0000.nii.gz"):
                patient_id = fname.replace("_hip_0000.nii.gz", "")
                ct_index[patient_id] = os.path.join(root, fname)

    # 2) Match segmentations to CTs
    pairs = []
    for root, _, files in os.walk(seg_dir):
        for fname in files:
            if fname.endswith("_hip.nii.gz") and not fname.endswith("_hip_0000.nii.gz"):
                patient_id = fname.replace("_hip.nii.gz", "")
                seg_path = os.path.join(root, fname)

                ct_path = ct_index.get(patient_id)
                if ct_path is None:
                    print(f"WARNING: No CT found for {patient_id}")
                    continue

                pairs.append([patient_id, ct_path, seg_path])

    return pairs


def rename_seg_files(dir_name):
    """
    Renames segmented files by removing everything apart from ID and adding an _seg.nii.gz extension
    :param dir_name: directory with segmented files
    """
    for sub in os.listdir(dir_name):
        sub_dir = os.path.join(dir_name, sub)
        print()
        print(sub_dir)

        if os.path.isdir(sub_dir):  # rerun in case it is a directory
            rename_seg_files(sub_dir)
        else:
            if (sub_dir.endswith(".nii") or sub_dir.endswith(".nii.gz")) and os.path.isfile(sub_dir):
                scan_id = str(sub.split(".nii")[0])
                print()
                print(f"Scan ID: {scan_id}")
                file_prefix = str(sub_dir.split(scan_id)[0])
                print()
                print('prefix', file_prefix)
                new_file = file_prefix + scan_id + "_seg.nii.gz"
                print()
                print('new file name ', new_file)
                os.rename(os.fsencode(sub_dir), os.fsencode(new_file))


def rename_ct_files(dir_name):
    """
    Renames ct files by removing anything but ID and _seg.nii.gz
    :param dir_name: directory with segmented files
    """
    for sub in os.listdir(dir_name):
        sub_dir = os.path.join(dir_name, sub)
        print()
        print(sub_dir)

        if os.path.isdir(sub_dir):
            rename_seg_files(sub_dir)
        else:
            if (sub_dir.endswith("_0000.nii") or sub_dir.endswith("_0000.nii.gz")) and os.path.isfile(sub_dir):
                scan_id = str(sub.split("_0000")[0])
                print()
                print(f"Scan ID: {scan_id}")
                file_prefix = str(sub_dir.split(scan_id)[0])
                print()
                print('prefix', file_prefix)
                new_file = file_prefix + scan_id + ".nii.gz"
                print()
                print('new file name ', new_file)
                os.rename(os.fsencode(sub_dir), os.fsencode(new_file))

