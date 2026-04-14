"""
Created on 20220807
@created by: Yvan Gugler, Paula Cameron, Christina Wapp
Last updated: 20260305
{


}
"""
import skimage.morphology

# imports
from Pipeline_Modules import img_geom_utils, femur_pre, stt
from Pipeline_Modules.lib import *



def soft_tissue_extraction(patient_id, file_ct, file_segmentation, calibration_status, calibration_file, vis, verb):
    """
    This function calls various sub_functions in the stt.py module to measure the soft tissue thickness.
    Manual adjustment is required for: (1) folder name where screenshots are saved, (2) slope and intercept labelling
    in function defining new HU hresholds in case calibrated image as used

    Parameters
    :param patient_id: Patient ID
    :param file_ct: path to ct file
    :param file_segmentation: path to segmentation file
    :param calibration_status: state "calibrated" or "uncalibrated"
    :param calibration_file: path to calibration file in case it is calibrated
    :param vis: if vis = 1, images are created
    :param verb: if verb = 1, text is printed
    :return out_rowdict: dictionary with all output information; detailed description is commented in dictionary itself
    """

    file_type = "." + re.split(r"\.", file_segmentation)[-1]

    # --- Read metadata and image ---
    metadata, image_info = img_geom_utils.read_image_information(file_segmentation)

    print("!!!!!!!!!")
    print(metadata)
    print(image_info)
    print("!!!!!!!!!")

    if file_type == ".mhd" or file_type == ".mha":
        img_ct_sitk = img_geom_utils.read_mhd(file_ct)  # read mhd file
        img_seg_sitk = img_geom_utils.read_mhd(file_segmentation)
    elif file_type == ".nii" or file_type == ".gz":
        img_ct_sitk = img_geom_utils.read_nifti(file_ct)  # read mhd file
        img_seg_sitk = img_geom_utils.read_nifti(file_segmentation)

    # image properties
    in_spacing = image_info["Spacing"]
    in_origin = image_info["Origin"]
    in_size = image_info["Size"]

    out_spacing = [min(in_spacing)] * 3
    out_origin = in_origin
    out_size = (int(in_spacing[0] / out_spacing[0] * in_size[0]), int(in_spacing[1] / out_spacing[1] * in_size[1]),
                int(in_spacing[2] / out_spacing[2] * in_size[2]))

    # --- Image preprocessing ---
    # Resample
    ResamplingFilter = sitk.ResampleImageFilter()
    ResamplingFilter.SetOutputSpacing(out_spacing)
    ResamplingFilter.SetOutputOrigin(out_origin)
    ResamplingFilter.SetSize(out_size)
    ResamplingFilter.SetInterpolator(sitk.sitkLinear)  # filter for ct image
    img_ct_sitk_res = ResamplingFilter.Execute(img_ct_sitk)

    ResamplingFilter.SetInterpolator(sitk.sitkNearestNeighbor)  # filter for segmented image
    img_seg_sitk_res = ResamplingFilter.Execute(img_seg_sitk)

    # transform sitk images to numpy arrays
    img_ct_np = img_geom_utils.sitk2np(img_ct_sitk)
    img_ct_np_res = img_geom_utils.sitk2np(img_ct_sitk_res)
    img_seg_np = img_geom_utils.sitk2np(img_seg_sitk, pixel_type=np.int16)
    img_seg_np_res = img_geom_utils.sitk2np(img_seg_sitk_res, pixel_type=np.int16)

    # remove everything from the segmentation np array except for the two largest blobs
    seg_np_labeled = measure.label(img_seg_np)
    labels, counts = np.unique(seg_np_labeled, return_counts=True)
    sorted_indices = np.argsort(counts)
    img_seg_np = np.zeros(np.shape(img_seg_np))
    img_seg_np[seg_np_labeled == sorted_indices[-2]] = 1
    img_seg_np[seg_np_labeled == sorted_indices[-3]] = 1
    del seg_np_labeled

    # remove everything from the resampled segmentation np array except for the two largest blobs
    seg_res_np_labeled = measure.label(img_seg_np_res)
    labels, counts = np.unique(seg_res_np_labeled, return_counts=True)
    sorted_indices = np.argsort(counts)
    img_seg_np_res = np.zeros(np.shape(img_seg_np_res))
    img_seg_np_res[seg_res_np_labeled == sorted_indices[-2]] = 1
    img_seg_np_res[seg_res_np_labeled == sorted_indices[-3]] = 1
    del seg_res_np_labeled

    # split resampled numpy array of masks for coordinate system detection
    dim = np.shape(img_seg_np_res)
    split_x = int(dim[0] / 2)

    img_seg_np_1 = img_seg_np_res[0:split_x, :, :]  # femur right
    img_seg_np_2 = img_seg_np_res[split_x:, :, :]  # femur left

    # --- Detect landmarks ---
    # right side (side = 0)
    intersec_1_orig, v_fea_1, head_center_1_orig, intersection_neck_shaft_new_final_1_orig, status_1 = \
        stt.detect_stt_landmarks(img_seg_np_1, split_x, out_spacing, 0)
    r_gt = intersec_1_orig * out_spacing

    print(f"Right GT: {r_gt}")
    print(f"Status right femur: {status_1}")
    print()

    # left side (side = 1)
    intersec_2_orig, v_fea_2, head_center_2_orig, intersection_neck_shaft_new_final_2_orig, status_2 = \
        stt.detect_stt_landmarks(img_seg_np_2, split_x, out_spacing, 1)
    l_gt = intersec_2_orig * out_spacing

    print(f"Left GT: {l_gt}")
    print(f"Status left femur: {status_2}")
    print()

    # Calculate the vector between the femoral head centers
    v_bw_heads_1 = img_geom_utils.normi(head_center_1_orig - head_center_2_orig)
    v_bw_heads_2 = img_geom_utils.normi(head_center_2_orig - head_center_1_orig)

    # -- Adjust HU thresholds for calibrated images
    if calibration_status == 'calibrated':
        file_cal_equ = pd.read_csv(calibration_file)
        row = file_cal_equ.loc[file_cal_equ['record_id'] == patient_id]

        # geneva
        # calibration_equation = {
        #     'slope': row['Asyn_calib_slope'].iloc[0],
        #     'intercept': row['Asyn_calib_intercept'].iloc[0],
        # }

        # bern
        calibration_equation = {
            'slope': row['Slope'].iloc[0],
            'intercept': row['Intercept'].iloc[0],
        }

        hu_thresholds = stt.hu_threshold_calibration(calibration_status, calibration_equation)
    else:
        calibration_equation = {}
        hu_thresholds = stt.hu_threshold_calibration(calibration_status, calibration_equation)

    # --- Call function to measure STT (both sides) ---
    # define size of plane that checks if scanning region is larger than field of view
    size_mm_y = 60
    size_mm_z = 40

    # output right/left = array[gt, min_point, fea_point]
    d_r, d_l, right, left, verts, faces, status, larger_flag, n_stab_points, n_stab_1, stab_1_lp_r, stab_1_lp_l = (
        stt.get_stt(r_gt, l_gt, img_ct_np, in_spacing, v_fea_1, v_fea_2, size_mm_y, size_mm_z,
                    [status_1, status_2], hu_thresholds))

    # --- Use the CT scan to measure the muscle thickness and adipose thickness
    blur = img_ct_np  # removing the Gaussian filter on the image

    # RIGHT
    if status[0] == 1 and larger_flag[0] == 0:
        d_m_r_2, p3_r_2, hu_r_2, cyl_points_r, col_arr_r, dist_methods_r = stt.get_muscle_point(right[0], right[2],
                                                                                                blur, in_spacing,
                                                                                                hu_thresholds)
        d2_r = np.linalg.norm(right[0] - right[2])

    else:
        d2_r = np.nan
        d_m_r_2 = np.nan
        dist_methods_r = np.full(5, np.nan)

    # LEFT
    if status[1] == 1 and larger_flag[1] == 0:
        d_m_l_2, p3_l_2, hu_l_2, cyl_points_l, col_arr_l, dist_methods_l = stt.get_muscle_point(left[0], left[2],
                                                                                                blur, in_spacing,
                                                                                                hu_thresholds)
        d2_l = np.linalg.norm(left[0] - left[2])

    else:
        d2_l = np.nan
        d_m_l_2 = np.nan
        dist_methods_l = np.full(5, np.nan)


    # --- When the function is called, if verb = 1, then more text will be output
    if verb:
        print()
        print("Trochanteric soft tissue thickness:")
        print(f"Right: {d2_r:.2f} mm")
        print(f"Left: {d2_l:.2f} mm")
        print()

        print(f"Right side: Is the subject larger than the scanning region? (0: no, 1:yes) {larger_flag[0]}")
        print(f"Left side: Is the subject larger than the scanning region? (0: no, 1:yes) {larger_flag[1]}")

    # --- When the function is called, if vis = 1, then graphics will be generated
    # --- Plot of Soft tissue air boundary, femur, and STT line
    if vis:
        # choose between next two lines (interactive plot window, no window)
        plotter = pv.Plotter()  # interactive
        # plotter = pv.Plotter(off_screen=True)  # not interactive
        pv.Plotter.add_title(plotter, title=str(patient_id))

        # Segmentation mesh femur
        verts_seg, faces_seg, _, _ = measure.marching_cubes(
            img_seg_np, spacing=in_spacing
        )
        stt.pv_add_mesh(plotter, verts_seg, faces_seg, color='seagreen', opacity=0.5)

        # Original mesh
        stt.pv_add_mesh(plotter, verts, faces, color='dodgerblue', opacity=0.5)

        # Right
        # plot points of greater trochanter and STAB
        if status[0] == 1:
            stt.pv_add_sphere(plotter, right[0], radius=3, color='gold')
            stt.pv_add_sphere(plotter, right[2], radius=3, color='gold')

        # add line of soft tissue thickness
        if larger_flag[0] == 0:
            stt.pv_add_line(plotter, right[0], right[2], color='gold')

        # add a plane that checks STAB presence plus a line in lateral direction from GT
        # if status[0] == 1:
        #     stt.pv_add_line(plotter, right[0], right[0] - [40, 0, 0], color=(255, 165, 0))
        #     stt.pv_add_plane_around_gt(plotter, right[0], pix_dim=in_spacing, size_y_mm=size_mm_y, size_z_mm=size_mm_z,
        #                                color=(255, 215, 0), opacity=0.5)

        # visualize which points do not cross a vertice in case scanning object is too large with small line
        # if larger_flag[0] == 1:
        #     for i in range(len(stab_1_lp_r)):
        #         stt.pv_add_line(plotter, stab_1_lp_r[i], stab_1_lp_r[i] - [10, 0, 0], color=(255, 165, 0))

        # Left
        # landmarks of GT and ST
        if status[1] == 1:
            stt.pv_add_sphere(plotter, left[0], radius=3, color='gold')
            stt.pv_add_sphere(plotter, left[2], radius=3, color='gold')

        # line STT
        if larger_flag[1] == 0:
            stt.pv_add_line(plotter, left[0], left[2], color='gold')

        # add a plane that checks STAB presence plus a line in lateral direction from GT
        # if status[1] == 1:
        #     stt.pv_add_line(plotter, left[0], left[0] + [40, 0, 0], color=(255, 165, 0))
        #     stt.pv_add_plane_around_gt(plotter, left[0], pix_dim=in_spacing, size_y_mm=size_mm_y, size_z_mm=size_mm_z,
        #                                color=(255, 215, 0), opacity=1)

        # points without STAB
        # if larger_flag[1] == 1:
        #     for i in range(len(stab_1_lp_l)):
        #         stt.pv_add_line(plotter, stab_1_lp_l[i], stab_1_lp_l[i] + [10, 0, 0], color=(255, 165, 0))

        # Set up image and screenshots of different perspectives; define folder name for screenshots to be saved in
        plotter.show_axes()
        plotter.show(auto_close=False)
        out_dir = Path("screenshots_test_forensic")
        out_dir.mkdir(exist_ok=True)

        plotter.view_xy()
        plotter.render()
        title_xy = str(patient_id) + '_xy.png'
        plotter.screenshot(out_dir / title_xy)

        plotter.view_yx()
        plotter.render()
        title_yx = str(patient_id) + '_yx.png'
        plotter.screenshot(out_dir / title_yx)

        plotter.view_yz()
        plotter.render()
        title_yz = str(patient_id) + '_yz.png'
        plotter.screenshot(out_dir / title_yz)

        plotter.camera_position = 'yz'
        plotter.camera.azimuth = 180
        plotter.render()
        title_yz_180 = str(patient_id) + '_yz_180.png'
        plotter.screenshot(out_dir / title_yz_180)

        plotter.camera_position = 'xz'
        plotter.camera.azimuth = +45
        plotter.render()
        title_xz_p45 = str(patient_id) + '_xz_p45.png'
        plotter.screenshot(out_dir / title_xz_p45)

        plotter.camera_position = 'xz'
        plotter.camera.azimuth = +65
        plotter.render()
        title_xz_p65 = str(patient_id) + '_xz_p65.png'
        plotter.screenshot(out_dir / title_xz_p65)

        title_xz = str(patient_id) + '_xz.png'
        plotter.view_xz()
        plotter.render()
        plotter.screenshot(out_dir / title_xz)

        title_xz_m45 = str(patient_id) + '_xz_m45.png'
        plotter.camera_position = 'xz'
        plotter.camera.azimuth = -45
        plotter.render()
        plotter.screenshot(out_dir / title_xz_m45)

        title_xz_m65 = str(patient_id) + '_xz_m65.png'
        plotter.camera_position = 'xz'
        plotter.camera.azimuth = -65
        plotter.render()
        plotter.screenshot(out_dir / title_xz_m65)

        title_iso = str(patient_id) + '_iso.png'
        plotter.view_isometric()
        plotter.render()
        plotter.screenshot(out_dir / title_iso)

    # --- Plot of femur with STT line and six parallel lines, as well as increments
    # plot both masked femurs with head_center, neck_shaft_intersection and the point
    if vis:
        # choose option by (un)commenting one of the next two lines
        plotter = pv.Plotter()  # opening plot in a window
        # plotter = pv.Plotter(off_screen=True) # no window opened
        pv.Plotter.add_title(plotter, title=str(patient_id))

        # create mesh of femoral head from segmented image
        verts_m, faces_m, _, _ = measure.marching_cubes(img_seg_np_res)
        stt.pv_add_mesh(plotter, verts_m, faces_m, color='#009e73', opacity=0.5)

        # Right
        # plot landmarks on femur
        if status[0] == 1:
            stt.pv_add_sphere(plotter, head_center_1_orig, radius=2)
            stt.pv_add_sphere(plotter, intersection_neck_shaft_new_final_1_orig, radius=2)
            stt.pv_add_sphere(plotter, intersec_1_orig, radius=2)

        # plot STT increments
        if larger_flag[0] == 0:
            for i, p in enumerate(cyl_points_r):
                p_out = p / out_spacing
                stt.pv_add_sphere(plotter, p_out, radius=1, color=col_arr_r[i])

                fea1 = right[2] / out_spacing
                stt.pv_add_sphere(plotter, fea1, radius=2)

        # Left
        if status[1] == 1:
            stt.pv_add_sphere(plotter, head_center_2_orig, radius=2)
            stt.pv_add_sphere(plotter, intersection_neck_shaft_new_final_2_orig, radius=2)
            stt.pv_add_sphere(plotter, intersec_2_orig, radius=2)

        if larger_flag[1] == 0:
            for i, p in enumerate(cyl_points_l):
                p_out = p / out_spacing
                stt.pv_add_sphere(plotter, p_out, radius=1, color=col_arr_l[i])

                fea2 = left[2] / out_spacing
                stt.pv_add_sphere(plotter, fea2, radius=2)

        # Screenshot of different perspectives of plots; define folder name for screenshots
        plotter.show_axes()
        plotter.show(auto_close=False)
        out_dir = Path("screenshots_test_forensic")
        out_dir.mkdir(exist_ok=True)

        title_femur_xz = str(patient_id) + '_femur_xz.png'
        plotter.view_xz()
        plotter.render()
        plotter.screenshot(out_dir / title_femur_xz)

        plotter.camera_position = 'xz'
        plotter.camera.azimuth = 180
        plotter.render()
        title_xz_femur_180 = str(patient_id) + '_femur_xz_180.png'
        plotter.screenshot(out_dir / title_xz_femur_180)

        plotter.close()

    # output dictionary from analysis
    out_rowdict = {'patient_id': patient_id,
                   'calibration_status': calibration_status,
                   'FEA_dist_R': d2_r,  # STT right along FEA vector
                   'FEA_dist_L': d2_l,  # STT left along FEA vector
                   'FEA_muscle_dist_R': d_m_r_2,  # muscle distance right old line method
                   'FEA_muscle_dist_L': d_m_l_2,  # muscle distance left old line method
                   'NLM_muscle_R': dist_methods_r[0],  # muscle distance right new line method
                   'NLM_adipose_R': dist_methods_r[1],  # adipose distance right new line method
                   'NLM_muscle_L': dist_methods_l[0],  # muscle distance left new line method
                   'NLM_adipose_L': dist_methods_l[1],  # adipose distance left new line method
                   '6LM_muscle_R': dist_methods_r[2],  # muscle distance right six lines method
                   '6LM_adipose_R': dist_methods_r[3],  # adipose distance right six lines method
                   '6LM_SAT_R': dist_methods_r[4],  # SAT distance right six lines method
                   '6LM_muscle_L': dist_methods_l[2],  # muscle distance left six lines method
                   '6LM_adipose_L': dist_methods_l[3],  # adipose distance left six lines method
                   '6LM_SAT_L': dist_methods_l[4],  # SAT distance left six lines method
                   'status_R': status[0],  # right GT detected
                   'status_L': status[1],  # left GT detected
                   'larger_flag_r': larger_flag[0],  # body larger than scanning field of view
                   'larger_flag_l': larger_flag[1],
                   'n_stab_points': n_stab_points,  # number of points in soft tissue air boundary (stab) plane
                   'n_stab_1_r': n_stab_1[0],  # number of points without STAB within plane on the right
                   'n_stab_1_l': n_stab_1[1]}  # left
    return out_rowdict


def run_stt(ct_dir, seg_dir, cohort, filename_output, calibration_status, filename_input=None, calibration_file=None):
    """
    This function calls the main upstream function "soft_tissue_extraction" and applies it to all scans.

    Parameters
    ----------
    ct_dir: directory where all ct scans can be found
    seg_dir: directory where all segmentations can be found
    cohort: choose between "affirmct" and "forensic"
    filename_output: name for the csv file where results will be saved
    calibration_status: state whether the ct scans are calibrated ("calibrated" or "uncalibrated")
    filename_input: if some scans were already processed, give the directory of the csv with the results and ID, so these can be skipped
    calibration_file: in case ct scans are calibrated, give a file with the calibration equations

    Returns
    -------
    output: csv file with the results of the pipeline
    """

    start_time = time.time()

    # list of ID's/names that have already been calculated
    if filename_input is not None:
        data = pd.read_csv(filename_input)
        patient_id_calculated = set(data['patient_id'])
    else:
        patient_id_calculated = []

    non_computable_INS = [2099, 1256, 122, 2411, 1930, 1144, 2561, 2029]
    non_computable_HUG = [2289]

    if cohort == 'affirm_ct':
        filenames = stt.find_ct_scans_affirm_ct(ct_dir, seg_dir)

    if cohort == 'forensic':
        filenames = stt.find_ct_scans_forensic(ct_dir, seg_dir)

    # optional: if only certain IDs are of interest, create your own set
    IDs_set = [row[0] for row in filenames[4:6]]
    IDs_set = set([1651, 2568])
    print(IDs_set)
    filenames_selected = []
    for item in filenames:
        patient_id = item[0]  # position 0 is ID
        if patient_id in IDs_set:
            filenames_selected.append(item)

    filenames = filenames_selected

    print('Number of Scans: ', len(filenames))
    results = []

    for idx, row in enumerate(filenames):
        patient_id = row[0]
        file_ct = row[1]
        file_seg = row[2]

        print()
        print(f"{idx} -- Name: {patient_id}")

        # Add in this if statement when certain scans should be excluded
        if patient_id not in patient_id_calculated and patient_id not in non_computable_INS and patient_id not in non_computable_HUG:
            print()
            c = datetime.now()
            current_time = c.strftime('%H:%M:%S')
            print('Current time is: ', current_time)
            print(f"Time elapsed: {(time.time() - start_time):.2f} seconds")
            print()
            print(patient_id)
            print("____________________")
            out_rowdict = soft_tissue_extraction(patient_id, file_ct, file_seg, calibration_status,
                                                 calibration_file, vis=1, verb=0)
            results.append(out_rowdict)

        output = pd.DataFrame(results)

        print(output)
        output.to_csv(filename_output)
