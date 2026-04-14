"""
Created on 20210226
@author: Yvan Gugler
{

}
"""
# import modules
from Pipeline_Modules import femur_pre, img_geom_utils
from Pipeline_Modules.lib import *


def mhd2fea(config, im_file, sampleID, laterality=None):
    """
    :param config:
    :param im_file:
    :param sampleID:
    :param laterality:
    :return:
    """
    # __________________________________________________________________________________________________________________
    # Configuration of the simulation
    # __________________________________________________________________________________________________________________
    voxel_size = config["el_size"]  # FE element size in mm
    exp_config = config["exp_config_image"]  # experimental configuration
    scan_head_down = config["invert_anteropost_axis"]  # scan protocol with patient looking downwards (coord. sys. def.)
    fabric_on = config["ftype"]  # anisotropy included in model
    if fabric_on == "iso":
        fabric_on = False
    else:
        fabric_on = True

    # definition of load case
    dia_angle_fall = config["dia_angle_fall"]  # inclination of diaphysis in side fall configuration
    rot_angle = config["rot_angle"]  # rotation angle around diaphysis axis
    dia_angle_stance = config["dia_angle_stance"]  # inclination of diaphysis for stance configuration

    # homogenization
    sphere_volume = config["sphere_vol"]
    # background layer to add to image
    add_layer = config["layer"]  # thickness of voxel layer added to image from bounding box (in mm)

    # __________________________________________________________________________________________________________________
    # Inputs
    # __________________________________________________________________________________________________________________
    file_type = config["imfile_type"]
    # __________________________________________________________________________________________________________________
    # Import image and initialize analysis
    # __________________________________________________________________________________________________________________
    metadata, image_info = img_geom_utils.read_image_information(im_file)  # read metaData
    if file_type == ".mhd":
        img_mask_sitk = img_geom_utils.read_mhd(im_file)  # read mhd file
    elif file_type == ".vtk":
        img_mask_sitk = img_geom_utils.read_vtk(im_file)  # read vtk file

    spacing = img_mask_sitk.GetSpacing()  # spacing
    add_layer = int(add_layer/spacing[0])  # convert added layer to nbr of voxels

    # initialize dictionary for further analysis
    femur = femur_pre.initialize_bone_dict(img_mask_sitk, image_info, metadata, laterality,
                                               scanHeadDown=scan_head_down, add_layer=add_layer)

    femur["Mask_Array"], removed_voxels = \
        img_geom_utils.remove_badly_connected_voxels(femur["Mask_Array"].astype('int'), con=6)
    femur["Mask_Array"] = img_geom_utils.close_holes(femur["Mask_Array"].astype('int'))
    femur["Fabric"] = fabric_on
    femur["Layer"] = add_layer
    del img_mask_sitk, image_info, metadata, removed_voxels

    # __________________________________________________________________________________________________________________
    # Outputs
    # __________________________________________________________________________________________________________________
    fea_path = config["feadir"] + "/" + sampleID
    sum_path = config["sumdir"] + "/" + sampleID
    mesh_file_name = fea_path + "/" + sampleID + "_" + femur["Laterality"] + "_" + exp_config + "_" + 'mesh.inp'
    input_file_name = fea_path + "/" + sampleID + "_" + femur["Laterality"] + "_" + exp_config + '.inp'

    mask_mhd_file = sum_path + "/" + sampleID + "_" + femur["Laterality"] + "_" + exp_config + "_" + 'mask.mhd'
    bmd_mhd_file = sum_path + "/" + sampleID + "_" + femur["Laterality"] + "_" + exp_config + "_" + 'bmd_mask.mhd'

    filenames = {"Image": im_file,
                 "Input": input_file_name,
                 "Mesh": mesh_file_name,
                 "FEA_Mask": mask_mhd_file,
                 "FEA_BMD": bmd_mhd_file}
    # __________________________________________________________________________________________________________________
    # Detect coordinate system of the proximal femur and compute transformation to experimental positions
    # __________________________________________________________________________________________________________________
    head_center, head_radius, v_dia, v_neck, intersection_neck_shaft, \
        neck_shaft_angle, diaphysis_base_point = femur_pre.detect_coordinate_system(femur["Mask_Array"],
                                                                                        add_layer, femur['Spacing'])
    # compute transformations
    rot_center = (femur["Spacing"][0]) * intersection_neck_shaft
    rot_center = rot_center.astype('int')

    if exp_config == "fall":
        trans_matrix, inv_trans_matrix = \
            femur_pre.compute_transform_to_side_fall_position(v_neck, v_dia, dia_angle_fall, rot_angle, rot_center)
        femur["Trans Matrix Fall"] = trans_matrix
        femur["Trans Matrix Stance"] = None

    elif exp_config == "stance":
        trans_matrix, inv_trans_matrix = \
            femur_pre.compute_transform_to_stance_position(v_neck, v_dia, dia_angle_stance, rot_center)
        femur["Trans Matrix Fall"] = None
        femur["Trans Matrix Stance"] = trans_matrix

    # update femur dictionary with geometry information
    femur["Head Center"] = head_center
    femur["Head Radius"] = head_radius
    femur["Diaphysis Axis"] = v_dia
    femur["Neck Axis"] = v_neck
    femur["Neck-Shaft-Intersection"] = intersection_neck_shaft
    femur["CCD Angle"] = neck_shaft_angle
    femur["Diaphysis-Base"] = diaphysis_base_point
    del head_center, head_radius, v_dia, v_neck, intersection_neck_shaft, neck_shaft_angle, diaphysis_base_point

    # __________________________________________________________________________________________________________________
    # Creation of arrays for FE Mesh generation --> Material mapping, embedding and cropping
    # __________________________________________________________________________________________________________________
    sphere_radius_homogenization = img_geom_utils.sphere_radius_from_volume(sphere_volume)  # sphere for homogenization
    femur["Struct_Array_FE"] = femur_pre.material_mapping(femur["Mask_Array"], femur["BMD_Array"],
                                                            trans_matrix, inv_trans_matrix,
                                                            femur["Spacing"],
                                                            radius=sphere_radius_homogenization,
                                                            vox_size=voxel_size, fabric_on=False)

    femur["Struct_Array_FE"]["Mask"], removed_voxels = img_geom_utils.remove_badly_connected_voxels(femur["Struct_Array_FE"]["Mask"])
    femur["Struct_Array_FE"]["BVTV"] = femur["Struct_Array_FE"]["BVTV"]*(femur["Struct_Array_FE"]["Mask"] == 1)
    del removed_voxels

    # compute initial and final mass and volume
    initial_mean_bmd, initial_bmc, initial_bone_vol, initial_tot_vol = \
        femur_pre.compute_bmc_and_volume(femur["BMD_Array"], spacing[0], femur_pre.bmd2bvtv(femur["BMD_Array"]))
    final_mean_bmd, final_bmc, final_bone_vol, final_tot_vol = femur_pre.compute_bmc_and_volume(
        femur["Struct_Array_FE"]["BMD"], voxel_size, bvtv_array=femur["Struct_Array_FE"]["BVTV"])

    # Embedding and cropping
    if exp_config == "fall":
        femur["Struct_Array_FE"]['Mask'] = femur_pre.embed_side_fall(femur["Struct_Array_FE"]['Mask'], voxel_size,
                                                                     head_radius)
    elif exp_config == "stance":
        femur["Struct_Array_FE"]['Mask'] = femur_pre.embed_stance(femur["Struct_Array_FE"]['Mask'], voxel_size,
                                                                  head_radius)

    cropping_limits = femur_pre.crop_femur(femur["Struct_Array_FE"]['Mask'])
    femur["Struct_Array_FE"] = femur["Struct_Array_FE"][cropping_limits[0]:cropping_limits[1],
                                                        cropping_limits[2]:cropping_limits[3],
                                                        cropping_limits[4]:cropping_limits[5]]

    # Compute head center in experimental configuration for creation of driving node
    head_center_exp = 1/spacing[0] * img_geom_utils.transform_point(spacing[0]*femur["Head Center"], trans_matrix)
    head_center_exp = spacing[0] * (head_center_exp - voxel_size/spacing[0]*np.array([cropping_limits[0],
                                                                                       cropping_limits[2],
                                                                                       cropping_limits[4]]))
    v_dia_exp = img_geom_utils.transform_point(femur["Diaphysis Axis"], trans_matrix)
    v_neck_exp = img_geom_utils.transform_point(femur["Neck Axis"], trans_matrix)
    femur["v_dia_exp"] = v_dia_exp
    femur["v_neck_exp"] = v_neck_exp
    femur["head_center_exp"] = head_center_exp
    del trans_matrix, inv_trans_matrix, v_dia_exp, v_neck_exp
    # __________________________________________________________________________________________________________________
    # Writing FE Mesh and main input file
    # __________________________________________________________________________________________________________________
    nbr_elems, nbr_bone_elems = femur_pre.create_voxelmesh(mesh_file_name, femur["Struct_Array_FE"]['Mask'], femur["Struct_Array_FE"]['BVTV'], voxel_size, config["umat_parameters"])
    if exp_config == "fall":
        femur_pre.create_maininput(input_file_name, mesh_file_name, 'fall', config['load_displacement_fall'], head_center_exp)
    if exp_config == "stance":
        femur_pre.create_maininput(input_file_name, mesh_file_name, 'stance', config['load_displacement_stance'],
                                  head_center_exp)

    # __________________________________________________________________________________________________________________
    # Summary variables for summary (log) file as quality control and writing of mhd files with mask and bmd information
    # experimental configuration
    # __________________________________________________________________________________________________________________
    femur["Initial Mean BMD"] = np.around(initial_mean_bmd, decimals=2)
    femur["Initial BMC"] = np.around(initial_bmc, decimals=2)
    femur["Initial Volume"] = np.around(initial_tot_vol, decimals=2)
    femur["Initial Bone Volume"] = np.around(initial_bone_vol, decimals=2)
    femur["Final BMC"] = np.around(final_bmc, decimals=2)
    femur["Final Mean BMD"] = np.around(final_mean_bmd, decimals=2)
    femur["Final Volume"] = np.around(final_tot_vol, decimals=2)
    femur["Final Bone Volume"] = np.around(final_bone_vol, decimals=2)
    del initial_mean_bmd, initial_bmc, initial_tot_vol, initial_bone_vol, final_mean_bmd, final_bmc, final_tot_vol, \
        final_bone_vol

    femur["Tot Elements"] = nbr_elems
    femur["Bone Elements"] = nbr_bone_elems
    del nbr_elems, nbr_bone_elems

    img_geom_utils.write_MHD(femur["Struct_Array_FE"]["Mask"], mask_mhd_file)
    img_geom_utils.write_MHD(femur["Struct_Array_FE"]["BMD"], bmd_mhd_file)
    # ******************************************************************************************************************
    return femur, filenames


def mhd2fea_new(config, im_file, sampleID, laterality=None):
    """
    :param config:
    :param im_file:
    :param sampleID:
    :param laterality:
    :return:
    """
    # __________________________________________________________________________________________________________________
    # Configuration of the simulation
    # __________________________________________________________________________________________________________________
    voxel_size = config["el_size"]  # FE element size in mm
    exp_config = config["exp_config_image"]  # experimental configuration
    scan_head_down = config["invert_anteropost_axis"]  # scan protocol with patient looking downwards (coord. sys. def.)
    fabric_on = config["ftype"]  # anisotropy included in model
    if fabric_on == "iso":
        fabric_on = False
    else:
        fabric_on = True

    bmc_correction = config["bmc_correction"]  # options: "BMC", "BMD", "Off"
    distal_extension = config["distal_extension"]
    distal_cut_length = config["distal_cut_length"]
    perp_cut = config["perpendicular_cut"]

    # definition of load case
    dia_angle_fall = config["dia_angle_fall"]  # inclination of diaphysis in side fall configuration
    rot_angle = config["rot_angle"]  # rotation angle around diaphysis axis
    dia_angle_stance = config["dia_angle_stance"]  # inclination of diaphysis for stance configuration

    # homogenization
    sphere_volume = config["sphere_vol"]
    # background layer to add to image
    add_layer = config["layer"]  # thickness of voxel layer added to image from bounding box (in mm)

    # __________________________________________________________________________________________________________________
    # Inputs
    # __________________________________________________________________________________________________________________
    file_type = config["imfile_type"]
    # __________________________________________________________________________________________________________________
    # Import image and initialize analysis
    # __________________________________________________________________________________________________________________
    metadata, image_info = img_geom_utils.read_image_information(im_file)  # read metaData
    if file_type == ".mhd":
        img_mask_sitk = img_geom_utils.read_mhd(im_file)  # read mhd file
    elif file_type == ".vtk":
        img_mask_sitk = img_geom_utils.read_vtk(im_file)  # read vtk file

    spacing = img_mask_sitk.GetSpacing()  # spacing
    add_layer = int(add_layer/spacing[0])  # convert added layer to nbr of voxels


    # initialize dictionary for further analysis
    femur = femur_pre.initialize_bone_dict(img_mask_sitk, image_info, metadata, laterality,
                                               scanHeadDown=scan_head_down, add_layer=add_layer,
                                               csv_filename=config["csv_file_case"])

    # original bone mass and volume
    original_mean_bmd, original_bmc, original_bone_vol, original_tot_vol = \
        femur_pre.compute_bmc_and_volume(femur["BMD_Array"], spacing[0])

    femur["Sample_ID"] = sampleID
    femur["Mask_Array"], removed_voxels = img_geom_utils.remove_badly_connected_voxels(femur["Mask_Array"].astype('int'), con=6)
    femur["Mask_Array"] = img_geom_utils.close_holes(femur["Mask_Array"].astype('int'))
    femur["Fabric"] = fabric_on
    femur["Layer"] = add_layer
    laterality = femur["Laterality"]

    del img_mask_sitk, image_info, metadata, removed_voxels

    # __________________________________________________________________________________________________________________
    # Outputs
    # __________________________________________________________________________________________________________________
    fea_path = config["feadir"] + "/" + sampleID
    sum_path = config["sumdir"] + "/" + sampleID
    png_path = sum_path + "/PNG/"

    if not os.path.exists(png_path):
        os.mkdir(png_path)

    mesh_file_name = fea_path + "/" + sampleID + "_" + femur["Laterality"] + "_" + exp_config + "_" + 'mesh.inp'
    input_file_name = fea_path + "/" + sampleID + "_" + femur["Laterality"] + "_" + exp_config + '.inp'

    mask_mhd_file = sum_path + "/" + sampleID + "_" + femur["Laterality"] + "_" + exp_config + "_" + 'mask.mhd'
    bmd_mhd_file = sum_path + "/" + sampleID + "_" + femur["Laterality"] + "_" + exp_config + "_" + 'bmd_mask.mhd'

    regions_mhd_file = sum_path + "/" + sampleID + "_" + femur["Laterality"] + "_" + 'regions_masked.mhd'

    histogram_image_file = sum_path + "/" + sampleID + "_" + femur["Laterality"] + "_" + 'histograms.png'

    mayavi_imfile1 = png_path + sampleID + "_" + laterality + "_1.png"
    mayavi_imfile2 = png_path + sampleID + "_" + laterality + "_2.png"
    mayavi_imfile3 = png_path + sampleID + "_" + laterality + "_3.png"
    mayavi_imfile4 = png_path + sampleID + "_" + laterality + "_4.png"
    mayavi_imfile5 = png_path + sampleID + "_" + laterality + "_5.png"

    filenames = {"Image": im_file,
                 "Input": input_file_name,
                 "Mesh": mesh_file_name,
                 "FEA_Mask": mask_mhd_file,
                 "FEA_BMD": bmd_mhd_file,
                 "Region_Masks": regions_mhd_file,
                 "Histogram_Image": histogram_image_file,
                 "Coord_File_1": mayavi_imfile1,
                 "Coord_File_2": mayavi_imfile2,
                 "Coord_File_3": mayavi_imfile3,
                 "Coord_File_4": mayavi_imfile4,
                 "Coord_File_5": mayavi_imfile5}
    # __________________________________________________________________________________________________________________
    # Detect coordinate system of the proximal femur and compute transformation to experimental positions
    # __________________________________________________________________________________________________________________
    head_center, head_radius, v_dia, v_neck, intersection_neck_shaft, \
        neck_shaft_angle, d_point, \
        head_z, head_ml, head_ap, head_pa, v_dia_in, v_neck_in, new_d_point \
            = femur_pre.detect_coordinate_system(femur["Mask_Array"], add_layer, femur['Spacing'])

    # __________________________________________________________________________________________________________________
    # Detect LT and GT and correct coordinate axes
    # __________________________________________________________________________________________________________________
    v_dia_new_final, v_neck_new_final, intersection_neck_shaft_new_final, db_point, lt_mask_orig_orientation, \
    gt_mask_medial_orig_orientation, gt_mask_lateral_orig_orientation, lt_peak, lt_peak_alt, cog_lt_peak, cog_lt_mid, \
    new_neck_shaft_angle, neck_lt_angle, dia_flag, dl, new_base_point_orig_config, lt_peak_axis, V_prox_femur = \
        femur_pre.detect_landmarks_and_correct_coordinate_system(femur["Mask_Array"], head_center, head_radius,
                                                                 v_dia, v_neck, intersection_neck_shaft, d_point,
                                                                 v_neck_in, add_layer=add_layer,
                                                                 spacing=femur["Spacing"])

    # __________________________________________________________________________________________________________________
    # Rendering of coordinate system
    # __________________________________________________________________________________________________________________

    img_geom_utils.set_mayavi_offscreen(config["offscreen"])
    fig01 = img_geom_utils.mayavi_plot_mesh(femur["Mask_Array"], color=(0, 0, 1), opacity=0.4)
    img_geom_utils.mayavi_plot_mesh(lt_mask_orig_orientation, color=(1, 1, 1), fig=fig01, opacity=0.4)
    img_geom_utils.mayavi_plot_mesh(gt_mask_medial_orig_orientation, color=(0, 1, 1), fig=fig01, opacity=0.4)
    img_geom_utils.mayavi_plot_mesh(gt_mask_lateral_orig_orientation, color=(0, 1, 1), fig=fig01, opacity=0.4)

    img_geom_utils.mayavi_add_arrow(v_dia_new_final, db_point, length=200, fig=fig01,
                                    scaling_factor=1, color=(1, 0, 0))
    img_geom_utils.mayavi_add_arrow(v_neck_new_final, intersection_neck_shaft_new_final, length=200,
                                    fig=fig01, scaling_factor=1, color=(1, 0, 0))

    img_geom_utils.mayavi_add_sphere(intersection_neck_shaft_new_final, fig=fig01, radius=2)
    img_geom_utils.mayavi_add_sphere(head_center, fig=fig01, radius=2)
    img_geom_utils.mayavi_add_sphere(db_point, fig=fig01, radius=2)

    img_geom_utils.mayavi_add_sphere(lt_peak, fig=fig01, radius=2)
    img_geom_utils.mayavi_add_sphere(lt_peak_axis, fig=fig01, radius=2)

    cam_angle1, cam_angle2 = femur_pre.compute_cam_angles(v_neck_new_final)

    img_geom_utils.set_mayavi_orientation_axes()
    img_geom_utils.save_mayavi_to_png(mayavi_imfile1, fig01)

    img_geom_utils.set_mayavi_view(cam_angle1 + neck_lt_angle / np.pi * 180, 75)
    img_geom_utils.save_mayavi_to_png(mayavi_imfile2, fig01)

    img_geom_utils.set_mayavi_view(cam_angle1, 90)
    img_geom_utils.save_mayavi_to_png(mayavi_imfile3, fig01)

    img_geom_utils.set_mayavi_view(cam_angle2, 90)
    img_geom_utils.save_mayavi_to_png(mayavi_imfile4, fig01)

    # __________________________________________________________________________________________________________________
    # Distal extension
    # __________________________________________________________________________________________________________________
    if distal_extension is not None:
        """
        mask_extended, inv_overlap, mask_to_add, extension_flag = \
            femur_pre.distally_extend_femur(femur["BMD_Array"], lt_peak,
                                            new_base_point_orig_config, intersection_neck_shaft_new_final,
                                            v_dia_new_final, spacing=femur["Spacing"],
                                            target_distal_length=distal_extension)
        """
        mask_extended, inv_overlap, mask_to_add, extension_flag = \
            femur_pre.distally_extend_femur(femur["BMD_Array"], lt_peak,
                                            new_d_point, intersection_neck_shaft_new_final,
                                            v_dia_in, spacing=femur["Spacing"],
                                            target_distal_length=distal_extension)

        if extension_flag is True:
            img_geom_utils.mayavi_plot_mesh(mask_to_add, color=(1, 0, 1), fig=fig01, opacity=0.4)
            img_geom_utils.mayavi_plot_mesh(inv_overlap, color=(1, 0, 0), fig=fig01, opacity=0.4)

        img_geom_utils.set_mayavi_view(cam_angle2, 90)
        img_geom_utils.save_mayavi_to_png(mayavi_imfile5, fig01)

        femur["BMD_Array"] = mask_extended
        mask_array = np.copy(femur["BMD_Array"])
        mask_array[mask_extended != 0] = 1
        mask_array.astype("uint8")
        femur["Mask_Array"] = mask_array

        # volume and bone mass of extended model
        extended_mean_bmd, extended_bmc, extended_bone_vol, extended_tot_vol = \
            femur_pre.compute_bmc_and_volume(femur["BMD_Array"], spacing[0])

        femur["Distal_Extension_Length"] = distal_extension - dl
        femur["Extension_flag"] = extension_flag

        femur["Region_Masks"] = np.copy(femur["Mask_Array"])  # add LT, GT med. and GT lat
        femur["Region_Masks"][lt_mask_orig_orientation != 0] = 2
        femur["Region_Masks"][gt_mask_lateral_orig_orientation != 0] = 3
        femur["Region_Masks"][gt_mask_medial_orig_orientation != 0] = 4
        femur["Region_Masks"][mask_to_add != 0] = 5
        femur["Region_Masks"][inv_overlap == 1] = 6

        del mask_array, mask_extended, extension_flag

    else:
        femur["Region_Masks"] = np.copy(femur["Mask_Array"])  # add LT, GT med. and GT lat
        femur["Region_Masks"][lt_mask_orig_orientation != 0] = 2
        femur["Region_Masks"][gt_mask_lateral_orig_orientation != 0] = 3
        femur["Region_Masks"][gt_mask_medial_orig_orientation != 0] = 4

    img_geom_utils.mayavi_close_figure(fig01)

    # __________________________________________________________________________________________________________________
    # Compute transformations
    # __________________________________________________________________________________________________________________
    rot_center = (femur["Spacing"][0])*intersection_neck_shaft_new_final
    rot_center2 = intersection_neck_shaft_new_final

    if exp_config == "fall":
        trans_matrix, inv_trans_matrix = \
            femur_pre.compute_transform_to_side_fall_position(v_neck_new_final, v_dia_new_final,
                                                              dia_angle_fall, rot_angle, rot_center)
        trans_matrix2, inv_trans_matrix2 = \
            femur_pre.compute_transform_to_side_fall_position(v_neck_new_final, v_dia_new_final,
                                                              dia_angle_fall, rot_angle, rot_center2)
        femur["Trans Matrix Fall"] = trans_matrix
        femur["Trans Matrix Stance"] = None

    elif exp_config == "stance":
        trans_matrix, inv_trans_matrix = \
            femur_pre.compute_transform_to_stance_position(v_neck_new_final, v_dia_new_final, dia_angle_stance,
                                                           rot_center)
        trans_matrix2, inv_trans_matrix2 = \
            femur_pre.compute_transform_to_side_fall_position(v_neck_new_final, v_dia_new_final,
                                                              dia_angle_fall, rot_angle, rot_center2)
        femur["Trans Matrix Fall"] = None
        femur["Trans Matrix Stance"] = trans_matrix

    # update femur dictionary with geometry information
    femur["Head Center"] = head_center
    femur["Head Radius"] = head_radius
    femur["Lesser Trochanter"] = lt_peak
    femur["LT projected"] = lt_peak_axis

    femur["Diaphysis Axis"] = v_dia_new_final
    femur["Neck Axis"] = v_neck_new_final
    femur["Initial Diaphysis Axis"] = v_dia
    femur["Initial Neck Axis"] = v_neck

    femur["Neck-Shaft-Intersection"] = intersection_neck_shaft_new_final
    femur["Diaphysis-Base"] = db_point

    femur["CCD Angle"] = new_neck_shaft_angle
    femur["Neck_LT_angle"] = img_geom_utils.rad2deg(neck_lt_angle)
    femur["Distal_length"] = dl
    femur["Distal_Base_Point"] = new_base_point_orig_config
    femur["Diaphysis_Computation_Flag"] = dia_flag

    femur["Proximal Femur Volume"] = V_prox_femur
    # __________________________________________________________________________________________________________________
    # Creation of arrays for FE Mesh generation --> Material mapping, embedding and cropping
    # __________________________________________________________________________________________________________________
    sphere_radius_homogenization = img_geom_utils.sphere_radius_from_volume(sphere_volume)  # sphere for homogenization
    femur["Struct_Array_FE"] = femur_pre.material_mapping(femur["Mask_Array"], femur["BMD_Array"],
                                                            trans_matrix, inv_trans_matrix,
                                                            femur["Spacing"],
                                                            radius=sphere_radius_homogenization,
                                                            vox_size=voxel_size, fabric_on=False)

    femur["Struct_Array_FE"]["Mask"], removed_voxels = \
        img_geom_utils.remove_badly_connected_voxels(femur["Struct_Array_FE"]["Mask"])
    femur["Struct_Array_FE"]["BVTV"] = femur["Struct_Array_FE"]["BVTV"]*(femur["Struct_Array_FE"]["Mask"] == 1)
    femur["Struct_Array_FE"]["BMD"] = femur["Struct_Array_FE"]["BMD"] * (femur["Struct_Array_FE"]["Mask"] == 1)
    del removed_voxels

    # __________________________________________________________________________________________________________________
    # Embedding and cropping including BMC correction
    # __________________________________________________________________________________________________________________
    if exp_config == "fall":
        femur["Struct_Array_FE"]['Mask'], embedding_depth_voxels_fall = \
            femur_pre.embed_side_fall(femur["Struct_Array_FE"]['Mask'], voxel_size)
    elif exp_config == "stance":
        femur["Struct_Array_FE"]['Mask'], embedding_depth_voxels_stance = \
            femur_pre.embed_stance(femur["Struct_Array_FE"]['Mask'], voxel_size)

    # define cut location
    v_dia_new_final_trans = np.dot(trans_matrix[0:3, 0:3], v_dia_new_final)  # dia axis in exp. config.
    v_neck_new_final_trans = np.dot(trans_matrix[0:3, 0:3], v_neck_new_final)
    lt_peak_axis_trans = img_geom_utils.transform_point(lt_peak_axis, trans_matrix2)

    if type(distal_cut_length) == int:
        cut_length = distal_cut_length  # in mm from lesser trochanter
        cut_loc = (lt_peak_axis_trans - cut_length / femur["Spacing"][0] * v_dia_new_final_trans) * \
                  femur["Spacing"][0] / voxel_size
    elif type(distal_cut_length) == float:
        cut_length = distal_cut_length * head_radius * spacing[0]
        cut_loc = (lt_peak_axis_trans - cut_length / femur["Spacing"][0] * v_dia_new_final_trans) * \
                  femur["Spacing"][0] / voxel_size
    else:
        cut_loc = None

    crop_limits, remaining_mask, cut_off_mask, boundary_el = femur_pre.crop_femur_new(femur["Struct_Array_FE"]["Mask"],
                            voxel_size, perpendicular_cut=perp_cut, cut_location=cut_loc, v_dia=v_dia_new_final_trans)

    femur["Struct_Array_FE"]["Mask"] = femur["Struct_Array_FE"]["Mask"] * remaining_mask
    femur["Struct_Array_FE"]["BMD"] = femur["Struct_Array_FE"]["BMD"] * remaining_mask
    femur["Struct_Array_FE"]["BVTV"] = femur["Struct_Array_FE"]["BVTV"] * remaining_mask

    correction_factor, remaining_inp_mask = \
        femur_pre.bmc_correction(femur["BMD_Array"], femur["Struct_Array_FE"]["BMD"], femur["Spacing"],
                                     (voxel_size, voxel_size, voxel_size), cut_off_mask, inv_trans_matrix)

    if bmc_correction is True:
        femur["Struct_Array_FE"]["BMD"] = femur["Struct_Array_FE"]["BMD"] * correction_factor
        femur["Struct_Array_FE"]["BVTV"] = femur_pre.bmd2bvtv(femur["Struct_Array_FE"]["BMD"])

    # compute bone properties (bmc, mean bmd, bvtv, tot Vol.) of FE mesh and of original image cut at the same location
    initial_mean_bmd, initial_bmc, initial_bone_vol, initial_tot_vol = \
        femur_pre.compute_bmc_and_volume(remaining_inp_mask, spacing[0])
    final_mean_bmd, final_bmc, final_bone_vol, final_tot_vol = femur_pre.compute_bmc_and_volume(
        femur["Struct_Array_FE"]["BMD"], voxel_size, bvtv_array=femur["Struct_Array_FE"]["BVTV"])

    femur["Struct_Array_FE"] = femur["Struct_Array_FE"][crop_limits[0]:crop_limits[1], crop_limits[2]:crop_limits[3],
                                                        crop_limits[4]:crop_limits[5]]

    if boundary_el is not None:
        boundary_el = boundary_el[crop_limits[0]:crop_limits[1], crop_limits[2]:crop_limits[3],
                                  crop_limits[4]:crop_limits[5]]

    # Compute head center in experimental configuration for creation of driving node
    head_center_exp = 1/spacing[0] * img_geom_utils.transform_point(spacing[0]*femur["Head Center"], trans_matrix)
    head_center_exp = spacing[0] * (head_center_exp - voxel_size/spacing[0]*np.array([crop_limits[0],
                                                                                      crop_limits[2],
                                                                                      crop_limits[4]]))

    femur["v_dia_exp"] = v_dia_new_final_trans
    femur["v_neck_exp"] = v_neck_new_final_trans
    femur["head_center_exp"] = head_center_exp

    try:
        femur["embedding_depth_fall"] = embedding_depth_voxels_fall
        hc_to_gt = femur["head_center_exp"][0] - (27 - femur["embedding_depth_fall"] * voxel_size)
        femur["Head_Center_to_GT"] = hc_to_gt
    except:
        pass

    try:
        femur["embedding_depth_stance"] = embedding_depth_voxels_stance
    except:
        pass

    # __________________________________________________________________________________________________________________
    # Writing FE Mesh and main input file
    # __________________________________________________________________________________________________________________
    nbr_elems, nbr_bone_elems, nbr_nodes = femur_pre.create_voxelmesh(
        mesh_file_name, femur["Struct_Array_FE"]['Mask'],
        femur["Struct_Array_FE"]['BVTV'], voxel_size, config["umat_parameters"], boundary_els=boundary_el)

    if exp_config == "fall":
        if config["load_displacement_fall"]["type"] == "abs":
            disp = config["load_displacement_fall"]["value"]
        elif config["load_displacement_fall"]["type"] == "rel":
            disp = config["load_displacement_fall"]["value"] * hc_to_gt
        else:
            disp = 5
        femur_pre.create_maininput(input_file_name, mesh_file_name, config,
                                   disp, head_center_exp)
    if exp_config == "stance":
        if config["load_displacement_stance"]["type"] == "abs":
            disp = config["load_displacement_stance"]["value"]
        elif config["load_displacement_stance"]["type"] == "rel":
            disp = config["load_displacement_stance"]["value"] * head_radius * spacing[0]
        else:
            disp = 3
        femur_pre.create_maininput(input_file_name, mesh_file_name, config, disp,
                                   head_center_exp)

    bvtv_histogram_FE, hist_bin_edges_FE = \
        np.histogram(femur["Struct_Array_FE"]["BVTV"][femur["Struct_Array_FE"]["Mask"] == 1], bins=10,
                     range=(0.0, 1.0))
    bvtv_histogram_image, hist_bin_edges_image = \
        np.histogram(femur_pre.bmd2bvtv(remaining_inp_mask)[remaining_inp_mask != 0],
                                        bins=10, range=(0.0, 1.0))
    bin_edges = np.histogram_bin_edges(femur["Struct_Array_FE"]["BVTV"], bins=10, range=(0.0, 1.0))
    nbr_voxels = np.count_nonzero(remaining_inp_mask)

    fig, axes = plt.subplots(1, 2, sharex=False, sharey='row', figsize=(8, 6))
    axes[0].bar(bin_edges[:-1], 100 / nbr_voxels * bvtv_histogram_image, width=0.1, align='edge')
    axes[0].set_title("Histogram of BVTV values - Image")
    axes[0].set_xlabel("BVTV")
    axes[0].set_ylabel("Relative part of volume [%]")

    axes[1].bar(bin_edges[:-1], 100 / nbr_bone_elems * bvtv_histogram_FE, width=0.1, align='edge')
    axes[1].set_title("Histogram of BVTV values - FE mesh", loc="center")
    axes[1].set_xlabel("BVTV")

    plt.savefig(histogram_image_file)
    plt.close()

    # __________________________________________________________________________________________________________________
    # Summary variables for summary file as quality control and writing of mhd files with mask and bmd information
    # experimental configuration
    # __________________________________________________________________________________________________________________
    femur["Original Mean BMD"] = np.around(original_mean_bmd, decimals=2)
    femur["Original BMC"] = np.around(original_bmc, decimals=2)
    femur["Original Volume"] = np.around(original_tot_vol, decimals=2)
    femur["Original Bone Volume"] = np.around(original_bone_vol, decimals=2)

    femur["Extended_Mean_BMD"] = np.around(extended_mean_bmd, decimals=2)
    femur["Extended_BMC"] = np.around(extended_bmc, decimals=2)
    femur["Extended_Bone_Vol"] = np.around(extended_bone_vol, decimals=2)
    femur["Extended_Tot_Vol"] = np.around(extended_tot_vol, decimals=2)

    femur["Initial Mean BMD"] = np.around(initial_mean_bmd, decimals=2)
    femur["Initial BMC"] = np.around(initial_bmc, decimals=2)
    femur["Initial Volume"] = np.around(initial_tot_vol, decimals=2)
    femur["Initial Bone Volume"] = np.around(initial_bone_vol, decimals=2)

    femur["Final Mean BMD"] = np.around(final_mean_bmd, decimals=2)
    femur["Final BMC"] = np.around(final_bmc, decimals=2)
    femur["Final Volume"] = np.around(final_tot_vol, decimals=2)
    femur["Final Bone Volume"] = np.around(final_bone_vol, decimals=2)
    femur["BMC_correction_factor"] = correction_factor

    femur["Histogram Image"] = bvtv_histogram_image
    femur["Histogram FE"] = bvtv_histogram_FE
    femur["Histogram Bins"] = bin_edges

    del original_mean_bmd, original_bmc, original_tot_vol, original_bone_vol, initial_mean_bmd, initial_bmc,\
        initial_tot_vol, initial_bone_vol, final_mean_bmd, final_bmc, final_tot_vol, final_bone_vol, correction_factor,\
        bvtv_histogram_image, bvtv_histogram_FE, bin_edges

    femur["Tot Elements"] = nbr_elems
    femur["Bone Elements"] = nbr_bone_elems
    femur["Nodes"] = nbr_nodes
    del nbr_elems, nbr_bone_elems, nbr_nodes

    img_geom_utils.write_MHD(femur["Struct_Array_FE"]["Mask"], mask_mhd_file)
    img_geom_utils.write_MHD(femur["Struct_Array_FE"]["BMD"], bmd_mhd_file)
    img_geom_utils.write_MHD(femur["Region_Masks"], regions_mhd_file)
    # ******************************************************************************************************************
    return femur, filenames
