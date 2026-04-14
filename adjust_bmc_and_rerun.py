"""
Created on 20220721
@author: Yvan Gugler

Adjust the bone mineral content of a mask to the one of a corresponding mask and perform FE analysis with the adjusted
masked BMD image. Used for adjusting BMC values of 3D-DXA images to QCT. Large differences in BMC values are most likely
due to different image calibration approaches.
Call as follows:
python adjust_bmc_and_rerun.py "/path/to/workdir/" "/path/to_MHD_images_of_saved_Masks_to_adjust" "/path/to_MHD_images_of_saved_masks_that_are_reference"
"""

from Pipeline_Modules.lib import *
from Pipeline_Modules import io_utils, femur_pre, img_geom_utils, femur_post

cli_args = sys.argv[1:]  # remove file name (argv[0])
print(cli_args)

running_file_path = os.path.dirname(os.path.realpath(__file__))
umat = running_file_path + "/UMAT/UMAT_QUADRIC_PRIMAL_rincon_FEAmurPYpeline.f"

workdir = cli_args[0]
# workdir = "/home/ygugler/OneDrive/FEAMUR2020/Test_BMC_Adjustment/"

path_img_to_adjust = cli_args[1]
path_img_ref = cli_args[2]
# path_img_to_adjust = "/home/ygugler/OneDrive/FEAMUR2020/Test_BMC_Adjustment/3d_dxa"
# path_img_ref = "/home/ygugler/OneDrive/FEAMUR2020/Test_BMC_Adjustment/qct"

# extensions to look for saved images
ext_img_to_adjust = "bmd_mask.mhd"
ext_img_ref = "bmd_mask.mhd"

# parsing pattern for filenames
parsing_pattern = '/|\_|\.'
positions = {"Sample_ID": -6, "Laterality": -5, "exp_config": -4}
positions_log = {"Sample_ID": -5, "Laterality": -4, "exp_config": -3}

# field_output to save
save_field_output = "SDV2"

# list bmd mask files of 3d_dxa where bmc needs to be corrected
list_of_images_to_adjust = io_utils.create_file_list(path_img_to_adjust, ext_img_to_adjust)
list_of_logfiles = io_utils.create_file_list(path_img_to_adjust, ".csv")

# list corresponding file of the qct
list_of_reference_images = io_utils.create_file_list(path_img_ref, ext_img_ref)

# column names for pandas dataframes
column_names = ["Sample_ID", "Laterality", "Config", "Case", "File", "File_with_path", "Mask_File_with_path"]
column_names_ref = ["Sample_ID", "Laterality", "Config", "Case", "File", "File_with_path"]
column_names_log = ["Case", "LogFile"]

images_to_adjust_DF = pd.DataFrame(columns=column_names)  # images to adjust
images_reference_DF = pd.DataFrame(columns=column_names_ref)  # reference images (to adjust to)
log_file_DF = pd.DataFrame(columns=column_names_log)  # log files


# go through list of images to adjust and add files to dataframe
for img in list_of_images_to_adjust:
    sample_ID, exp_config, laterality = io_utils.parse_filename(img, parsing_pattern, positions)
    case = sample_ID + "_" + laterality + "_" + exp_config
    filename = re.split("/", img)
    filename = filename[-1]
    path = img.replace(filename, '')
    split_filename = re.split("_", filename)
    mask_img = path + split_filename[0] + "_" + split_filename[1] + "_" + split_filename[2] + "_" + split_filename[4]

    add_to_DF = pd.DataFrame([sample_ID, laterality, exp_config, case, filename, img, mask_img]).T
    add_to_DF.columns = column_names
    images_to_adjust_DF = images_to_adjust_DF.append(add_to_DF)

# go though list of logfiles and add info to dataframe
for file in list_of_logfiles:
    sample_ID, exp_config, laterality = io_utils.parse_filename(file, parsing_pattern, positions_log)
    case = sample_ID + "_" + laterality + "_" + exp_config
    add_to_DF = pd.DataFrame([case, file]).T
    add_to_DF.columns = column_names_log
    log_file_DF = log_file_DF.append(add_to_DF)

# go through list of reference images and to dataframe
for img in list_of_reference_images:
    sample_ID, exp_config, laterality = io_utils.parse_filename(img, parsing_pattern, positions)
    case = sample_ID + "_" + laterality + "_" + exp_config
    filename = re.split("/", img)
    filename = filename[-1]
    add_to_DF = pd.DataFrame([sample_ID, laterality, exp_config, case, filename, img]).T
    add_to_DF.columns = column_names_ref
    images_reference_DF = images_reference_DF.append(add_to_DF)

# put all info together in one only dataframe
images_to_adjust_DF = images_to_adjust_DF.sort_values("Case")
images_reference_DF = images_reference_DF.sort_values("Case")
log_file_DF = log_file_DF.sort_values("Case")

images_to_adjust_DF["Test"] = images_to_adjust_DF["Case"].isin(images_reference_DF["Case"])
images_reference_DF["Test"] = images_reference_DF["Case"].isin(images_to_adjust_DF["Case"])

images_to_adjust_DF = images_to_adjust_DF[images_to_adjust_DF["Test"]==True]
images_reference_DF = images_reference_DF[images_reference_DF["Test"]==True]

images_to_adjust_DF["Reference_File_with_path"] = images_reference_DF["File_with_path"]

log_file_DF["Test"] = log_file_DF["Case"].isin(images_to_adjust_DF["Case"])
log_file_DF = log_file_DF[log_file_DF["Test"] == True]

images_to_adjust_DF["Log_File"] = log_file_DF["LogFile"]
del images_reference_DF, log_file_DF

# create directories for FEM simulation files and summary files
femdir = workdir + "FEM/"
sumdir = workdir + "Summaries/"

if os.path.exists(workdir) is False:
    os.mkdir(workdir)

if os.path.exists(femdir) is False:
    os.mkdir(femdir)

if os.path.exists(sumdir) is False:
    os.mkdir(sumdir)

# go through images in dataframe and perform simulations
for index, row in images_to_adjust_DF.iterrows():
    file_to_adjust_sitk = img_geom_utils.read_mhd(row['File_with_path'])  # read file to adjust
    file_mask_sitk = img_geom_utils.read_mhd(row["Mask_File_with_path"])
    file_ref_sitk = img_geom_utils.read_mhd(row["Reference_File_with_path"])  # read reference file
    log_file = io_utils.read_csv_to_dict_2(row["Log_File"])  # read log_file
    file_to_adjust_np = img_geom_utils.sitk2np(file_to_adjust_sitk)
    file_mask_np = img_geom_utils.sitk2np(file_mask_sitk)
    file_ref_np = img_geom_utils.sitk2np(file_ref_sitk)

    bmc_adjust = np.sum(file_to_adjust_np)
    bmc_ref = np.sum(file_ref_np)

    ratio = bmc_adjust / bmc_ref  # ratio of BMCs

    file_to_adjust_np = 1/ratio * file_to_adjust_np  # adjust BMC

    bvtv_mask_np = femur_pre.bmd2bvtv(file_to_adjust_np)

    # holes that have to be closed
    holes = np.where(file_mask_np == 1, 1, 0) - np.where(file_to_adjust_np != 0, 1, 0)
    bvtv_mask_np = np.where(holes == 1, 0.01, bvtv_mask_np)

    # FEM and Summary directories for the case
    femdir_case = femdir + row["Sample_ID"]
    sumdir_case = sumdir + row["Sample_ID"]

    if os.path.exists(femdir_case) is False:
        os.mkdir(femdir_case)

    if os.path.exists(sumdir_case) is False:
        os.mkdir(sumdir_case)

    # new mesh and main_input file
    mesh_file_name = row["Sample_ID"] + "_" + row["Laterality"] + "_" + exp_config + "_mesh.inp"
    input_file_name = row["Sample_ID"] + "_" + row["Laterality"] + "_" + exp_config + "_main_input.inp"

    mesh_file_name = femdir_case + "/" + mesh_file_name
    input_file_name = femdir_case + "/" + input_file_name

    umat_parameters = {"DENSFL": 0, "VISCFL": 0, "PYFL": 0}
    femur_pre.create_voxelmesh(mesh_file_name, file_mask_np,
                               bvtv_mask_np, 3, umat_parameters)

    sim_config = {"exp_config_image": row["Config"], "save_field_output": "SDV2"}

    try:
        hc_to_gt = log_file["head_center_exp"][0] - (27 - log_file["embedding_depth_fall"] * log_file["el_size"])
    except:
        print("Warning: Stance not Fall configuration")

    if row["Config"] == "fall":
        if log_file["load_displacement_fall"]["type"] == "abs":
            disp = log_file["load_displacement_fall"]["value"]
        elif log_file["load_displacement_fall"]["type"] == "rel":
            disp = log_file["load_displacement_fall"]["value"] * hc_to_gt
        else:
            disp = 5

    if row["Config"] == "stance":
        if log_file["load_displacement_stance"]["type"] == "abs":
            disp = log_file["load_displacement_stance"]["value"]
        elif log_file["load_displacement_stance"]["type"] == "rel":
            disp = log_file["load_displacement_stance"]["value"] * log_file["Head Radius"] * log_file["Spacing"][0]
        else:
            disp = 3

    femur_pre.create_maininput(input_file_name, mesh_file_name, sim_config, disp, log_file["head_center_exp"])

    # perform Abaqus simulation with new mesh
    simdir = femdir_case + "/Simulation/"
    if os.path.exists(simdir) is False:
        os.mkdir(simdir)

    os.chdir(simdir)
    nprocs = 8

    job = row["Sample_ID"] + "_" + row["Laterality"] + "_" + row["Config"]

    command = "{} interactive cpus={:d} job={} inp={} user={} ask_delete=OFF".format(
        "abaqus", nprocs, job, input_file_name, umat
    )

    subprocess.call(command, cwd=simdir, shell=True)

    # extract FU data from dat file
    dat_file = job + ".dat"
    out_file = job + ".txt"

    dof = 0
    if row["Config"] == "fall":
        dof = 0
        try:
            f_ult_criterion = log_file["f_ult_criterion_fall"].copy()
        except:
            f_ult_criterion = None

    elif row["Config"] == "stance":
        dof = 2
        try:
            f_ult_criterion = log_file["f_ult_criterion_stance"].copy()
        except:
            f_ult_criterion = None

    if f_ult_criterion is not None:
        if type(f_ult_criterion[1]) is float:
            pass
        elif f_ult_criterion[1] == "Center to GT":
            f_ult_criterion[1] = log_file["head_center_exp"][0] - (27 - log_file["embedding_depth_fall"] * log_file["el_size"])
        elif f_ult_criterion[1] == "Head Radius":
            f_ult_criterion[1] = log_file["Head Radius"]

    d, f = femur_post.extract_force_disp(dat_file, dof, output_file=out_file)
    fu_chars = list(femur_post.compute_force_disp_characteristics_new(f, d, f_ult_criterion))

    # BVTV histograms for both cases
    os.chdir(sumdir_case)
    fu_plot_file = job + ".png"
    femur_post.plot_force_disp(fu_chars[5], fu_chars[4], showfig=False, savefig=True,
                               filename=fu_plot_file, f_ult=fu_chars[0])

    bvtv_histogram_FE_adjusted, hist_bin_edges_FE = \
        np.histogram(bvtv_mask_np[file_mask_np == 1], bins=10,
                     range=(0.0, 1.0))
    bvtv_histogram_FE_ref, hist_bin_edges_image = \
        np.histogram(femur_pre.bmd2bvtv(file_ref_np)[file_ref_np != 0],
                                        bins=10, range=(0.0, 1.0))

    histogram_image_file = job + "_hist.png"

    bin_edges = np.histogram_bin_edges(bvtv_mask_np, bins=10, range=(0.0, 1.0))
    nbr_voxels = np.count_nonzero(file_ref_np)

    fig, axes = plt.subplots(1, 2, sharex=False, sharey='row', figsize=(8, 6))
    axes[0].bar(bin_edges[:-1], 100 / nbr_voxels * bvtv_histogram_FE_ref, width=0.1, align='edge')
    axes[0].set_title("BVTV values - FE - QCT-based")
    axes[0].set_xlabel("BVTV")
    axes[0].set_ylabel("Relative part of volume [%]")

    axes[1].bar(bin_edges[:-1], 100 / log_file["Bone Elements"] * bvtv_histogram_FE_adjusted, width=0.1, align='edge')
    axes[1].set_title("BVTV values - FE - 3D-DXA-based", loc="center")
    axes[1].set_xlabel("BVTV")

    plt.savefig(histogram_image_file)
    plt.close()

    odb_file = simdir + job + ".odb"

    # write field output (SDV2 - scalar Damage variable) to MHD file
    if os.path.exists(odb_file):
        if save_field_output is not None:
            field_output_file_1 = odb_file[0:-4] + "_" + save_field_output + ".npy"
            field_output_file_2 = field_output_file_1[0:-4] + ".mhd"
            background = -1.0
            inc = np.argwhere(fu_chars[-1] == fu_chars[0])[0][0]

            if abs(fu_chars[0] - fu_chars[-1][inc - 1]) > abs(fu_chars[0] - fu_chars[-1][inc + 1]):
                inc += 1
            else:
                inc -= 1

            command = "abaqus python abq_field_output_to_mhd.py [{:s},{:d},{:s},{:f},{:s},{:f}]".format(
                odb_file, inc, save_field_output, log_file["el_size"], field_output_file_1,
                background
            )

            subprocess.call(command, cwd=running_file_path, shell=True)

            img_geom_utils.npy2mhd(field_output_file_1, field_output_file_2, sitk.sitkFloat32)
            os.remove(field_output_file_1)

        try:
            os.remove(odb_file)
        except:
            pass

    summary_file = sumdir_case + "/" + row["Case"] + "_" + row["Config"] + "_log.csv"

    sum_dict = {}

    sum_dict["Sample_ID"] = row["Sample_ID"]
    sum_dict["Laterality"] = row["Laterality"]
    sum_dict["Config"] = row["Config"]
    sum_dict["Strength_characteristics"] = fu_chars[0:4]
    sum_dict["FU_displacement"] = fu_chars[4]
    sum_dict["FU_force"] = fu_chars[5]
    sum_dict["Elements Reference Mesh"] = nbr_voxels
    sum_dict["Elements Adjusted Mesh"] = log_file["Bone Elements"]
    sum_dict["Adjusted BMC"] = bmc_ref

    io_utils.write_csv_from_dict_2(summary_file, sum_dict)
