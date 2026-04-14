"""
Created on 20210226
@author: Yvan Gugler

{
Main file for femurPYpeline. Takes the config.yaml file as argument.
Has to be run as follows: python3 pipeline.py --configfile "/pathToConfigFile/config"
}
"""
# imports
from Pipeline_Modules.lib import *
from Pipeline_Modules.mhd2fea import mhd2fea_new
from Pipeline_Modules import io_utils, femur_post, img_geom_utils

running_file_path = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
# model parameters
parser.add_argument(
    "--configfile",
    type=str,
    default="config",
    help="A configuration file containing parameter values",
)


def main(args):
    """
    :return:
    """
    start_time_batch = time.time()
    config = io_utils.read_config_file(args.configfile)
    print(config["load_displacement_fall"])
    print(config["load_displacement_stance"])
    print(config["save_field_output"])

    # offscreen rendering
    mlab.options.offscreen = config["offscreen"]

    imdir = config["imdir"]
    filetype = config["imfile_type"]
    if config["pattern"] is not None:
        image_list = io_utils.create_file_list(imdir, config["pattern"])
    else:
        image_list = io_utils.create_file_list(imdir, filetype)

    print(image_list)
    message_file = config["sumdir"] + "/" + "error_messages.txt"

    for image in image_list:
        # ______________________________________________________________________________________________________________
        # Preprocessing
        # ______________________________________________________________________________________________________________
        try:
            start_sample_preproc = time.time()
            sample_ID, exp_config, laterality = io_utils.parse_filename(image,
                                                                        config["parsing_pattern"], config["positions"])

            if laterality == "L":
                laterality = "Left"
            if laterality == "R":
                laterality = "Right"

            if config["exp_config"] == "from_file":
                config["exp_config_image"] = exp_config
            else:
                config["exp_config_image"] = config["exp_config"]

            fea_path = config["feadir"] + "/" + sample_ID
            sum_path = config["sumdir"] + "/" + sample_ID

            try:
                config["csv_file_case"] = config["csv_file"].replace("sample_ID", sample_ID)
            except:
                config["csv_file_case"] = None

            if not os.path.exists(fea_path):
                os.makedirs(fea_path)
            if not os.path.exists(sum_path):
                os.makedirs(sum_path)

            # femur, files = mhd2fea(config, image, sample_ID, laterality)
            femur, files = mhd2fea_new(config, image, sample_ID, laterality)
            end_sample_preproc = time.time()
            # ______________________________________________________________________________________________________________
            # Simulation
            # ______________________________________________________________________________________________________________
            # create subprocess that calls abaqus
            abaqus = config["abaqus"]
            nprocs = config["nprocs"]
            inputfile = files["Input"]
            umat = config["umat"]

            if umat is None:
                umat = running_file_path + "/UMAT/UMAT_QUADRIC_PRIMAL_rincon_FEAmurPYpeline.f"

            print("UMAT: ", umat)

            job = sample_ID + "_" + femur["Laterality"] + "_" + config["exp_config_image"]

            abqworkdir = config["feadir"] + "/" + sample_ID + "/" + "Simulation"
            if not os.path.exists(abqworkdir):
                os.mkdir(abqworkdir)
            os.chdir(abqworkdir)

            command = "{} interactive cpus={:d} job={} inp={} user={} ask_delete=OFF".format(
                abaqus, nprocs, job, inputfile, umat
            )

            subprocess.call(command, cwd=abqworkdir, shell=True)
            end_sample_sim = time.time()
            # ______________________________________________________________________________________________________________
            # Postprocessing
            # ______________________________________________________________________________________________________________
            files["abq_dat"] = job + ".dat"
            files["fu_out"] = job + ".txt"
            files["fu_plot"] = job + "_fu.png"
            files["abq_macro"] = job + ".py"
            files["abq_odb"] = job + ".odb"
            files["bvtv_cut"] = job + "_bvtv.png"
            files["damage_cut"] = job + "_dam.png"

            voxel_size = config["el_size"]
            dof = 0
            if config["exp_config_image"] == "fall":
                dof = 0
                try:
                    f_ult_criterion = config["f_ult_criterion_fall"].copy()
                except:
                    f_ult_criterion = None

            elif config["exp_config_image"] == "stance":
                dof = 2
                try:
                    f_ult_criterion = config["f_ult_criterion_stance"].copy()
                except:
                    f_ult_criterion = None

            if f_ult_criterion is not None:
                if type(f_ult_criterion[1]) is float:
                    pass
                elif f_ult_criterion[1] == "Center to GT":
                    f_ult_criterion[1] = femur["head_center_exp"][0] - (27 - femur["embedding_depth_fall"] * voxel_size)
                elif f_ult_criterion[1] == "Head Radius":
                    f_ult_criterion[1] = femur["Head Radius"]

            # FU-curve
            d, f = femur_post.extract_force_disp(files["abq_dat"], dof, files["fu_out"])
            fu_chars = list(femur_post.compute_force_disp_characteristics_new(f, d, f_ult_criterion))

            femur_post.plot_force_disp(fu_chars[5], fu_chars[4], showfig=False, savefig=True,
                                       filename=files["fu_plot"], f_ult=fu_chars[0])

            # cut location for abaqus cut views
            cut_origin = (femur["head_center_exp"][0], femur["head_center_exp"][1], femur["head_center_exp"][2])

            femur_post.create_abaqus_macro(files["abq_macro"], files["abq_odb"], cut_origin, files["bvtv_cut"],
                                           files["damage_cut"])
            command_postproc = "abaqus viewer noGUI=%s" % files["abq_macro"]
            subprocess.call(command_postproc, cwd=abqworkdir, shell=True)

            # Generate PDF report of the simulation
            umat_flags = [config["umat_parameters"]["DENSFL"], config["umat_parameters"]["VISCFL"],
                          config["umat_parameters"]["PYFL"]]
            im_props = [femur["Initial Volume"], femur["Initial BMC"], femur["Initial Mean BMD"]]
            mesh_props = [femur["Final Volume"], femur["Final BMC"], femur["Final Mean BMD"]]

            femur_post.pdf_report_new(job, files["Coord_File_1"], files["Coord_File_2"], files["Coord_File_3"],
                                      files["Coord_File_5"], files["Histogram_Image"],
                                      files["bvtv_cut"], files["damage_cut"], files["fu_plot"], im_props, mesh_props,
                                      fu_chars, umat_flags)

            if os.path.exists(files["abq_odb"]):
                if config["save_field_output"] is not None:
                    odb_file = abqworkdir + "/" + files["abq_odb"]
                    field_output_file_1 = odb_file[0:-4] + "_" + config["save_field_output"] + ".npy"
                    field_output_file_2 = field_output_file_1[0:-4] + ".mhd"
                    background = -1.0
                    inc = np.argwhere(fu_chars[-1] == fu_chars[0])[0][0]
                    print(inc)
                    print(field_output_file_1)
                    if abs(fu_chars[0] - fu_chars[-1][inc-1]) > abs(fu_chars[0] - fu_chars[-1][inc+1]):
                        inc += 1
                    else:
                        inc -= 1

                    command = "abaqus python abq_field_output_to_mhd.py [{:s},{:d},{:s},{:f},{:s},{:f}]".format(
                        odb_file, inc, config["save_field_output"], voxel_size, field_output_file_1,
                        background
                        )
                    print(command)
                    subprocess.call(command, cwd=running_file_path, shell=True)

                    img_geom_utils.npy2mhd(field_output_file_1, field_output_file_2, sitk.sitkFloat32)
                    os.remove(field_output_file_1)

                if config["keep_ODB"] is False:
                    os.remove(files["abq_odb"])

            # Summary file containing different information
            end_sample_tot = time.time()
            femur["Preprocessing Time"] = end_sample_preproc - start_sample_preproc
            femur["Simulation Time"] = end_sample_sim - end_sample_preproc
            femur["Postprocessing Time"] = end_sample_tot - end_sample_sim
            femur["Computation Time"] = end_sample_tot - start_sample_preproc
            summary_file_name = sum_path + "/" + sample_ID + "_" + femur["Laterality"] + "_" + \
                                config["exp_config_image"] + "_" + 'log.csv'
            files["Summary File"] = summary_file_name

            femur["Strength_characteristics"] = fu_chars[0:4]
            femur["FU_displacement"] = fu_chars[4]
            femur["FU_force"] = fu_chars[5]
            femur["f_ult_criterion_final"] = f_ult_criterion
            femur["Files"] = files
            del femur["Struct_Array_FE"], femur["BMD_Array"], femur["Mask_Array"], femur["Region_Masks"]

            femur_post.write_sample_summary_file_new(femur, config, files["Summary File"])  # write summary file

        except Exception:
            print("Exception in user code:")
            print("-" * 60)
            traceback.print_exc(file=sys.stdout)
            print("-" * 60)
            error_message = traceback.format_exc()
            io_utils.write_message_file(message_file, image, error_message)

        os.chdir(config["imdir"])  # go back to image directory
    end_time_batch = time.time()
    print(end_time_batch - start_time_batch)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
