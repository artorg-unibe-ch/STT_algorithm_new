"""
Rerun simulation for a specific case with another element size.
Author: Yvan Gugler
Date: 2022-02-14
"""

from Pipeline_Modules.lib import *
from Pipeline_Modules import img_geom_utils, femur_pre, femur_post, io_utils

running_file_path = os.path.dirname(os.path.realpath(__file__))


def main(args):
    parser = argparse.ArgumentParser()
    # model parameters
    parser.add_argument(
        "--csv_file",
        type=str,
        help="csv-log-file")

    parser.add_argument(
        "--new_voxel_size",
        type=float,
        help="float for isotropic FE-element size")

    parser.add_argument(
        "--cpus",
        type=int,
        help="Number of CPUS for Abaqus simulation"
    )

    args = parser.parse_args(args)

    # read info from csv-log-file to dictionary
    d = io_utils.read_csv_to_dict_2(args.csv_file, ";")

    old_input_file_name = d["Files"]["Input"]

    filename_bmd = d["Files"]["FEA_BMD"]
    filename_mask = d["Files"]["FEA_Mask"]

    old_voxel_size = d["el_size"]

    new_voxel_size = args.new_voxel_size
    new_voxel_size_str = str(new_voxel_size)
    new_voxel_size_str = new_voxel_size_str.replace(".", "p")

    feadir = d["feadir"] + "_" + new_voxel_size_str

    sampleID, exp_config, laterality = io_utils.parse_filename(old_input_file_name, '/|\_|\.',
                                                                {"Sample_ID": -4, "Laterality": -3, "exp_config": -2})

    fea_path = feadir + "/" + sampleID
    abqworkdir = fea_path + "/" + "Simulation"

    if not os.path.exists(fea_path):
        os.makedirs(fea_path)

    if not os.path.exists(abqworkdir):
        os.makedirs(abqworkdir)

    new_mesh_file_name = fea_path + "/" + sampleID + "_" + laterality + "_" + exp_config + "_" + \
                         new_voxel_size_str + "_" + 'mesh.inp'
    new_input_file_name = fea_path + "/" + sampleID + "_" + laterality + "_" + exp_config + "_" + \
                          new_voxel_size_str + '.inp'

    # read files
    img_sitk_bmd = img_geom_utils.read_mhd(filename_bmd)
    img_sitk_mask = img_geom_utils.read_mhd(filename_mask)

    # spacings
    spacing = (1, 1, 1)
    old_spacing = tuple(old_voxel_size * x for x in spacing)
    new_spacing = tuple(new_voxel_size * x for x in spacing)

    img_sitk_bmd.SetSpacing(old_spacing)
    img_sitk_mask.SetSpacing(old_spacing)

    # resample volume and mask
    img_sitk_bmd_new_res = img_geom_utils.resample_volume_sitk(img_sitk_bmd, interpolator=sitk.sitkNearestNeighbor,
                                                             new_spacing=new_spacing)
    img_sitk_mask_new_res = img_geom_utils.resample_volume_sitk(img_sitk_mask, interpolator=sitk.sitkNearestNeighbor,
                                                             new_spacing=new_spacing)

    # transform to numpy arrays
    img_np_bmd = img_geom_utils.sitk2np(img_sitk_bmd_new_res)
    img_np_mask = img_geom_utils.sitk2np(img_sitk_mask_new_res)

    # bmd to bvtv conversion
    img_np_bvtv = femur_pre.bmd2bvtv(img_np_bmd)

    test_array = np.where(img_np_bvtv == 0, 1, 0) * np.where(img_np_mask == 1, 1, 0)
    img_np_bvtv = np.where(test_array == 1, 0.01, img_np_bvtv)

    # write new mesh file
    flags = {"DENSFL": 0, "VISCFL": 0, "PYFL": 0}
    nbr_els, nbr_bone_els, nbr_nodes = femur_pre.create_voxelmesh(new_mesh_file_name, img_np_mask, img_np_bvtv,
                                                                  new_voxel_size, flags, boundary_els=None)

    # create new input file
    with open(old_input_file_name, 'r') as infile:
        with open(new_input_file_name, 'w') as outfile:
            for line in infile:
                if "*INCLUDE,input=" in line:
                    line = line.replace(line, "*INCLUDE,input=" + new_mesh_file_name+"\n")
                outfile.write(line)
            outfile.close()
        infile.close()

    # run simulation
    abaqus = d["abaqus"]
    nprocs = args.cpus

    umat = running_file_path + "/UMAT/UMAT_QUADRIC_PRIMAL_rincon_FEAmurPYpeline.f"

    job = sampleID + "_" + laterality + "_" + exp_config
    os.chdir(abqworkdir)

    command = "{} interactive cpus={:d} job={} inp={} user={} ask_delete=OFF".format(
        abaqus, nprocs, job, new_input_file_name, umat)

    sim_time_0 = time.time()
    subprocess.call(command, cwd=abqworkdir, shell=True)
    sim_time_1 = time.time()

    odb_file = job+".odb"
    if os.path.exists(odb_file):
        os.remove(odb_file)

    # create FU characteristic
    dof = 0
    if exp_config == "fall":
        dof = 0
        f_ult_criterion = d["f_ult_criterion_fall"]
    elif exp_config == "stance":
        dof = 2
        f_ult_criterion = d["f_ult_criterion_stance"]

    try:
        if math.isnan(f_ult_criterion):
            f_ult_criterion = None
    except:
        pass

    dat_file_name = job + ".dat"
    out_file_name = job + ".txt"
    fu_plot_file = job + "_fu.png"
    summary_file = job + "_log.csv"

    disp, force = femur_post.extract_force_disp(dat_file_name, dof, out_file_name)
    fu_chars = list(femur_post.compute_force_disp_characteristics_new(force, disp, f_ult_criterion))
    femur_post.plot_force_disp(fu_chars[5], fu_chars[4], showfig=False, savefig=True,
                                           filename=fu_plot_file, f_ult=fu_chars[0])

    # write log file for this simulation
    simulation_time = sim_time_1 - sim_time_0

    summary_dict = {"Simulation Time": simulation_time,
                    "CPUS": nprocs,
                    "Strength_characteristics": fu_chars,
                    "El. size": new_voxel_size,
                    "Tot Elements": nbr_els,
                    "Bone Elements": nbr_bone_els,
                    "Nodes": nbr_nodes
                    }

    io_utils.write_csv_from_dict_2(summary_file, summary_dict)
    print(simulation_time)
    print(fu_chars)


if __name__ == "__main__":
    # args = parser.parse_args()
    # main(args)
    main(sys.argv[1:])
