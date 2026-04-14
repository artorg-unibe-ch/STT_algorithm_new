"""
File that summarizes ultimate force results from a batch of files in one file.
Author: Yvan Gugler
Date: 2022-02-23
"""

from Pipeline_Modules.lib import *
from Pipeline_Modules import io_utils

running_file_path = os.path.dirname(os.path.realpath(__file__))


def main(args):

    parser = argparse.ArgumentParser()
    # model parameters
    parser.add_argument(
        "--output_file",
        type=str,
        help="name of output file")

    parser.add_argument(
        "--path",
        type=str,
        help="search path for csv log files")

    parser.add_argument(
        "--list_file",
        type=str,
        default=None,
        help="Name of list file (relative path from 'path'")

    parser.add_argument(
        "--BMC_Adj2Ref",
        type=bool,
        default=False,
        help="For log files from adjust_bmc_and_rerun.py"
    )

    parser.add_argument(
        "--laterality",
        type=bool,
        default=False,
        help="Write Laterality separately"
    )

    args = parser.parse_args(args)
    print(args)
    path = args.path
    output_file = args.output_file

    # search log files and write them to a list
    file_type = ".csv"
    log_file_list = io_utils.create_file_list(path, file_type)
    
    if args.list_file is not None:
        io_utils.write_file_list_to_file(log_file_list, path, args.list_file)

    # copy template file
    if args.laterality is False:
        template_file = running_file_path + "/Misc/result_fu_temp.dat"
    elif args.laterality is True:
        template_file = running_file_path + "/Misc/result_fu_temp_lat.dat"

    if args.BMC_Adj2Ref is True:
        template_file = running_file_path + "/Misc/result_fu_BMC_corr_temp.dat"
        print(template_file)
    
    shutil.copyfile(template_file, output_file)  # copy template file for output file

    with open(args.output_file, 'a+') as f:
        for log_file in log_file_list:
            dict = io_utils.read_csv_to_dict_2(log_file)
            list = re.split(r'/|_', log_file)
            id = list[-4] + "_" + list[-3][0]
            
            if args.laterality is True:
                id = list[-4]
                lat = list[-3][0]

            config = list[-2]

            fu_chars = dict["Strength_characteristics"]
            try:
                simulation_time = dict["Simulation Time"]
            except:
                simulation_time = 0

            try:
                preproc_time = dict["Preprocessing Time"]
            except KeyError:
                preproc_time = 0

            try:
                postproc_time = dict["Postprocessing Time"]
            except KeyError:
                postproc_time = 0

            try:
                cpus = dict["nprocs"]
            except KeyError:
                try:
                    cpus = dict["CPUS"]
                except:
                    cpus = 0
            except:
                print("No information about CPUS in log file - cpus set to 1")
                cpus = 1

            f_ult = fu_chars[0]
            d_ult = fu_chars[1]
            k_init = fu_chars[2]
            w_to_ult = fu_chars[3]

            sim_cpu_time = cpus * simulation_time
            tot_cpu_time = preproc_time + sim_cpu_time + postproc_time

            try:
                nodes = dict["Nodes"]
                elements = dict["Tot Elements"]
            except:
                pass

            try:
                ref_mesh_elements = dict["Elements Reference Mesh"]
                adj_mesh_elements = dict["Elements Adjusted Mesh"]
            except:
                pass

            if args.laterality is True and args.BMC_Adj2Ref is False:
                f.write("%s %s %s %s %s %s %s %s %s %s %s\n" % (id, lat, config, f_ult, d_ult, k_init, w_to_ult,
                                                                tot_cpu_time, sim_cpu_time, nodes, elements))
            elif args.laterality is False and args.BMC_Adj2Ref is False:
                f.write("%s %s %s %s %s %s %s %s %s %s\n" % (id, config, f_ult, d_ult, k_init, w_to_ult, tot_cpu_time,
                                                             sim_cpu_time, nodes, elements))
            elif args.BMC_Adj2Ref is True:
                f.write("%s %s %s %s %s %s %s %s\n" % (id, config, f_ult, d_ult, k_init, w_to_ult, ref_mesh_elements,
                                                       adj_mesh_elements))


if __name__ == "__main__":
    # args = parser.parse_args()
    # main(args)
    main(sys.argv[1:])
