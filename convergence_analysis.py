"""

Author: Yvan Gugler
Date: 2022-02-16
"""

from Pipeline_Modules.lib import *
from Pipeline_Modules import io_utils
import rerun_new_resolution

running_file_path = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser()

# model parameters
parser.add_argument(
    "--path",
    type=str,
    help="csv-log-file")

parser.add_argument(
    "--batches",
    type=int,
    help="number of batches for processing")

parser.add_argument(
    "--batch_nbr",
    type=int,
    help="Batch nbr. of batches")

parser.add_argument(
    "--el_sizes",
    nargs="*",  # 0 or more values expected => creates a list
    type=float,
    default=[1.5, 1.0],
    help="list with element sizes")

parser.add_argument(
    "--cpus",
    type=int,
    help="CPUS for abaqus simulation")


def main(args):
    file_list = io_utils.create_file_list(args.path, ".csv")
    nbr_of_files_batch = int(len(file_list) / args.batches)

    batch_nbr = args.batch_nbr

    ind0 = (batch_nbr-1) * nbr_of_files_batch
    ind1 = batch_nbr * nbr_of_files_batch

    if batch_nbr < nbr_of_files_batch:
        file_list = file_list[ind0:ind1]
    else:
        file_list = file_list[ind0:]

    print(file_list)

    for file in file_list:
        print(file)
        for size in args.el_sizes:
            print(size)
            print(args.cpus)
            rerun_new_resolution.main(["--csv_file", file, "--new_voxel_size", str(size), "--cpus", str(args.cpus)])


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

