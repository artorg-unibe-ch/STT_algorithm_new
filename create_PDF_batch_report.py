"""
PDF report of all cases in a badge.
Author: Yvan Gugler
Date: 2022-03-08
"""

from Pipeline_Modules.lib import *
from Pipeline_Modules import io_utils


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
        help="search path for pdf report files of single cases")

    args = parser.parse_args(args)

    path = args.path
    output_file = args.output_file

    # search log files and write them to a list
    file_type = ".pdf"
    pdf_file_list = io_utils.create_file_list(path, file_type)

    io_utils.merge_list_of_pdfs_to_single_pdf(pdf_file_list, output_file)


if __name__ == "__main__":
    main(sys.argv[1:])
