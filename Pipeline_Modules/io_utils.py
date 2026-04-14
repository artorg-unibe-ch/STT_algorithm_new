"""
Created on 20210226
@author: Yvan Gugler
{

}
"""
# imports
from Pipeline_Modules.lib import *


def read_config_file(filename):
    """
    Reads YAML configuration file and returns a dictionary with the configuration settings.
    :param filename: string
    :return: config dictionary
    """
    with open(filename, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def create_file_list(path, extension):
    """
    From path look for files with specific extension.
    :param path: string
    :param extension: string
    :return: filelist: list of strings
    """
    os.chdir(path)
    file_list = glob.glob("**/*" + extension, recursive=True)  # look for files recursively
    return file_list


def parse_filename(file_path, pattern, positions):
    """
    Parses the complete path of file (incl. filename and extension) according to pattern. Positions defines the elements
    of the parsed filename that are extracted.
    :param file_path: string
    :param pattern: string
    :param positions: dictionary with the positions of Sample_ID, Laterality and exp_config in the parsed list.
    :return:
    """
    parsed_filename = re.split(pattern, file_path)
    sampleID = parsed_filename[positions["Sample_ID"]]
    if positions["Laterality"] is not None:
        laterality = parsed_filename[positions["Laterality"]]
    else:
        laterality = None
    if positions["exp_config"] is not None:
        exp_config = parsed_filename[positions["exp_config"]]
    else:
        exp_config = None

    return sampleID, exp_config, laterality


def write_message_file(msg_file, sample, err_str):
    """

    :param msg_file:
    :param sample:
    :param err_str:
    :return:
    """
    f = open(msg_file, "a")
    f.write("Error messages:\n")
    f.write("ID:  " + sample + "\n")
    f.write(err_str)
    f.close()


def write_csv_from_dict(filename, dict, types):
    """
    needs specification of types
    keys as headers in one row
    values for each key in row below
    Parameters
    ----------
    filename :
    dict :

    Returns
    -------

    """
    keys = list(dict.keys())
    list_len = np.shape(dict[keys[0]])[0]

    dt = np.dtype(types)
    data_array = np.empty((list_len,), dt)

    for i in range(0, list_len):
        line = []
        for key in dict.keys():
            line.append(dict[key][i])

        data_array[i] = tuple(line)

    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(list(dict.keys()))

        for i in range(0, list_len):
            writer.writerow(data_array[i])


def read_csv_to_dict(file_to_load, line_numbers, types=None):
    """
    Function that reads lines from a csv file into a python dictionary.

    """
    if types is None:
        n = len(line_numbers)
        types = [float] * n

    with open(file_to_load, 'r') as datafile:
        reader = csv.reader(datafile, delimiter=',')
        first_row = next(reader)
        names = [first_row[i] for i in line_numbers]
        data = {title: np.empty(0) for title in names}
        for row in reader:
            if row == first_row:
                pass
            else:
                for key in data.keys():
                    i = names.index(key)
                    data[key] = np.append(data[key], row[line_numbers[i]]).astype(types[i])
    return data


def merge_list_of_pdfs_to_single_pdf(list_of_pdfs, output_file_name):
    """

    @param list_of_pdfs:
    @type list_of_pdfs:
    @param output_file_name:
    @type output_file_name:
    @return:
    @rtype:
    """
    merger = PdfFileMerger()

    for pdf in list_of_pdfs:
        # merger.append(pdf)
        merger.append(PdfFileReader(open(pdf, 'rb')))

    merger.write(output_file_name)
    merger.close()


def create_file_list(path, extension):
    """
    From path look for files with specific extension.
    :param path: string
    :param extension: string
    :return: filelist: list of strings
    """
    os.chdir(path)
    # file_list = glob.glob("**/*" + extension, recursive=True)  # look for files recursively
    file_list = glob.glob(path + "/**/*" + extension, recursive=True)  # look for files recursively
    return file_list


def write_file_list_to_file(file_list, path, list_file_name):
    """

    @param file_list:
    @type file_list:
    @param path:
    @type path:
    @param list_file_name:
    @type list_file_name:
    @return:
    @rtype:
    """
    file_to_write = path + "/" + list_file_name
    with open(file_to_write, 'w') as f:
        for filename in file_list:
            f.write('%s \n' % filename)


def copy_list_of_files(file_list, destination):
    """

    @param file_list:
    @type file_list:
    @param destination:
    @type destination:
    @return:
    @rtype:
    """

    command = "xargs -a " + file_list + " cp -t " + destination
    subprocess.call(command, shell=True)


def copy_list_of_files_remote(file_list, zip_file_name, destination):
    """

    @param file_list:
    @type file_list:
    @param destination:
    @type destination:
    @return:
    @rtype:
    """

    zipObj = ZipFile(zip_file_name, 'w')

    for file in file_list:
        file_split = file.split("/")

        filename = file_split[-1]
        name_length = len(filename)
        path = file[0: -name_length]
        zipObj.write(path + filename, filename)

    zipObj.close()

    command =  "scp " + zip_file_name + " " +  destination
    subprocess.call(command, shell=True)


def create_FU_summary(out_file_list, template_file, output_file, rows_to_skip=1):
    """

    @param out_file:
    @type out_file:
    @param template_file:
    @type template_file:
    @return:
    @rtype:
    """
    shutil.copyfile(template_file, output_file)  # copy template file for output file

    with open(output_file, 'a+') as f:
        for out_file in out_file_list:
            fu_array = np.loadtxt(out_file, skiprows=rows_to_skip)
            list = re.split(r'/|_', out_file)
            id = list[14] + "_" + list[15][0]
            config = list[16][0: -4]
            disp = fu_array[:, 0]
            force = fu_array[:, 1]

            f_ult = np.max(force)
            max_step = np.argmax(force) + 1
            nsteps = np.size(force)

            # stiffness computation
            d_disp = np.diff(disp)
            d_force = np.diff(force)
            stiffness = np.divide(d_force, d_disp)
            k_init = stiffness[0]

            f.write(" %s %s %s %s %s %s \n" % (id, config, f_ult, k_init, nsteps, max_step))



def create_FU_summary_2(out_file_list, template_file, output_file, rows_to_skip=1):
    """

    @param out_file:
    @type out_file:
    @param template_file:
    @type template_file:
    @return:
    @rtype:
    """
    shutil.copyfile(template_file, output_file)  # copy template file for output file

    with open(output_file, 'a+') as f:
        for out_file in out_file_list:
            fu_array = np.loadtxt(out_file, skiprows=rows_to_skip)
            list = re.split(r'/|_', out_file)
            id = list[14] + "_" + list[15][0]
            config = list[16][0: -4]
            disp = fu_array[:, 0]
            force = fu_array[:, 1]

            peaks, peak_props = signal.find_peaks(force)

            try:
                f_ult = force[peaks[0]]
            except:
                f_ult = np.max(force)

            max_step = np.argmax(force) + 1
            nsteps = np.size(force)

            # stiffness computation
            d_disp = np.diff(disp)
            d_force = np.diff(force)
            stiffness = np.divide(d_force, d_disp)
            k_init = stiffness[0]

            f.write(" %s %s %s %s %s %s \n" % (id, config, f_ult, k_init, nsteps, max_step))


def create_FU_summary_3(out_file_list, template_file, output_file, distance=4, rows_to_skip=1):
    """

    @param out_file:
    @type out_file:
    @param template_file:
    @type template_file:
    @return:
    @rtype:
    """
    shutil.copyfile(template_file, output_file)  # copy template file for output file

    with open(output_file, 'a+') as f:
        for out_file in out_file_list:
            fu_array = np.loadtxt(out_file, skiprows=rows_to_skip)
            list = re.split(r'/|_', out_file)
            id = list[-3] + "_" + list[-2][0]
            config = list[-1][0: -4]
            disp = fu_array[:, 0]
            force = fu_array[:, 1]

            d = np.abs(disp - distance)
            last_ind = np.argmin(d)
            print(last_ind)

            f_ult = np.max(force[0:last_ind+1])

            max_step = np.argmax(force) + 1
            nsteps = np.size(force)

            # stiffness computation
            d_disp = np.diff(disp)
            d_force = np.diff(force)
            stiffness = np.divide(d_force, d_disp)
            k_init = stiffness[0]

            f.write(" %s %s %s %s %s %s \n" % (id, config, f_ult, k_init, nsteps, max_step))


def read_txt_file_to_list(text_file):
    """

    @param text_file:
    @type text_file:
    @return:
    @rtype:
    """
    list = []
    with open(text_file, 'r') as f:
        for line in f:
            splitted_line = line.split(' ')
            if splitted_line[0] == '':
                splitted_line.pop(0)
            list.append(splitted_line)

    return list


def write_csv_from_dict_2(file, dict, delimiter_sign=';'):
    """
    Writes a dictionary to a csv-file. Eack key-value pair is written in a row. Iterabels are written as lists/lists of
    lists, etc.
    Parameters
    ----------
    file :
    dict :

    Returns
    -------

    """
    if os.path.isfile(file):
        mode = 'a'
    else:
        mode = 'w'

    with open(file, mode) as csvfile:
        for key in dict:
            csvwriter = csv.writer(csvfile, delimiter=delimiter_sign)
            if isinstance(dict[key], Iterable):
                try:
                    dict[key] = dict[key].tolist()
                except:
                    pass
            csvwriter.writerow([key, dict[key]])

        csvfile.close()


def read_csv_to_dict_2(csv_file, delim=";"):
    """

    Parameters
    ----------
    csv_file :
    delim :

    Returns
    -------

    """
    # read in csv file to dict using pandas / each line (row) corresponds to a key/value pair
    dict = {row[0]: row[1] for _, row in pd.read_csv(csv_file, delimiter=delim, header=None).iterrows()}
    d = dict.copy()  # copy the dict

    # till now all the values are strings -> convert them
    for k in dict:
        try:
            d[k] = int(dict[k])  # integers
        except:
            try:
                d[k] = float(dict[k])  # floats
            except:
                try:
                    res = d[k].strip('][').split(', ')  # make lists of all others
                    d[k] = res
                except:
                    pass

    for k in dict:
        if type(d[k]) == list:
            if len(d[k]) == 1:
                try:
                    d[k] = eval(dict[k][0])  # built-in words (e.g. None, False)
                except:
                    d[k] = d[k][0]  # strings
            else:
                try:
                    d[k] = eval(dict[k])  # values that are dictionaries
                except:
                    pass

    return d

