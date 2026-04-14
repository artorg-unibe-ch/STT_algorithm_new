"""
Created on 20210226
@author: Yvan Gugler
{

}
"""
# imports
from Pipeline_Modules.lib import *
from Pipeline_Modules import io_utils


def create_abaqus_macro(macrofile, odbfile, cut_origin, png_bvtv_file, png_dam_file):
    """
    Creates an abaqus python macrofile that opens the odbfile and generates a view cut parallel to the xz plane through
    the head center of the femur. A .png file is saved for BVTV distribution (frame 1) and damage distribution (last
    frame).
    :param macrofile:
    :param odbfile:
    :param cut_origin:
    :param png_bvtv_file:
    :param png_dam_file:
    :return:
    """
    g = open(macrofile, "w")
    string = "\n\
    from abaqus import *\n\
    from abaqusConstants import *\n\
    import __main__\n\n\
    def Macro1():\n\
        import visualization\n\
        import xyPlot\n\
        import displayGroupOdbToolset as dgo\n"
    string = textwrap.dedent(string)
    g.write(string)
    g.write("    o1 = session.openOdb(name='%s')\n" % odbfile)
    string ="    session.viewports['Viewport: 1'].setValues(displayedObject=o1)\n\
    session.viewports['Viewport: 1'].view.setProjection(projection=PARALLEL)\n\
    session.viewports['Viewport: 1'].view.setValues(session.views['Bottom'])\n\
    session.viewports['Viewport: 1'].odbDisplay.ViewCut(name='Cut-4', shape=PLANE,\n\
    origin=(%f, %f, %f), normal=(0.0, -1., 0.), axis2=(0.0, 1., 0.))\n" % cut_origin
    g.write(string)
    string = "    session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(variableLabel='SDV2', outputPosition=ELEMENT_CENTROID, )\n\
    session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=CONTOURS_ON_DEF)\n\
    session.viewports['Viewport: 1'].odbDisplay.contourOptions.setValues(contourType=QUILT)\n\
    session.viewports['Viewport: 1'].viewportAnnotationOptions.setValues(legendBox=OFF, title=OFF, state=OFF, annotations=OFF, compass=OFF)\n\
    session.printOptions.setValues(vpDecorations=OFF)\n\
    session.printToFile(fileName='%s',format=PNG)\n" % png_dam_file
    g.write(string)
    g.write("    session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=1 )\n")
    g.write("    session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(variableLabel='SDV21', outputPosition=ELEMENT_CENTROID,)\n")
    g.write("    session.printToFile(fileName='%s',format=PNG)\n" % png_bvtv_file)
    g.write("    session.odbs['%s'].close()\n\n\n" % odbfile)
    g.write("Macro1()\n")
    g.close()


def remove_empty_entries_list(input_list):
    """
    Remove empty strings from a list of strings.
    :param input_list: list of strings
    """
    str_list = list(filter(None, input_list))
    return str_list


def datfilereader(dat_filename):
    """
    Read out displacement and force data for a single node from dat file.
    The writing of the results in the .dat-file needs to be specified in the abaqus input file (key word: *NODE PRINT)
    :param dat_filename: string that corresponds to the file name
    """
    outfilename = dat_filename.replace(".dat", ".txt")
    # Read the .dat file
    with open(dat_filename, "r") as infile:
        lines = infile.readlines()
    # Lists used
    ref_nodedata = []
    disp = []
    force = []
    # Ref node outputs
    # Displacement vector U
    U1 = " "
    U2 = " "
    U3 = " "
    UR1 = " "
    UR2 = " "
    UR3 = " "
    # Rectionforce vector RF
    RF1 = " "
    RF2 = " "
    RF3 = " "
    RM1 = " "
    RM2 = " "
    RM3 = " "
    j = 0
    for i in range(0, len(lines)):
        if lines[i].find("U1") > -1:
            j = j + 1  # value of increment
            # Split U and R lines
            member1 = lines[i + 3].split(" ")
            member2 = lines[i + 12].split(" ")
            # Extract different U components
            member1_red = remove_empty_entries_list(member1)
            U1 = float(member1_red[1])
            U2 = float(member1_red[2])
            U3 = float(member1_red[3])
            UR1 = float(member1_red[4])
            UR2 = float(member1_red[5])
            UR3 = float(member1_red[6])
            try:
                # Extract different R components
                member2_red = remove_empty_entries_list(member2)
                # Clean member2_pass from "+" and replace with "E+"
                for k in range(0, len(member2_red)):
                    if member2_red[k].find("+") != -1:
                        if member2_red[k].find('E+') != -1:
                            pass
                        else:
                            member2_red[k] = member2_red[k].replace("+", "E+")
                RF1 = float(member2_red[1])
                RF2 = float(member2_red[2])
                RF3 = float(member2_red[3])
                RM1 = float(member2_red[4])
                RM2 = float(member2_red[5])
                RM3 = float(member2_red[6])
            except:
                RF1 = 0.0
                RF2 = 0.0
                RF3 = 0.0
                RM1 = 0.0
                RM2 = 0.0
                RM3 = 0.0

            disp.append([U1, U2, U3, UR1, UR2, UR3])
            force.append([RF1, RF2, RF3, RM1, RM2, RM3])
            ref_nodedata.append(
                str(j) + " " + str(U1) + " " + str(U2) + " " + str(U3) + " "
                + str(UR1) + " " + str(UR2) + " " + str(UR3) + " "
                + str(RF1) + " " + str(RF2) + " " + str(RF3) + " "
                + str(RM1) + " " + str(RM2) + " " + str(RM3)
            )
    # Create the output files
    with open(outfilename, "w") as out:
        out.write(
            """
*********************************************************
* Displacement and reaction force of the reference node *
*********************************************************
inc U1 U2 U3 UR1 UR2 UR3 RF1 RF2 RF3 RM1 RM2 RM3  
0  0  0  0  0  0  0  0  0  0  0  0  0
"""
        )
        for inc in range(0, len(ref_nodedata)):
            out.write(ref_nodedata[inc] + "\n")

    return disp, force


def extract_force_disp(dat_file, DOF, output_file=None):
    """
    Extract force and displacement (of specific DOF) from a dat_file, which contains the results for all 6 DOFs. The
    vectors can be written to a text file.
    :param dat_file: string
    :param DOF: integer
    :param output_file: string
    :return: disp, force (numpy arrays)
    """
    disp, force = datfilereader(dat_file)
    disp = np.transpose(disp)
    force = np.transpose(force)

    disp = np.absolute(disp[DOF])
    disp = np.insert(disp, 0, 0)
    force = np.absolute(force[DOF])
    force = np.insert(force, 0, 0)
    df = np.ones([np.shape(disp)[0], 2])
    df[:, 0] = disp
    df[:, 1] = force

    if output_file is not None:
        list = '\n'.join(
            ["Disp   Force",
             '\n'.join('\t'.join('%0.3f' % x for x in y) for y in df),
             ])
        with open(output_file, "w") as output:
            output.write(list)
    return disp, force


def compute_force_disp_characteristics(force, disp):
    """
    Compute the characteristics of force-displacement curve (fult, dult, init_stiff, Work to fult)
    :param force: np.array containing the force
    :param disp: np.array containing the displacement
    :return: fult, dult, init_stiff, W
    """
    fult = np.max(force)
    fult_arg = np.argmax(force)
    dult = disp[fult_arg]
    init_stiff = force[1]/disp[1]
    W = np.trapz(force[0:fult_arg+1], disp[0:fult_arg+1])
    return fult, dult, init_stiff, W


def compute_force_disp_characteristics_new(force_in, disp_in, f_ult_criterion=None):
    """
    Compute the characteristics of force-displacement curve (fult, dult, init_stiff, Work to fult)
    :param force_in: np.array containing the force
    :param disp_in: np.array containing the displacement
    :return: fult, dult, init_stiff, W
    """
    force = np.copy(force_in)
    disp = np.copy(disp_in)

    fmax = np.max(force)
    fmax_arg = np.argmax(force)
    d_at_fmax = disp[fmax_arg]

    if f_ult_criterion is None:
        fult = fmax
        dult = d_at_fmax
        work_to_fracture = np.trapz(force[0:fmax_arg + 1], disp[0:fmax_arg + 1])

    else:
        d_normalized = 1 / f_ult_criterion[1] * disp

        if d_at_fmax / f_ult_criterion[1] < f_ult_criterion[0]:
            fult = fmax
            dult = d_at_fmax
            work_to_fracture = np.trapz(force[0:fmax_arg + 1], disp[0:fmax_arg + 1])

        else:
            ind1 = np.argmax(d_normalized >= f_ult_criterion[0])
            ind0 = ind1 - 1

            force_0 = force[ind0]
            force_1 = force[ind1]
            d_n0 = d_normalized[ind0]
            d_n1 = d_normalized[ind1]
            d_0 = disp[ind0]
            d_1 = disp[ind1]

            fult = (f_ult_criterion[0] - d_n0) / (d_n1 - d_n0) * (force_1 - force_0) + force_0
            dult = (f_ult_criterion[0] - d_n0) / (d_n1 - d_n0) * (d_1 - d_0) + d_0
            fult = np.around(fult, 2)
            dult = np.around(dult, 2)

            force = np.insert(force, ind1, fult)
            disp = np.insert(disp, ind1, dult)

            work_to_fracture = np.trapz(force[0:ind1+1], disp[0:ind1+1])

    work_to_fracture = np.around(work_to_fracture, 2)
    init_stiff = force[1]/disp[1]
    init_stiff = np.around(init_stiff, 2)

    return fult, dult, init_stiff, work_to_fracture, disp, force


def plot_force_disp(force, disp, showfig=False, savefig=False, filename=None, f_ult=None):
    """
    Create force-displacement curve and save to file.
    :param force:
    :param disp:
    :return:
    """
    fig = plt.figure()

    ax = plt.axes()
    ax.set_xlim((0, disp[-1]))
    ax.set_ylim((0, np.max(force) * 1.1))
    ax.set_title("Force-displacement curve")
    ax.set_xlabel("Displacement [mm]")
    ax.set_ylabel("Force [N]")

    ax.plot(disp, force, color="blue", marker="o", markersize=3)
    if f_ult is not None:
        index = np.argwhere(force == f_ult)[0][0]
        d_ult = disp[index]
        ax.scatter(d_ult, f_ult, color="red", marker="o")

    fig.add_axes(ax)
    if showfig:
        fig.show()
    if savefig:
        fig.savefig(filename)
    return fig


def pdf_report(job, png_bvtv, png_dam, png_fu, image_props, mesh_props, fu_characteristics, flags):
    """
    Function that generates a pdf report of the simulation.
    :param job: job name (string)
    :param png_bvtv: image file of bvtv distribution
    :param png_dam: image file of damage distribution
    :param png_fu: image file with force-displacement curve
    :param png_coord: image file ofcoordinate system detection
    :param image_props: list containing bone properties from original image (volume of mask, BMC, mean BMD)
    :param mesh_props: list containing bone properties from FE mesh (volume, BMC, mean BMD)
    :param fu_characteristics: list containing Fult, dult, k, Wult
    :param flags: list with flags for UMAT (DENSFL, VISCFL, PYFL)
    :return:
    """
    image_props.insert(0, "Image")  # prepare the arguments for the work below
    mesh_props.insert(0, "FE model")
    vol_mass_info = [image_props, mesh_props]
    fu_characteristics = [fu_characteristics]
    flags = [flags]

    pdf_file = FPDF(format="A4", unit="mm")  # generate pdf file
    left_margin = 15
    top_margin = 25
    pdf_file.set_margins(left_margin, top_margin)
    page_width = pdf_file.w
    epw = page_width - 2 * left_margin  # effective page width

    pdf_file.add_page()
    pdf_file.set_font('Arial', 'B', 16)  # title
    pdf_file.cell(epw, 10, 'Report - Femur: ' + job, border=1, ln=1, align='C')

    cur_y = pdf_file.y
    pdf_file.set_font('Arial', '', 10)
    pdf_file.cell(epw/2, 5, 'BVTV distribution', border=0)
    pdf_file.image(png_bvtv, x=left_margin, y=cur_y+5, w=epw/2)  # BVTV image

    pdf_file.set_xy(left_margin+epw/2, cur_y)
    pdf_file.cell(epw/2, 5, 'Damage distribution', border=0)
    pdf_file.image(png_dam, x=left_margin + epw/2, y=cur_y+5, w=epw/2)  # damage image
    pdf_file.set_xy(left_margin, epw/2 + 15)
    del cur_y

    ht = pdf_file.font_size
    cw = epw/4

    vol_mass_info.insert(0, ["", "Vol. [mm3]", "BMC [mg]", "Mean BMD [mg/cm3]"])
    vol_mass_info.append(["Ratio", round(vol_mass_info[2][1]/vol_mass_info[1][1], 4),
                          round(vol_mass_info[2][2]/vol_mass_info[1][2], 4),
                          round(vol_mass_info[2][3]/vol_mass_info[1][3], 4)])
    for row in vol_mass_info:
        for col in row:
            # Enter data in columns
            # Notice the use of the function str to coerce any input to the
            # string type. This is needed
            # since pyFPDF expects a string, not a number.
            pdf_file.cell(cw, ht, str(col), border=1)
        pdf_file.ln(ht)
    del cw

    pdf_file.set_y(pdf_file.y+5)
    pdf_file.image(png_fu, x=left_margin, y=pdf_file.y, w=epw/2)

    cw = epw/4
    cur_y = pdf_file.y
    x_pos = left_margin+epw/2
    # pdf_file.image(png_coord, x=x_pos, y=cur_y, w=epw/2)

    cur_y = pdf_file.y + 75
    pdf_file.set_xy(left_margin, cur_y)
    fu_characteristics.insert(0, ["Fult [N]", "dult [mm]", "k [N/mm]", "W [Nmm]"])
    fu_characteristics = map(list, zip(*fu_characteristics))

    i = 1
    for row in fu_characteristics:
        for col in row:
            pdf_file.cell(cw, ht, str(col), border=1)
        y_actual = cur_y + i * ht
        pdf_file.set_xy(left_margin, y_actual)
        i += 1
    del cur_y, x_pos

    flags.insert(0, ["DENSFL", "VISCFL", "PYFL"])
    flags = map(list, zip(*flags))
    cur_y = y_actual + ht
    pdf_file.set_xy(left_margin, cur_y)

    for row in flags:
        for col in row:
            pdf_file.cell(cw, ht, str(col), border=1)
        cur_y += ht
        pdf_file.set_xy(left_margin, cur_y)
        i += 1

    filename = job + ".pdf"
    pdf_file.output(filename)


def write_sample_summary_file(femur_dict, config_dict, filenames):
    """

    :param femur_dict:
    :param config_dict:
    :param filenames:
    :return:
    """
    # femur_dict
    spacing = femur_dict["Spacing"]
    orig_size = femur_dict["Original Size"]
    corrected_size = femur_dict["Size After Standardization"]
    added_layer = femur_dict["Layer"]

    scan_head_down = config_dict["invert_anteropost_axis"]
    axis_flip = femur_dict["Axis Flip"]
    laterality = femur_dict["Laterality"]
    head_center = femur_dict["Head Center"]
    head_radius = femur_dict["Head Radius"]
    v_dia = femur_dict["Diaphysis Axis"]
    v_neck = femur_dict["Neck Axis"]
    intersection_neck_shaft = femur_dict["Neck-Shaft-Intersection"]
    neck_shaft_angle = femur_dict["CCD Angle"]
    diaphysis_base_point = femur_dict["Diaphysis-Base"]

    trans_matrix_fall = femur_dict["Trans Matrix Fall"]
    trans_matrix_stance = femur_dict["Trans Matrix Stance"]
    if trans_matrix_fall is None:
        trans_matrix = trans_matrix_stance
    elif trans_matrix_stance is None:
        trans_matrix = trans_matrix_fall

    initial_mean_bmd = femur_dict["Initial Mean BMD"]
    initial_bmc = femur_dict["Initial BMC"]
    initial_tot_vol = femur_dict["Initial Volume"]
    initial_bone_vol = femur_dict["Initial Bone Volume"]
    initial_bvtv = initial_bone_vol / initial_tot_vol
    mesh_mean_bmd = femur_dict["Final Mean BMD"]
    mesh_bmc = femur_dict["Final BMC"]
    mesh_tot_vol = femur_dict["Final Volume"]
    mesh_bone_vol = femur_dict["Final Bone Volume"]
    mesh_bvtv = mesh_bone_vol / mesh_tot_vol

    bmc_ratio = mesh_bmc / initial_bmc
    vol_ratio = mesh_tot_vol / initial_tot_vol
    bvtv_ratio = mesh_bvtv / initial_bvtv
    # bvol_ratio = mesh_bone_vol / initial_bone_vol

    elems_tot = femur_dict["Tot Elements"]
    elems_bone = femur_dict["Bone Elements"]

    comp_time = femur_dict["Computation Time"]
    pre_time = femur_dict["Preprocessing Time"]
    sim_time = femur_dict["Simulation Time"]
    post_time = femur_dict["Postprocessing Time"]

    # config_dict
    el_size = config_dict["el_size"]
    exp_config = config_dict["exp_config_image"]
    dia_angle_fall = config_dict["dia_angle_fall"]
    rot_angle = config_dict["rot_angle"]
    dia_angle_stance = config_dict["dia_angle_stance"]

    # filenames
    imfile = filenames["Image"]
    input_file_name = filenames["Input"]
    mesh_file_name = filenames["Mesh"]
    mask_mhd_file = filenames["FEA_Mask"]
    bmd_mhd_file = filenames["FEA_BMD"]
    summary_file_name = filenames["Summary File"]

    summary = "\n".join(
        [
            """
******************************************************************
**                         SUMMARY FILE                         **
**                hFE pipeline for the femur                    **
******************************************************************""",
            "File                 : {}".format(imfile),
            "System computed on   : {}".format(socket.gethostname()),
            "Simulation Type      : isotropic",
            "*****************************************************************",
            "Image Size           : {:.3f}, {:.3f}, {:.3f}".format(*orig_size),
            "Corrected Size       : {:.3f}, {:.3f}, {:.3f}".format(*corrected_size),
            "Layer added          : {:d}".format(added_layer),
            "Laterality           : {}".format(laterality),
            "Spacing              : {:.3f}, {:.3f}, {:.3f} mm".format(*spacing),
            "FE element size      : {:.3f} mm".format(el_size),
            "Nbr. of elements     : {:d}".format(elems_tot),
            "Nbr. of bone elements: {:d}".format(elems_bone),
            "Scanned head down    :  {}".format(scan_head_down),
            "Flipped              :  {}".format(axis_flip),
            "******************************************************************",
            "Geometry                                                          ",
            "-------------------------------------------------------------------",
            "Head Center          : {:.3f}, {:.3f}, {:.3f}, ".format(*head_center),
            "Head Radius          : {:.3f}".format(head_radius),
            "Dia. Axis            :{:.3f}, {:.3f}, {:.3f}".format(*v_dia),
            "Neck Axis            :{:.3f}, {:.3f}, {:.3f}".format(*v_neck),
            "Neck-Shaft-Inter.    :{:.3f}, {:.3f}, {:.3f}".format(*intersection_neck_shaft),
            "Neck-Shaft-Angle.    :{:.3f}° ".format(neck_shaft_angle),
            "Dia. Base Point      :{:.3f}, {:.3f}, {:.3f}".format(*diaphysis_base_point),
            "Trans. Matrix        :", '\n'.join('\t'.join('%0.3f' %x for x in y) for y in trans_matrix),
            "******************************************************************",
            "Configuration of the simulation                                   ",
            "-------------------------------------------------------------------",
            "Exp. Configuration       : {}".format(exp_config),
            "Dia. Angle Fall          : {:.2f}".format(dia_angle_fall),
            "Rot. Angle               : {:.2f}".format(rot_angle),
            "Dia. Angle Stance        : {:.2f}".format(dia_angle_stance),
            "******************************************************************",
            "Volumes of mask",
            "-------------------------------------------------------------------",
            "Mask Volume image    : {:.1f} mm^3".format(initial_tot_vol),
            "Mask Volume FE mesh  : {:.1f} mm^3".format(mesh_tot_vol),
            "Mask Volume ratio    : {:.3f} ".format(vol_ratio),
            "******************************************************************",
            "BMD and BVTV",
            "-------------------------------------------------------------------",
            "Mean BMD image           : {:.2f} mgHA/ccm".format(initial_mean_bmd),
            "Mean BMD mesh            : {:.2f} mgHA/ccm".format(mesh_mean_bmd),
            "BVTV image               : {:.2f}%".format(initial_bvtv * 100.0),
            "BVTV mesh                : {:.2f}%".format(mesh_bvtv * 100.0),
            "BVTV ratio               : {:.3f}%".format(bvtv_ratio * 100.0),
            "******************************************************************",
            "BMC",
            "-------------------------------------------------------------------",
            "BMC image            : {:.1f} mgHA".format(initial_bmc),
            "BMC mesh             : {:.1f} mgHA".format(mesh_bmc),
            "BMC ratio            : {:.3f} ".format(bmc_ratio),
            "******************************************************************",
            "Files",
            "-------------------------------------------------------------------",
            "Input File                 : {}".format(input_file_name),
            "Mesh File                  : {}".format(mesh_file_name),
            "Mask MHD File              : {}".format(mask_mhd_file),
            "BMD MHD File               : {}".format(bmd_mhd_file),
            "******************************************************************",
            "Computation time"
            "-------------------------------------------------------------------",
            "Preprocessing time        : {:.1f} s".format(pre_time),
            "Simulation time           : {:.1f} s".format(sim_time),
            "Postprocessing time       : {:.1f} s".format(post_time),
            "Total computation time    : {:.1f} s".format(comp_time),
        ]
    )

    print(summary)
    with open(summary_file_name, "w") as sumFile:
        sumFile.write(summary)

def write_sample_summary_file_new(femur_dict, config_dict, summary_file_name):
    """

    Parameters
    ----------
    femur_dict :
    config_dict :
    summary_file_name :

    Returns
    -------

    """
    io_utils.write_csv_from_dict_2(summary_file_name, config_dict)
    io_utils.write_csv_from_dict_2(summary_file_name, femur_dict)

    print("Summary File written")


def pdf_report_new(job, png_coord_sys1, png_coord_sys2, png_coord_sys3, png_coord_sys4, png_hist,
                   png_bvtv, png_dam, png_fu, image_props, mesh_props, fu_characteristics, flags):
    """
    Function that generates a pdf report of the simulation.
    :param job: job name (string)
    :param png_bvtv: image file of bvtv distribution
    :param png_dam: image file of damage distribution
    :param png_fu: image file with force-displacement curve
    :param png_coord: image file ofcoordinate system detection
    :param image_props: list containing bone properties from original image (volume of mask, BMC, mean BMD)
    :param mesh_props: list containing bone properties from FE mesh (volume, BMC, mean BMD)
    :param fu_characteristics: list containing Fult, dult, k, Wult
    :param flags: list with flags for UMAT (DENSFL, VISCFL, PYFL)
    :return:
    """
    image_props.insert(0, "Image")  # prepare the arguments for the work below
    mesh_props.insert(0, "FE model")
    vol_mass_info = [image_props, mesh_props]
    fu_characteristics = [fu_characteristics]
    flags = [flags]

    pdf_file = FPDF(format="A4", unit="mm")  # generate pdf file
    left_margin = 15
    top_margin = 25
    pdf_file.set_margins(left_margin, top_margin)
    page_width = pdf_file.w
    epw = page_width - 2 * left_margin  # effective page width

    pdf_file.add_page()
    pdf_file.set_font('Arial', 'B', 16)  # title
    pdf_file.cell(epw, 10, 'Report - Femur: ' + job, border=1, ln=1, align='C')

    cur_y = pdf_file.y

    pdf_file.set_font('Arial', '', 8)
    pdf_file.cell(epw, 4, 'Femur Coordinate system and extension', border=0)
    pdf_file.image(png_coord_sys1, x=left_margin, y=cur_y+4, w=epw/4)  # coord. system image 1

    pdf_file.set_xy(left_margin+epw/4, cur_y)
    pdf_file.image(png_coord_sys2, x=left_margin + epw/4, y=cur_y+4, w=epw/4)  # coord. system image 2

    pdf_file.set_xy(left_margin+epw/2, cur_y)
    pdf_file.image(png_coord_sys3, x=left_margin + epw/2, y=cur_y+4, w=epw/4)  # coord. system image 3

    pdf_file.set_xy(left_margin+3*epw/4, cur_y)
    pdf_file.image(png_coord_sys4, x=left_margin + 3*epw/4, y=cur_y+4, w=epw/4)  # coord. system image 3

    pdf_file.set_xy(left_margin, cur_y+50)
    ht = pdf_file.font_size
    cw = epw/4

    vol_mass_info.insert(0, ["", "Vol. [mm3]", "BMC [mg]", "Mean BMD [mg/cm3]"])
    vol_mass_info.append(["Ratio", round(vol_mass_info[2][1]/vol_mass_info[1][1], 4),
                          round(vol_mass_info[2][2]/vol_mass_info[1][2], 4),
                          round(vol_mass_info[2][3]/vol_mass_info[1][3], 4)])

    for row in vol_mass_info:
        for col in row:
            # Enter data in columns
            # Notice the use of the function str to coerce any input to the
            # string type. This is needed
            # since pyFPDF expects a string, not a number.
            pdf_file.cell(cw, ht, str(col), border=1)
        pdf_file.ln(ht)
    del cw

    cw = epw/2
    cur_y = pdf_file.y + 4 * ht

    pdf_file.set_font('Arial', '', 8)
    pdf_file.cell(epw, 4, 'Histogram image', border=0)
    pdf_file.image(png_hist, x=left_margin, y=cur_y, w=epw)  # histogram 1 (image)
    del cur_y

    #___________________________________________________________________________________________________________________
    # Second page
    # __________________________________________________________________________________________________________________

    pdf_file.add_page()

    cur_y = pdf_file.y
    pdf_file.set_font('Arial', '', 8)
    pdf_file.cell(epw/2, 5, 'BVTV distribution', border=0)
    pdf_file.image(png_bvtv, x=left_margin, y=cur_y+5, w=epw/2)  # BVTV image

    pdf_file.set_xy(left_margin+epw/2, cur_y)
    pdf_file.cell(epw/2, 5, 'Damage distribution', border=0)
    pdf_file.image(png_dam, x=left_margin + epw/2, y=cur_y+5, w=epw/2)  # damage image
    del cur_y

    cur_y = pdf_file.y + 75
    pdf_file.set_xy(left_margin, cur_y)
    fu_characteristics.insert(0, ["Fult [N]", "dult [mm]", "k [N/mm]", "W [Nmm]"])
    fu_characteristics = map(list, zip(*fu_characteristics))

    i = 1
    for row in fu_characteristics:
        for col in row:
            pdf_file.cell(cw, ht, str(col), border=1)
        y_actual = cur_y + i * ht
        pdf_file.set_xy(left_margin, y_actual)
        i += 1
    del cur_y

    flags.insert(0, ["DENSFL", "VISCFL", "PYFL"])
    flags = map(list, zip(*flags))
    cur_y = y_actual + ht
    pdf_file.set_xy(left_margin, cur_y)

    for row in flags:
        for col in row:
            pdf_file.cell(cw, ht, str(col), border=1)
        cur_y += ht
        pdf_file.set_xy(left_margin, cur_y)
        i += 1

    pdf_file.set_y(pdf_file.y + 5)
    pdf_file.image(png_fu, x=left_margin, y=pdf_file.y, w=epw / 2)

    filename = job + ".pdf"
    pdf_file.output(filename)

