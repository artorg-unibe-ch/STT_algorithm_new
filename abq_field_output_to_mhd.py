"""
Read field output variable from ODB file.
Run with abaqus python!
20220617

Call as follows:
abaqus python abq_field_output_to_mhd.py [/home/ygugler/Desktop/FEAMUR2020_Images/Test/3_FEA/PA08A02A_2/test.odb,4,SDV2,3.0,/home/ygugler/Desktop/test.npy,-1.0]
"""

import sys
import re

import numpy as np

from odbAccess import *
from abaqusConstants import *

# command line arguments
cli_args = sys.argv[1][1:-1]  # remove file name (argv[0])
args = re.split('\,', cli_args)  # CLI arguments are put in as list
print(args)

odb_file = args[0]  # name of ODB
increment_number = int(args[1])  # increment to look at
field_var = args[2]  # field variable to read out
resolution = float(args[3])  # resolution of mesh
out_file = args[4]  #
background_value = float(args[5])

odb_file_upgraded = odb_file[0:-4] + '_new.odb'


if isUpgradeRequiredForOdb(upgradeRequiredOdbPath=odb_file):
    upgradeOdb(existingOdbPath=odb_file, upgradedOdbPath=odb_file_upgraded)
    odb = openOdb(odb_file_upgraded, readOnly=True)
    print("\n...converting odb file to 2021 version")

else:
    odb = openOdb(odb_file, readOnly=True)

step1 = odb.steps["Step-1"]
frames = step1.frames

nodes = odb.rootAssembly.instances['PART-1-1'].nodes
elements = odb.rootAssembly.instances['PART-1-1'].elements

nbr_elements = len(elements)

elements_struct_array = np.zeros(1, dtype={'names': ('El_label', 'Nodes', 'Centroid', 'Field_Output'),
                                                  'formats': ('i4', '8i4', '3f8', 'f8')})
vals = np.ones(1, dtype={'names': ('El_label', 'Nodes', 'Centroid', 'Field_Output'),
                                    'formats': ('i4', '8i4', '3f8', 'f8')})

elres_fieldVar = frames[increment_number].fieldOutputs[field_var]

nodes_struct_array = np.zeros(1, dtype={'names': ('Node_label', 'Coordinates'),
                                                  'formats': ('i4', '3f8')})

nodes_vals = np.zeros(1, dtype={'names': ('Node_label', 'Coordinates'),
                                                  'formats': ('i4', '3f8')})

nbr_Nodes = len(nodes)
for k in range(0, nbr_Nodes):
    nodes_struct_array[k]["Node_label"] = nodes[k].label
    nodes_struct_array[k]["Coordinates"] = nodes[k].coordinates
    nodes_struct_array = np.append(nodes_struct_array, nodes_vals)

nodes_struct_array = nodes_struct_array[0:-1]

i = 0
for value in elres_fieldVar.values:
    elements_struct_array[i]["El_label"] = value.elementLabel  # write element label
    elements_struct_array[i]["Field_Output"] = value.data  # write value of searched variable
    i += 1
    elements_struct_array = np.append(elements_struct_array, vals)

elements_struct_array = elements_struct_array[0:-1]

for elem in elements:
    label = int(elem.label)  # return label of element
    argument = np.argwhere(np.where(elements_struct_array["El_label"] == label, 1, 0))

    if len(argument) != 0:
        argument = argument[0][0]
        elements_struct_array[argument]["Nodes"] = elem.connectivity

        coords = np.array([0.0, 0.0, 0.0])

        for nodel in elements_struct_array[argument]["Nodes"]:
            arg = np.argwhere(np.where(nodes_struct_array["Node_label"] == nodel, 1, 0))
            coords += nodes_struct_array[arg]["Coordinates"][0][0]

        centroid = coords / 8
        elements_struct_array[argument]["Centroid"] = centroid

centroids = elements_struct_array["Centroid"]

x_min = np.min(centroids[0:, 0])
x_max = np.max(centroids[0:, 0])

y_min = np.min(centroids[0:, 1])
y_max = np.max(centroids[0:, 1])

z_min = np.min(centroids[0:, 2])
z_max = np.max(centroids[0:, 2])

# create empty numpy array
# --> compute size that is needed
# --> write empty array
# loop over structured array and write values to correct location within new array
# write mhd

xdim = int((x_max - x_min) / resolution)
ydim = int((y_max - y_min) / resolution)
zdim = int((z_max - z_min) / resolution)

array_to_write = background_value * np.ones([xdim + 1, ydim + 1, zdim + 1], dtype=np.float32)

for element in elements_struct_array:
    index = ((element["Centroid"] - np.array([x_min, y_min, z_min]))/resolution).astype(int)
    array_to_write[index[0], index[1], index[2]] = element["Field_Output"]

np.save(out_file, array_to_write)
