"""
Classes imported from medtool functions, needed for Abaqus read and write functions
{

}
"""
# imports
from Pipeline_Modules.lib import *


class element:
    """
    Element object.
    """
    def __init__(self, id, nodes, type):
        self.id = id
        self.nodes = nodes
        self.type = type
        self.part = None
        self.mat = None
        self.elems = {}
        self.bbox = {}
        return

    def get_id(self):
        return self.id

    def set_id(self, _id):
        self.id = _id

    def get_type(self):
        return self.type

    def set_type(self, _type):
        self.type = _type

    def get_nodes(self):
        return self.nodes

    def set_nodes(self, nlist):
        self.nodes = nlist

    def append_node(self, node):
        self.nodes.append(node)

    def get_part(self):
        return self.part

    def set_part(self, _id):
        self.part = _id

    def get_mat(self):
        return self.mat

    def set_mat(self, _mat):
        self.mat = _mat

    def get_elems(self):
        return self.elems

    def set_elems(self, elDict):
        self.elems = elDict

    def show(self):
        print('\nELEMENT Info:')
        print('id     =', self.id)
        print('nodes  =', self.nodes)
        print('type   =', self.type)

    def get_center(self):
        x = 0.0
        y = 0.0
        z = 0.0
        for noid in self.nodes:
            x += self.nodes[noid].get_x()
            y += self.nodes[noid].get_y()
            z += self.nodes[noid].get_z()

        return (x, y, z)


class node:
    """
    Node object.
    """
    def __init__(self, id, x, y=None, z=None):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.elemList = []

    def get_coord(self):
        return (self.x, self.y, self.z)

    def get_coord_numpy(self):
        return np.array([self.x, self.y, self.z])

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_z(self):
        return self.z

    def get_id(self):
        return self.id

    def set_id(self, _id):
        self.id = _id

    def set_coord(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def set_coord_numpy(self, arr):
        self.x = arr[0]
        self.y = arr[1]
        self.z = arr[2]

    def set_x(self, x):
        self.x = x

    def set_y(self, y):
        self.y = y

    def set_z(self, z):
        self.z = z

    def append_to_elemList(self, elementID):
        self.elemList.append(elementID)

    def get_elemList(self):
        return self.elemList

    def get_dimension(self):
        if self.y == None:
            return 1
        else:
            if self.z == None:
                return 2
            return 3
            return

    def show(self):
        print('\nNODE Info:')
        print('id       =', self.id)
        print('x,y,z    =', self.x, self.y, self.z)
        print('elemList =', self.elemList)

