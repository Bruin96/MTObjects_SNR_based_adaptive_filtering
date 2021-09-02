"""Build a maxtree from a numpy array."""

import ctypes as ct
import numpy.ctypeslib as npct
import numpy as np

from mtolib import _ctype_classes as mt_class


class MaxTree:
    """A container class for the C maxtree"""
    def __init__(self, image, verbosity):
        self.image = image
        self.verbosity = verbosity

        self.nodes = None
        self.node_attributes = None
        self.root = None

        self.mt = None

    def flood(self):
        raise NotImplementedError

    def free_objects(self):
        raise NotImplementedError

    def ctypes_maxtree(self, params):
        # Create image object
        img_pointer = self.image.ravel().ctypes.data_as(ct.POINTER(params.d_type))

        return mt_class.MtData(root=self.root,
                               nodes=self.nodes,
                               node_attributes=self.node_attributes,
                               img=mt_class.Image(img_pointer, *self.image.shape, self.image.size),
                               verbosity_level = self.verbosity)


class OriginalMaxTree(MaxTree):
    def __init__(self, image, verbosity, params, real_image=None):
        # Sets up CTypes to interact with C code; Sets up maxtree

        MaxTree.__init__(self, image, verbosity)

        # Get access to the compiled C maxtree library
        if params.d_type == ct.c_double:
            self.mt_lib = ct.CDLL('mtolib/lib/maxtree_double.so')
        else:
            self.mt_lib = ct.CDLL('mtolib/lib/maxtree.so')


        # Create image object
        img_pointer = image.ravel().ctypes.data_as(ct.POINTER(params.d_type))

        c_img = mt_class.Image(img_pointer, image.shape[0], image.shape[1], image.size)

        if real_image is None:
            r_img = c_img

        else:
            real_img_pointer = real_image.ravel().ctypes.data_as(ct.POINTER(params.d_type))

            r_img = mt_class.Image(real_img_pointer, image.shape[0], image.shape[1], image.size)

        
        # Create empty mt object
        self.mt = mt_class.MtData()
        
        # Set argument types for init function; Initialise max tree.
        self.mt_lib.mt_init.argtypes = (ct.POINTER(mt_class.MtData), ct.POINTER(mt_class.Image), ct.POINTER(mt_class.Image))
        
        self.mt_lib.mt_init(ct.byref(self.mt), ct.byref(c_img), ct.byref(r_img))
        
        # Set verbosity
        self.mt_lib.mt_set_verbosity_level.argtypes = (ct.POINTER(mt_class.MtData),
                                                  ct.c_int)
        self.mt_lib.mt_set_verbosity_level(ct.byref(self.mt), verbosity)
        
        #self.mt_lib.mt_label_good_nodes.argtypes = (ct.POINTER(mt_class.MtData), ct.c_int)
        #self.mt_lib.mt_print_node_stats.argtypes = [ct.POINTER(mt_class.MtData), ct.c_float, ct.c_float]


    def flood(self):
        # Call the C function to flood the maxtree

        # C flood function takes a pointer to an MtData (mt_data in C) object
        self.mt_lib.mt_flood.argtypes = [ct.POINTER(mt_class.MtData)]
        self.mt_lib.mt_flood(ct.byref(self.mt))

        # Get a 2D numpy array of the MtNode data
        # nodes = npct.as_array(ct.cast(self.mt.nodes, ct.POINTER(ct.c_int32)), (self.mt.img.size, 2))

        # node_attributes = npct.as_array(ct.cast(self.mt.node_attributes, ct.POINTER(ct.c_double)),
        #                                (self.mt.img.size, 2))

        self.root = self.mt.root
        self.nodes = self.mt.nodes
        self.node_attributes = self.mt.node_attributes

    def free_objects(self):
        # Free the memory used by the max tree
        self.mt_lib.mt_free.argtypes = [ct.POINTER(mt_class.MtData)]
        self.mt_lib.mt_free(ct.byref(self.mt))

    def ctypes_maxtree(self):
        return self.mt
        
    # def label_good_nodes(self,n):
        # self.mt_lib.mt_label_good_nodes(ct.byref(self.mt), n)
        
    # def print_node_stats(self, gain, bg_var):
        # self.mt_lib.mt_print_node_stats(ct.byref(self.mt), gain, bg_var)
