import numpy as np
from numpy.ctypeslib import ndpointer, as_ctypes
import ctypes
import pathlib
from math import ceil, log


libname = pathlib.Path().absolute() / "mtolib/adaptive_smooth_lib/adaptive_smooth.so"
c_lib = ctypes.CDLL(libname)

C_FLOAT = ctypes.c_float
C_PFLOAT = ctypes.POINTER(C_FLOAT)
C_PPFLOAT = ctypes.POINTER(C_PFLOAT)


def adaptive_smooth(im, kernel_size, SNR_target, iter_limit, depth, pad_size):	
    (M, N) = im.shape
    
    c_lib.adaptive_smooth_execute.argtypes = [ndpointer(ctypes.c_double, \
        flags="C_CONTIGUOUS"), ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), \
        ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int, \
        ctypes.c_int, ctypes.c_int]
    
    x_padding = (8 - M%8)%8
    y_padding = (8 - N%8)%8
    
    if x_padding > 0 or y_padding > 0: # Pad to a multiple of 8
        im = reflection_padding(im, N+y_padding, M+x_padding)
    
    raveled_im = im.ravel()
    im_pointer = im.ravel().ctypes.data_as(ctypes.POINTER(C_FLOAT))
        
    im_out = np.zeros(shape=(M+x_padding, N+y_padding), dtype=np.float64)
            
    
    c_lib.adaptive_smooth_execute(raveled_im, im_out, kernel_size, \
        ctypes.c_float(SNR_target), M+x_padding, N+y_padding, iter_limit, pad_size, depth)

    if x_padding > 0 or y_padding > 0: # Remove padding
        x_start = x_padding // 2
        y_start = y_padding // 2
        im_out = im_out[x_start:M+x_start, y_start:N+y_start]
                        	
    return im_out



def reflection_padding(im_grey, target_width, target_height):    
    left_height = (target_height - im_grey.shape[0]) // 2
    right_height = target_height - left_height - im_grey.shape[0] 

    left_width = (target_width - im_grey.shape[1]) // 2
    right_width = target_width - left_width - im_grey.shape[1]
    
    reflected_im = np.pad(im_grey, pad_width=((left_height, right_height), \
        (left_width, right_width)), mode='reflect')
    
    (M, N) = reflected_im.shape

    return reflected_im

