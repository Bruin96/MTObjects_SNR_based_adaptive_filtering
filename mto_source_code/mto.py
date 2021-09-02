import gc
import numpy as np
import math
import os
import time
import mtolib.main as mto
from skimage.color import gray2rgb
from mtolib.io_mto import write_fits_file, get_fits_header
from astropy.io import fits
import warnings
from astropy.utils.exceptions import AstropyDeprecationWarning
from mtolib.preprocessing import replace_nans, truncate


"""Example program - using original settings"""

# Get the input image and parameters
image, params = mto.setup()

use_adaptsmooth_lib = False

# Pre-process the image
processed_image, unsmoothed_image = mto.preprocess_image(image, params, n=2, gaussian_blur=False, adaptive_blur=True, separate_smooth=True, SNR_target=params.snr)

# Build a max tree
mt = mto.build_max_tree(processed_image, params, real_img=unsmoothed_image)

# Filter the tree and find objects
id_map, sig_ancs = mto.filter_tree(mt, processed_image.shape, params, sig_test=mto.sig_test_1)

# Relabel objects for clearer visualisation
id_map = mto.relabel_segments(id_map, shuffle_labels=True)

# Generate output files
mto.generate_image(image, id_map, params)

#mto.generate_parameters(image, id_map, sig_ancs, params)
