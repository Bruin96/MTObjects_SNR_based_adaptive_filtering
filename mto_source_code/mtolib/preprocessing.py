"""Image pre-processing functions."""

import matplotlib.cm as cm
import matplotlib.pyplot as plt

import numpy as np
from scipy import stats
import scipy.ndimage.filters as filters
from mtolib import background, utils
from mtolib.adaptive_smooth_lib.smooth_adaptive import adaptive_smooth
from mtolib.io_mto import generate_image

from math import ceil, log

def preprocess_image(img, p, gaussian_blur=False, adaptive_blur=True, n=2, nan_value=np.inf, separate_smooth=False, SNR_target=3):
    """Estimate an image's background, subtract it, smooth and truncate."""
    
    # Estimate and subtract the background
    estimate_background(img, p)
    new_img = subtract_background(img, p.bg_mean)    
    
    # Smooth the image
    if adaptive_blur:
        if adaptive_blur and gaussian_blur:
            print("Both adaptive_blur and gaussian_blur set, defaulting to adaptive blur.")
            
        new_img = new_img * p.gain # Scale to photon count
            
        smoothed_img = adaptive_smooth(new_img, 5, SNR_target, 10, 3, 256)
        smoothed_img = replace_nans(truncate(smoothed_img), nan_value)
        
        # Scale down from photon count again
        new_img = new_img / p.gain
        smoothed_img = smoothed_img / p.gain
        
        if separate_smooth:
            return smoothed_img, new_img

        else:
            return smoothed_img, smoothed_img
        
    elif gaussian_blur:        
        smoothed_img = smooth_image(new_img, n)
        smoothed_img = replace_nans(truncate(smoothed_img), nan_value)
                        
        if separate_smooth:
            return smoothed_img, new_img

        else:
            return smoothed_img, smoothed_img
    else:
        new_img = replace_nans(truncate(new_img), nan_value)
        return new_img, new_img


def estimate_background(img, p):
    """Estimate background mean & variance"""

    if p.bg_mean is None or p.bg_variance < 0:

        if np.isnan(img).any():
            if p.verbosity > 0:
                print("WARNING: image contains NAN values which may affect output parameters")

        bg_mean_tmp, bg_variance_tmp = utils.time_function(background.estimate_bg,
                                                           (img, p.verbosity), p.verbosity,
                                                           "estimate background")

        if p.bg_mean is None:
            p.bg_mean = bg_mean_tmp

        if p.bg_variance < 0:
            p.bg_variance = bg_variance_tmp

    estimate_gain(img, p)

    if p.verbosity:
        print("\n---Background Estimates---")
        print("Background mean: ", p.bg_mean)
        print("Background variance: ", p.bg_variance)
        print("Gain: ", p.gain, " electrons/ADU")


def estimate_gain(img, p):
    """Estimate gain."""

    # Negative gains break sig test 4 - estimated gain should be positive
    if p.gain < 0:
        image_minimum = np.nanmin(img)
        if image_minimum < 0:
            p.soft_bias = image_minimum

        p.gain = (p.bg_mean - p.soft_bias) / p.bg_variance


def subtract_background(img, value):
    """Subtract the background from an image."""
    return img - value


def truncate(img):
    """Set all negative values in an array to zero."""
    return img.clip(min=0)


def smooth_image(img, n=2):
    """Apply a gaussian smoothing function to an image."""
    return filters.gaussian_filter(img, utils.fwhm_to_sigma(n))


def replace_nans(img, value=np.inf):
    if value == 0:
        return np.nan_to_num(img)
    else:
        img[np.isnan(img)] = value
        return img
