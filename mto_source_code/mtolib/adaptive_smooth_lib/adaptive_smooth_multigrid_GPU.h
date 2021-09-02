#ifndef ADAPTIVE_SMOOTH_MULTIGRID_GPU_H
#define ADAPTIVE_SMOOTH_MULTIGRID_GPU_H

#include "OpenCL_functions.h"

float** adaptive_smooth_multigrid_GPU(float** original_im, int kernel_size, float SNR_target, int M, int N, int iter_limit, int pad_size, int depth);

#endif
