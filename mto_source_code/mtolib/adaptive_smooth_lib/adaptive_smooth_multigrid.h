#ifndef ADAPTIVE_SMOOTH_MULTIGRID_H
#define ADAPTIVE_SMOOTH_MULTIGRID_H

float** adaptive_smooth_multigrid(float** original_im, int kernel_size, float SNR_target, int M, int N, int iter_limit, int pad_size, int depth);
float** subtract_image(float** left_im, float** right_im, int width, int height);
float** reduce(float** im, int width, int height);
float** interpolate(float** im, int width, int height);

#endif
