#ifndef ADAPTIVE_SMOOTH_H
#define ADAPTIVE_SMOOTH_H

float** adaptive_smooth(float** original_im, int kernel_size, float SNR_target, int M, int N, int iter_limit, int pad_size);

#endif
