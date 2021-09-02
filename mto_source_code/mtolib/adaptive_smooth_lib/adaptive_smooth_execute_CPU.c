#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "adaptive_smooth_multigrid.h"
#include "adaptive_smooth.h"
#include "array_utils.h"


void adaptive_smooth_execute(double* original_im, double *out_im, int kernel_size, float SNR_target, int M, int N, int iter_limit, int pad_size, int depth) {
    float** im_2D = convert1Ddoubleto2Dfloat(original_im, M, N);
    
    float** result_im;
    
    result_im = adaptive_smooth_multigrid(im_2D, kernel_size, SNR_target, M, N, iter_limit, pad_size, depth);
        
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            out_im[i*N+j] = result_im[i][j];
        } 
    }
    
    free_array(im_2D, M);
    free_array(result_im, M);
    
}
