

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "adaptive_smooth.h"
#include "array_utils.h"
#include "gauss_filter.h"
#include "adaptive_smooth_multigrid.h"

#define max(a, b) (a > b) ? a : b
#define min(a, b) (a < b) ? a : b


/* The multigrid algorithms works as follows: we have P levels, and at each level, we perform iter_limit
 * iterations of the adaptive smoothing operation. Then, we reduce the image, and do the same adaptive
 * smoothing on the coarser level. We go on until we hit done 'depth' levels, and then we move back up by 
 * computing the difference between the original, adaptively-smoothed image at that level and the inter-
 * polation of the result from the level one level lower. This produces a comparable image in a lot fewer 
 * iterations than traditional adaptive smoothing, thanks to faster convergence at smaller scales. This in
 * turn saves execution time.
 */ 
float** adaptive_smooth_multigrid(float** original_im, int kernel_size, float SNR_target, int M, int N, int iter_limit, int pad_size, int depth) 
{
    printf("depth = %d, pad_size = %d, M = %d, N = %d\n", depth, pad_size, M, N);
    
    if (depth < 1) {
        printf("Input depth was incorrect. Aborting.\n");
        exit(1);
    }
    
    float** curr_im = adaptive_smooth(original_im, kernel_size, SNR_target, M, N, iter_limit, pad_size);
    
    //return curr_im;
    
    printf("depth = %d - Finished adaptive_smooth pre-processing step.\n", depth);
    
    // Base case: at the bottom of the recursion, simply return the adaptive smoothing result
    if (depth == 1) { 
        return curr_im;
    }
    
    // Recursive case
    float** reduced_im = reduce(curr_im, N, M);
    
    float** recursive_im = adaptive_smooth_multigrid(reduced_im, kernel_size, SNR_target, M/2, N/2, iter_limit, max(1, pad_size), depth-1);
    printf("depth = %d - Finished recursive step\n", depth);
    
    float** diff_im = subtract_image(reduced_im, recursive_im, N/2, M/2);
    printf("depth = %d - Finished subtraction step\n", depth);
    
    float** interpolated_im = interpolate(diff_im, N/2, M/2);
    printf("depth = %d - Finished interpolation step\n", depth);
    
    float** curr_level_diff_im = subtract_image(curr_im, interpolated_im, N, M);
    
    float** result_im = adaptive_smooth(curr_level_diff_im, kernel_size, SNR_target, M, N, iter_limit, pad_size);
    
    free_array(curr_im, M);
    free_array(reduced_im, M/2);
    free_array(recursive_im, M/2);
    free_array(diff_im, M/2);
    free_array(interpolated_im, M);
    free_array(curr_level_diff_im, M);
    
    return result_im;
}

float** subtract_image(float** left_im, float** right_im, int width, int height) {
    float** diff_im = init_array(width, height);
        
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            diff_im[i][j] = left_im[i][j] - right_im[i][j];
        }
    }
    
    return diff_im;
}

float** reduce(float** im, int width, int height) {
    float** reduced_im = init_array(width/2, height/2);
    
    for (int i = 0; i < height/2; ++i) {
        for (int j = 0; j < width/2; ++j) {
            reduced_im[i][j] = im[2*i][2*j];
            //reduced_im[i][j] = 0.25*(im[2*i][2*j] + im[2*i+1][2*j] + im[2*i][2*j+1] + im[2*i+1][2*j+1]);
        }
    }
    
    return reduced_im;
}

float** interpolate(float** im, int width, int height) {
    float** interpolated_im = init_array(width*2, height*2);
    
    printf("Interpolation - (M, N) = (%d, %d)\n", height, width);
    
    for (int i = 0; i < height*2; ++i) {
        for (int j = 0; j < width*2; ++j) {
            interpolated_im[i][j] = im[i/2][j/2];
            
            /*
            if (j == 2*width-1 || i == 2*height-1) {
                interpolated_im[i][j] = im[i/2][j/2];
            }
            else if (j%2 == 1 && i%2 == 0) {
                interpolated_im[i][j] = 0.5*(im[i/2][j/2] + im[i/2][j/2+1]);
            }
            else if (j%2 == 0 && i%2 == 1) {
                interpolated_im[i][j] = 0.5*(im[i/2][j/2] + im[i/2+1][j/2]);
            }
            else if (j%2 == 1 && i%2 == 1) {
                interpolated_im[i][j] = 0.25*(im[i/2][j/2] + im[i/2+1][j/2] + im[i/2][j/2+1] + im[i/2+1][j/2+1]);
            }
            else {
                interpolated_im[i][j] = im[i/2][j/2];
            }
            */
            
        }
    }
    
    return interpolated_im;
}
