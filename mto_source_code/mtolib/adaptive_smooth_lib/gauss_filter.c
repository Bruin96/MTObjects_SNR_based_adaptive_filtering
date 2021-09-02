#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "gauss_filter.h"
#include "array_utils.h"

float filter_gauss(float** im, int kernel_size, int x, int y, float sigma) {
	float filtered_val = 0.0;
    int kernel_radius = kernel_size/2;
        
    float** filter = init_array(kernel_size, kernel_size);
        
    // Compute Gaussian kernel  
    float filter_sum = 0.0; 
    for (int i = -kernel_radius; i <= kernel_radius; ++i) {
        for (int j  = -kernel_radius; j <= kernel_radius; ++j) {
			float curr_val = (1/(2*M_PI*sigma*sigma)) * exp(-((float) (i*i + j*j)) / (2*sigma*sigma));
            filter[i+kernel_radius][j+kernel_radius] = curr_val;
            filter_sum += curr_val;
        }
    }
    
    // Normalise kernel
    for (int i = -kernel_radius; i <= kernel_radius; ++i) {
        for (int j  = -kernel_radius; j <= kernel_radius; ++j) {
            filter[i+kernel_radius][j+kernel_radius] = filter[i+kernel_radius][j+kernel_radius] / filter_sum;
        }
    }
    
    // Compute result of convolution
    for (int i = -kernel_radius; i <= kernel_radius; ++i) {
        for (int j = -kernel_radius; j <= kernel_radius; ++j) {
            filtered_val = filtered_val + im[x + i][y + j] * filter[i+kernel_radius][j+kernel_radius];
        } 
    }
    
    free_array(filter, kernel_size);
    
    return filtered_val;
}
