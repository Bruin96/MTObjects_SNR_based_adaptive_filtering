#include "array_utils.h"

#include <stdio.h>
#include <stdlib.h>

float** init_array(int width, int height) {
	float** im = (float**) malloc(height * sizeof(float*));
	
	for (int i = 0; i < height; ++i) {
		im[i] = (float*) malloc(width * sizeof(float));
	}
	
	return im;
}

void free_array(float**im, int height) {
	if (im == NULL) {
        return;
    }
    
    for (int i = 0; i < height; ++i) {
		free(im[i]);
	}
	
	free(im);
}

void swap(float** a, float** b) {
	float** temp = a;
	a = b;
	b = temp;
}


float** convert1Ddoubleto2Dfloat(double* im, int height, int width) {
    float** im_2D = (float**) malloc(height*sizeof(float*));
    for (int i = 0; i < height; ++i) {
        im_2D[i] = (float*) malloc(width*sizeof(float));
        for (int j = 0; j < width; ++j) {
            im_2D[i][j] = im[i*width + j];
        }
    }
    
    return im_2D;
}


double* convert2Dfloatto1Ddouble(float** im, int height, int width) {
    double* im_1D = (double*) malloc(height*width*sizeof(double));
    
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            im_1D[i*width+j] = im[i][j];
        }
    }
    
    return im_1D;
}


float** reflection_padding(float** im, int width, int height, int pad_size) {
    float** padded_im = (float**) malloc((height + 2*pad_size) * sizeof(float*));
    
    for (int i = 0; i < height + 2*pad_size; ++i) {
        padded_im[i] = (float*) malloc((width + 2*pad_size) * sizeof(float));
    }
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < height + 2*pad_size; ++i) {
        for (int j = 0; j < width + 2*pad_size; ++j) {
            
            if (i >= pad_size && i < height+pad_size && j >= pad_size && j < width + pad_size) {
                padded_im[i][j] = im[i-pad_size][j-pad_size]; // Center region
            }
            else if (i < pad_size && j < pad_size) {
                padded_im[i][j] = im[pad_size-i][pad_size-j]; // Top-left padding
            }
            else if (i >= pad_size && i < height + pad_size && j < pad_size) {
                padded_im[i][j] = im[i - pad_size][pad_size-j]; // Center-left padding
            }
            else if (i >= height + pad_size && j < pad_size) {
                padded_im[i][j] = im[2*height + pad_size - i - 1][pad_size-j]; // Bottom-left padding
            }
            else if ( i >= height + pad_size && j >= pad_size && j < width + pad_size) {
                padded_im[i][j] = im[2*height + pad_size - i - 1][j - pad_size]; // Bottom-center padding
            }
            else if (i >= height + pad_size && j >= width + pad_size) {
                padded_im[i][j] = im[2*height + pad_size - i - 1][2*width + pad_size - j - 1]; // Bottom-right padding
            }
            else if (i >= pad_size && i < height+pad_size && j >= width + pad_size) {
                padded_im[i][j] = im[i-pad_size][2*width + pad_size - j - 1]; // Center-right padding
            }
            else if (i < pad_size && j >= width + pad_size) {
                padded_im[i][j] = im[pad_size-i][2*width + pad_size - j - 1]; // Top-right padding
            }
            else {
                padded_im[i][j] = im[pad_size-i][j-pad_size]; // Top-center padding
            }
        }
    }
    
    return padded_im;
}
