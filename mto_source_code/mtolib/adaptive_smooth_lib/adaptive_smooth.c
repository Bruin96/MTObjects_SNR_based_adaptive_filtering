

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>

#include "gauss_filter.h"
#include "array_utils.h"

#define max(a, b) (a > b) ? a : b
#define min(a, b) (a < b) ? a : b



float** adaptive_smooth(float** original_im, int kernel_size, float SNR_target, int M, int N, int iter_limit, int pad_size) {
    
    float** padded_im = reflection_padding(original_im, N, M, pad_size);
    
    struct timeval begin, end;
    gettimeofday(&begin, 0);
    
	float** out_im = init_array(N, M);
	float** copy_im = init_array(N+2*pad_size, M+2*pad_size);
	float** im = init_array(N+2*pad_size, M+2*pad_size);
    	
	for (int m = 0; m < M+2*pad_size; ++m) {
		for (int n = 0; n < N+2*pad_size; ++n) {
			im[m][n] = padded_im[m][n];
			copy_im[m][n] = padded_im[m][n];
		}
	}
    	
	int kernel_radius = kernel_size / 2;
	float tolerance = 0.01*SNR_target;
	//int iter = 1;

	#pragma omp parallel
    {
        int iter = 0;
        while (iter <= iter_limit) 
        {
            #pragma omp single
            printf("Current iteration = %d\n", iter);
        
            #pragma omp for
            for (int i = 0; i < M; ++i) {
                if (i%100 == 0) {
                    printf("i == %d\n", i);
                }
                for (int j = 0; j < N; ++j) {
                    int x = i+pad_size;
                    int y = j+pad_size;
				
                    float curr_mean = 0.0;
                    for (int m = -kernel_radius; m <= kernel_radius; ++m) {
                        for (int n = -kernel_radius; n <= kernel_radius; ++n) {
                            curr_mean += im[x+m][y+n];
                        }
                    }
                    curr_mean /= (float) (kernel_size*kernel_size);

                    float curr_sigma = sqrt(curr_mean); // Poissonian noise
                
                    if (curr_mean < 0.0) {
                        curr_sigma = sqrt(-curr_mean);
                    }
                
                    if (curr_mean == 0.0) {
                        curr_sigma = 1e-16;
                    }
				
                    float SNR_curr = curr_mean / curr_sigma;
                				
                    if (fabs(SNR_curr - SNR_target) <= tolerance || SNR_curr > SNR_target + tolerance) {
                        // SNR is above target, so no smoothing
                        out_im[i][j] = im[x][y];
                    }
                    else { // Apply adaptive Gaussian smoothing
                        float sigma_gauss = 0.45*pow(fabs(SNR_target - SNR_curr), 0.2) + 0.01;

                        int gauss_size = ((int) ceil(6.0*fabs(sigma_gauss)))+1;
                        if (gauss_size%2 == 0) {
                            gauss_size = gauss_size + 1;
                        }
                        gauss_size = max(9, gauss_size);  
                        if (gauss_size > 17) {
                            gauss_size = 17;
                        }        
                    
                        copy_im[x][y] = filter_gauss(im, gauss_size, x, y, sigma_gauss);
                        out_im[i][j] = copy_im[x][y];
                    }
                }
            }
				        
            #pragma omp for
            for (int p = 0; p < M+2*pad_size; ++p) {
                for (int q = 0; q < N+2*pad_size; ++q) {
                    im[p][q] = copy_im[p][q];
                }
            }    
                 
            iter += 1;
        }
    }
	
	free_array(im, M + 2*pad_size);
	free_array(copy_im, M + 2*pad_size);
    free_array(padded_im, M+2*pad_size);
    
    gettimeofday(&end, 0);
    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    double result = seconds + microseconds*1e-6;
    
    printf("Execution took %lf seconds\n", result);
	
	return out_im;
}
