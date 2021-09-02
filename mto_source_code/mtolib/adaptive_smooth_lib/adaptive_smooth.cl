
float filter_gauss(__global float* im, int im_width, int kernel_size, int x, int y, float sigma) {
    int kernel_radius = kernel_size/2;
    
    float filter_sum = 0.0; 
    float two_sigma_squared = 2*sigma*sigma;
    float const_term = 1/(M_PI*two_sigma_squared);
    
    // Compute normalisation term
    for (int i = -kernel_radius; i <= kernel_radius; ++i) {
        for (int j  = -kernel_radius; j <= kernel_radius; ++j) {
			filter_sum += const_term * exp(-((float) (i*i + j*j)) / two_sigma_squared);
        }
    }
    
    // Compute output value
    float filtered_val = 0.0;
    for (int i = -kernel_radius; i <= kernel_radius; ++i) {
        for (int j = -kernel_radius; j <= kernel_radius; ++j) {
            filtered_val += im[(x + i)*im_width + y + j] * const_term * 
                exp(-((float) (i*i + j*j)) / two_sigma_squared) / filter_sum;
        }
    }
    
    return filtered_val;
}


__kernel void adaptive_smooth(
    __global float* im,
    __global float* copy_im,
    int kernel_size,
    float SNR_target,
    int M,
    int N,
    int pad_size)
{
    int kernel_radius = kernel_size/2;
    float tolerance = 0.01*SNR_target;
    int idx = get_global_id(0);
    int x = idx/N + pad_size;
    int y = idx%N + pad_size;
    int im_width = N+2*pad_size;
    
    if (x < M + pad_size && y < N + pad_size) {
        // Compute mean
        float curr_mean = 0.0;
        //float sigma_curr = 0.0;
        for (int m = -kernel_radius; m <= kernel_radius; ++m) {
            for (int n = -kernel_radius; n <= kernel_radius; ++n) {
                curr_mean += im[(x+m)*im_width + (y+n)];
            }
        }
        curr_mean /= (float) (kernel_size*kernel_size);
        
        //for (int m = -kernel_radius; m <= kernel_radius; ++m) {
        //    for (int n = -kernel_radius; n <= kernel_radius; ++n) {
        //        sigma_curr += (im[(x+m)*im_width + (y+n)] - curr_mean) * (im[(x+m)*im_width + (y+n)] - curr_mean);
        //    }
        //}
        //sigma_curr = sqrt(sigma_curr) / (kernel_size*kernel_size);
        // Compute sigma, SNR_curr
        float sigma_curr = sqrt(fabs(curr_mean));
        float SNR_curr = curr_mean / sigma_curr;
    
        // Compute new value if SNR_curr is not above or close to SNR_target
        if (SNR_target - SNR_curr > tolerance) {
            //float sigma_gauss = SNR_target - SNR_curr + 0.001;
            float sigma_gauss = 0.1*pow((double)fabs(SNR_target - SNR_curr), 0.5) + 0.01;
            //printf("SNR_diff = %f\n", SNR_target - SNR_curr);
            //printf("sigma_gauss = %f\n", sigma_gauss);
            
            int gauss_size = ((int) ceil(6.0*fabs(sigma_gauss)))+1;
            gauss_size = gauss_size + (1 - gauss_size%2);
            gauss_size = max(9, gauss_size);  
            gauss_size = min(gauss_size, pad_size);
        
            copy_im[x*im_width + y] = filter_gauss(im, im_width, gauss_size, x, y, sigma_gauss);
        }
        else {
            copy_im[x*im_width + y] = im[x*im_width + y];
        }
    }
    
}
