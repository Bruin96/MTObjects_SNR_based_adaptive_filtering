

__kernel void reduce(
    __global float* im_level_1,
    __global float* im_level_2,
    __global float* im_level_2_copy,
    int M_level_2,
    int N_level_2,
    int pad_size)
{
    int idx = get_global_id(0);
    int x = idx/(N_level_2 + 2*pad_size);
    int y = idx%(N_level_2 + 2*pad_size);
    int x_center = x - pad_size;
    int y_center = y - pad_size;
    
    if (x < M_level_2 + 2*pad_size && y < M_level_2 + 2*pad_size) {
    if (x < pad_size && y >= pad_size && y < N_level_2+pad_size) { // Top center padding
        im_level_2[x*(N_level_2+2*pad_size) + y] = im_level_1[x*(2*N_level_2+2*pad_size) + pad_size + 2*y_center];
        im_level_2_copy[x*(N_level_2+2*pad_size) + y] = im_level_1[x*(2*N_level_2+2*pad_size) + pad_size + 2*y_center];
    }
    else if (x >= M_level_2+pad_size && y >= pad_size && y < N_level_2+pad_size) { // Bottom center padding
        im_level_2[x*(N_level_2+2*pad_size) + y] = im_level_1[(x + M_level_2)*(2*N_level_2+2*pad_size) + pad_size + 2*y_center];
        im_level_2_copy[x*(N_level_2+2*pad_size) + y] = im_level_1[(x + M_level_2)*(2*N_level_2+2*pad_size) + pad_size + 2*y_center];
    }
    else if (x >= pad_size && x < M_level_2+pad_size && y < pad_size) { // Left middle padding
        im_level_2[x*(N_level_2+2*pad_size) + y] = im_level_1[(2*x_center+pad_size)*(2*N_level_2+2*pad_size) + y];
        im_level_2_copy[x*(N_level_2+2*pad_size) + y] = im_level_1[(2*x_center+pad_size)*(2*N_level_2+2*pad_size) + y];
    }
    else if (x >= pad_size && x < M_level_2+pad_size && y >= N_level_2+pad_size) { // Right middle padding
        im_level_2[x*(N_level_2+2*pad_size) + y] = im_level_1[(2*x_center+pad_size)*(2*N_level_2+2*pad_size) + pad_size + 2*y_center];
        im_level_2_copy[x*(N_level_2+2*pad_size) + y] = im_level_1[(2*x_center+pad_size)*(2*N_level_2+2*pad_size) + pad_size + 2*y_center];
    }
    else if (x < pad_size && y < pad_size) { // Top left padding
        im_level_2[x*(N_level_2+2*pad_size) + y] = im_level_1[x*(2*N_level_2+2*pad_size) + y];
        im_level_2_copy[x*(N_level_2+2*pad_size) + y] = im_level_1[x*(2*N_level_2+2*pad_size) + y];
    }
    else if (x >= M_level_2+pad_size && y < pad_size) { // Bottom left padding
        im_level_2[x*(N_level_2+2*pad_size) + y] = im_level_1[(x + M_level_2)*(2*N_level_2+2*pad_size) + y];
        im_level_2_copy[x*(N_level_2+2*pad_size) + y] = im_level_1[(x + M_level_2)*(2*N_level_2+2*pad_size) + y];
    }
    else if (x < pad_size && y >= N_level_2+pad_size) { // Top right padding
        im_level_2[x*(N_level_2+2*pad_size) + y] = im_level_1[x*(2*N_level_2+2*pad_size) + y + N_level_2];
        im_level_2_copy[x*(N_level_2+2*pad_size) + y] = im_level_1[x*(2*N_level_2+2*pad_size) + y + N_level_2];
    }
    else if (x >= M_level_2+pad_size && y >= N_level_2+pad_size) { // Bottom right padding
        im_level_2[x*(N_level_2+2*pad_size) + y] = im_level_1[(x + M_level_2)*(2*N_level_2+2*pad_size) + y + N_level_2];
        im_level_2_copy[x*(N_level_2+2*pad_size) + y] = im_level_1[(x + M_level_2)*(2*N_level_2+2*pad_size) + y + N_level_2];
    }
    else { // Interpolate image itself
        im_level_2[x*(N_level_2+2*pad_size) + y] = im_level_1[(2*x_center+pad_size)*(2*N_level_2+2*pad_size) + (2*y_center+pad_size)];
        im_level_2_copy[x*(N_level_2+2*pad_size) + y] = im_level_1[(2*x_center+pad_size)*(2*N_level_2+2*pad_size) + (2*y_center+pad_size)];
    }
    }
    
    
}
