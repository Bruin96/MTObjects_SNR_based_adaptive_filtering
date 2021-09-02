

__kernel void int_sub(
    __global float* level_1_im,
    __global float* level_2_reduced,
    int M,
    int N,
    int pad_size)
{    
    int idx = get_global_id(0);
    int x = idx/N;
    int y = idx%N;
    
    if (x < M + pad_size && y < N + pad_size) {
        level_1_im[(x+pad_size)*(N+2*pad_size) + y + pad_size] = level_1_im[(x+pad_size)*(N+2*pad_size) + y + pad_size] - 
            level_2_reduced[(x/2 + pad_size)*(N/2+2*pad_size) + y/2 + pad_size];
    }
}
