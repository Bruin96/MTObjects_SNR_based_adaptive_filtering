

__kernel void sub(
    __global float* im_left,
    __global float* im_right,
    int M,
    int N,
    int pad_size)
{
    int idx = get_global_id(0);
    int x = idx/N + pad_size;
    int y = idx%N + pad_size;
    im_left[x*(N+2*pad_size) + y] = im_left[x*(N+2*pad_size) + y] - im_right[x*(N+2*pad_size) + y];
}    
