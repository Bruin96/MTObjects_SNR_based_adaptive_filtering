

__kernel void overwrite(
    __global float* dest,
    __global float* src,
    int M,
    int N,
    int pad_size)
{
    int idx = get_global_id(0);
    int x = idx/N + pad_size;
    int y = idx%N + pad_size;
    
    dest[x*(N+2*pad_size) + y] = src[x*(N+2*pad_size) + y];
}
