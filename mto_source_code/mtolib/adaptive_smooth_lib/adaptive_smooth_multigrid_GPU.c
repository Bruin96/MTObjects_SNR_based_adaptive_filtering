#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <stdio.h>
#include <stdlib.h>

#include "adaptive_smooth_multigrid_GPU.h"
#include "array_utils.h"


float** adaptive_smooth_multigrid_GPU(float** original_im, int kernel_size, float SNR_target, 
                                        int M, int N, int iter_limit, int pad_size, int depth)
{
    clock_t start_GPU_total = clock();
    
    float** padded_im = reflection_padding(original_im, N, M, pad_size);
    
    int padded_width = N + 2*pad_size;
    int padded_height = M + 2*pad_size;
    int padded_size = padded_width * padded_height;
    
    float* padded_im_concat = (float*) malloc(padded_size*sizeof(float));
    for (int i = 0; i < padded_height; ++i) {
        for (int j = 0; j < padded_width; ++j) {
            padded_im_concat[i*padded_width + j] = padded_im[i][j];
        }
    }
    
    char filenames[5][200] = { "mtolib/adaptive_smooth_lib/adaptive_smooth.cl", 
        "mtolib/adaptive_smooth_lib/reduce.cl", "mtolib/adaptive_smooth_lib/sub.cl", 
        "mtolib/adaptive_smooth_lib/int_sub.cl", "mtolib/adaptive_smooth_lib/overwrite.cl" };
	char function_names[5][200] = { "adaptive_smooth", "reduce", "sub", "int_sub", "overwrite" };
    
    cl_uint num_platforms = 0;
	cl_uint num_devices = 0;

	cl_platform_id* platforms = getPlatforms(&num_platforms);
	cl_device_id* devices = getDevices(&num_devices, platforms, num_platforms);

	cl_context context = createContext(devices, num_devices);
	cl_command_queue cmd_queue = createCommandQueue(devices, context);
    
    // Allocate data to GPU
	cl_float* im_level_1 = convertToCLfloat(padded_im_concat, padded_size);
	cl_mem buffer_level_1 = createBuffer(context, im_level_1, (M+2*pad_size)*(N+2*pad_size), CL_MEM_READ_WRITE);
    cl_float* im_level_1_copy = convertToCLfloat(padded_im_concat, padded_size);
    cl_mem buffer_level_1_copy = createBuffer(context, im_level_1_copy, (M+2*pad_size)*(N+2*pad_size), CL_MEM_READ_WRITE);
    
	cl_float* im_level_2_copy = (cl_float*)malloc((M/2 + 2*pad_size)*(N/2 + 2*pad_size) * sizeof(cl_float));
	cl_mem buffer_level_2_copy = createBuffer(context, im_level_2_copy, (M/2 + 2*pad_size)*(N/2 + 2*pad_size), CL_MEM_READ_WRITE);
    cl_float* im_level_2 = (cl_float*)malloc((M/2 + 2*pad_size)*(N/2 + 2*pad_size) * sizeof(cl_float));
	cl_mem buffer_level_2 = createBuffer(context, im_level_2, (M/2 + 2*pad_size)*(N/2 + 2*pad_size), CL_MEM_READ_WRITE);
    
	cl_float* im_level_3 = (cl_float*)malloc((M/4+2*pad_size)*(N/4 + 2*pad_size) * sizeof(cl_float));
	cl_mem buffer_level_3 = createBuffer(context, im_level_3, (M/4+2*pad_size)*(N/4 + 2*pad_size), CL_MEM_READ_WRITE);
    cl_float* im_level_3_copy = (cl_float*)malloc((M/4+2*pad_size)*(N/4 + 2*pad_size) * sizeof(cl_float));
	cl_mem buffer_level_3_copy = createBuffer(context, im_level_3_copy, (M/4+2*pad_size)*(N/4 + 2*pad_size), CL_MEM_READ_WRITE);
    
    cl_float* im_level_1_reduced = (cl_float*)malloc((M/2 + 2*pad_size)*(N/2 + 2*pad_size) * sizeof(cl_float));
    cl_mem buffer_level_1_reduced = createBuffer(context, im_level_1_reduced, (M/2 + 2*pad_size)*(N/2 + 2*pad_size), CL_MEM_READ_WRITE);
    cl_float* im_level_2_reduced = (cl_float*)malloc((M/4+2*pad_size)*(N/4 + 2*pad_size) * sizeof(cl_float));
    cl_mem buffer_level_2_reduced = createBuffer(context, im_level_2_reduced, (M/4+2*pad_size)*(N/4 + 2*pad_size), CL_MEM_READ_WRITE);
    
    
    writeHostToDevice(cmd_queue, buffer_level_1_copy, im_level_1_copy, padded_size, CL_TRUE);
    writeHostToDevice(cmd_queue, buffer_level_1, im_level_1, padded_size, CL_TRUE);
    
    // Create and build program, create kernels
    int num_kernels = 5;
	cl_program cp_program[num_kernels];
	cl_kernel ck_kernel[num_kernels];
    
    for (int i = 0; i < num_kernels; i++) {
		cp_program[i] = createProgram(context, filenames[i]);
		buildProgram(devices, cp_program[i]);

		ck_kernel[i] = createKernel(cp_program[i], function_names[i]);
	}
    
    // Execute multigrid algorithm
    int local_size = 1024;
    size_t local_work_size[2] = { 1024, 1 }; // TODO: 1024 should be replaced with max local_work_size of device
    int global_size = M*N + (1024 - (M*N)%local_size);
	size_t global_work_size[2] = { global_size, 1 };
    
    // Adaptive smooth level 1
    cl_int ciErr = clSetKernelArg(ck_kernel[0], 0, sizeof(cl_mem), (void*)&buffer_level_1);
    ciErr = clSetKernelArg(ck_kernel[0], 1, sizeof(cl_mem), (void*)&buffer_level_1_copy);
    ciErr = clSetKernelArg(ck_kernel[0], 2, sizeof(cl_int), (void*)&kernel_size);
    ciErr = clSetKernelArg(ck_kernel[0], 3, sizeof(cl_float), (void*)&SNR_target);
    ciErr = clSetKernelArg(ck_kernel[0], 4, sizeof(cl_int), (void*)&M);
    ciErr = clSetKernelArg(ck_kernel[0], 5, sizeof(cl_int), (void*)&N);
    ciErr = clSetKernelArg(ck_kernel[0], 6, sizeof(cl_int), (void*)&pad_size);
    
    ciErr = clSetKernelArg(ck_kernel[4], 0, sizeof(cl_mem), (void*)&buffer_level_1);
    ciErr = clSetKernelArg(ck_kernel[4], 1, sizeof(cl_mem), (void*)&buffer_level_1_copy);
    ciErr = clSetKernelArg(ck_kernel[4], 2, sizeof(cl_int), (void*)&M);
    ciErr = clSetKernelArg(ck_kernel[4], 3, sizeof(cl_int), (void*)&N);
    ciErr = clSetKernelArg(ck_kernel[4], 4, sizeof(cl_int), (void*)&pad_size);
    
    for (int i = 0; i < iter_limit; i++) { // Execute iteration, wait for it to finish, then do next iteration
        enqueueBuffer(cmd_queue, ck_kernel[0], NULL, global_work_size, local_work_size);
        enqueueBuffer(cmd_queue, ck_kernel[4], NULL, global_work_size, local_work_size);
		clFinish(cmd_queue);
    }
    
    
    // Reduce level 1
    int N_level_2 = N/2;
    int M_level_2 = M/2;
    ciErr = clSetKernelArg(ck_kernel[1], 0, sizeof(cl_mem), (void*)&buffer_level_1);
    ciErr = clSetKernelArg(ck_kernel[1], 1, sizeof(cl_mem), (void*)&buffer_level_2);
    ciErr = clSetKernelArg(ck_kernel[1], 2, sizeof(cl_mem), (void*)&buffer_level_1_reduced);
    ciErr = clSetKernelArg(ck_kernel[1], 3, sizeof(cl_int), (void*)&M_level_2);
    ciErr = clSetKernelArg(ck_kernel[1], 4, sizeof(cl_int), (void*)&N_level_2);
    ciErr = clSetKernelArg(ck_kernel[1], 5, sizeof(cl_int), (void*)&pad_size);
    
    global_size = (M_level_2+2*pad_size)*(N_level_2+2*pad_size) + (1024 - ((M_level_2+2*pad_size)*(N_level_2+2*pad_size))%local_size);
	global_work_size[0] = global_size;
    
    enqueueBuffer(cmd_queue, ck_kernel[1], NULL, global_work_size, local_work_size);
    clFinish(cmd_queue);
    
    
    // Adaptive smooth level 2
    global_size = M_level_2*N_level_2 + (1024 - (M_level_2*N_level_2)%local_size);
    global_work_size[0] = global_size;
    
    ciErr = clSetKernelArg(ck_kernel[0], 0, sizeof(cl_mem), (void*)&buffer_level_2);
    ciErr = clSetKernelArg(ck_kernel[0], 1, sizeof(cl_mem), (void*)&buffer_level_2_copy);
    ciErr = clSetKernelArg(ck_kernel[0], 2, sizeof(cl_int), (void*)&kernel_size);
    ciErr = clSetKernelArg(ck_kernel[0], 3, sizeof(cl_float), (void*)&SNR_target);
    ciErr = clSetKernelArg(ck_kernel[0], 4, sizeof(cl_int), (void*)&M_level_2);
    ciErr = clSetKernelArg(ck_kernel[0], 5, sizeof(cl_int), (void*)&N_level_2);
    ciErr = clSetKernelArg(ck_kernel[0], 6, sizeof(cl_int), (void*)&pad_size);
    
    ciErr = clSetKernelArg(ck_kernel[4], 0, sizeof(cl_mem), (void*)&buffer_level_2);
    ciErr = clSetKernelArg(ck_kernel[4], 1, sizeof(cl_mem), (void*)&buffer_level_2_copy);
    ciErr = clSetKernelArg(ck_kernel[4], 2, sizeof(cl_int), (void*)&M_level_2);
    ciErr = clSetKernelArg(ck_kernel[4], 3, sizeof(cl_int), (void*)&N_level_2);
    ciErr = clSetKernelArg(ck_kernel[4], 4, sizeof(cl_int), (void*)&pad_size);
    
    for (int i = 0; i < iter_limit; i++) { // Execute iteration, wait for it to finish, then do next iteration
        enqueueBuffer(cmd_queue, ck_kernel[0], NULL, global_work_size, local_work_size);
        enqueueBuffer(cmd_queue, ck_kernel[4], NULL, global_work_size, local_work_size);
		clFinish(cmd_queue);
    }
    
    // Reduce level 2
    int N_level_3 = N/4;
    int M_level_3 = M/4;
    ciErr = clSetKernelArg(ck_kernel[1], 0, sizeof(cl_mem), (void*)&buffer_level_2);
    ciErr = clSetKernelArg(ck_kernel[1], 1, sizeof(cl_mem), (void*)&buffer_level_3);
    ciErr = clSetKernelArg(ck_kernel[1], 2, sizeof(cl_mem), (void*)&buffer_level_2_reduced);
    ciErr = clSetKernelArg(ck_kernel[1], 3, sizeof(cl_int), (void*)&M_level_3);
    ciErr = clSetKernelArg(ck_kernel[1], 4, sizeof(cl_int), (void*)&N_level_3);
    ciErr = clSetKernelArg(ck_kernel[1], 5, sizeof(cl_int), (void*)&pad_size);
    
    global_size = (M_level_3+2*pad_size)*(N_level_3+2*pad_size) + (1024 - ((M_level_3+2*pad_size)*(N_level_3+2*pad_size))%local_size);
	global_work_size[0] = global_size;
    
    enqueueBuffer(cmd_queue, ck_kernel[1], NULL, global_work_size, local_work_size);
    clFinish(cmd_queue);
    
    // Adaptive smooth level 3
    global_size = M_level_3*N_level_3 + (1024 - (M_level_3*N_level_3)%local_size);
    global_work_size[0] = global_size;
    
    ciErr = clSetKernelArg(ck_kernel[0], 0, sizeof(cl_mem), (void*)&buffer_level_3);
    ciErr = clSetKernelArg(ck_kernel[0], 1, sizeof(cl_mem), (void*)&buffer_level_3_copy);
    ciErr = clSetKernelArg(ck_kernel[0], 2, sizeof(cl_int), (void*)&kernel_size);
    ciErr = clSetKernelArg(ck_kernel[0], 3, sizeof(cl_float), (void*)&SNR_target);
    ciErr = clSetKernelArg(ck_kernel[0], 4, sizeof(cl_int), (void*)&M_level_3);
    ciErr = clSetKernelArg(ck_kernel[0], 5, sizeof(cl_int), (void*)&N_level_3);
    ciErr = clSetKernelArg(ck_kernel[0], 6, sizeof(cl_int), (void*)&pad_size);
    
    ciErr = clSetKernelArg(ck_kernel[4], 0, sizeof(cl_mem), (void*)&buffer_level_3);
    ciErr = clSetKernelArg(ck_kernel[4], 1, sizeof(cl_mem), (void*)&buffer_level_3_copy);
    ciErr = clSetKernelArg(ck_kernel[4], 2, sizeof(cl_int), (void*)&M_level_3);
    ciErr = clSetKernelArg(ck_kernel[4], 3, sizeof(cl_int), (void*)&N_level_3);
    ciErr = clSetKernelArg(ck_kernel[4], 4, sizeof(cl_int), (void*)&pad_size);
    
    for (int i = 0; i < iter_limit; i++) { // Execute iteration, wait for it to finish, then do next iteration
        enqueueBuffer(cmd_queue, ck_kernel[0], NULL, global_work_size, local_work_size);
        enqueueBuffer(cmd_queue, ck_kernel[4], NULL, global_work_size, local_work_size);
		clFinish(cmd_queue);
    }
    
    // sub level 3
    ciErr = clSetKernelArg(ck_kernel[2], 0, sizeof(cl_mem), (void*)&buffer_level_2_reduced);
    ciErr = clSetKernelArg(ck_kernel[2], 1, sizeof(cl_mem), (void*)&buffer_level_3);
    ciErr = clSetKernelArg(ck_kernel[2], 2, sizeof(cl_int), (void*)&M_level_3);
    ciErr = clSetKernelArg(ck_kernel[2], 3, sizeof(cl_int), (void*)&N_level_3);
    ciErr = clSetKernelArg(ck_kernel[2], 4, sizeof(cl_int), (void*)&pad_size);
    
    enqueueBuffer(cmd_queue, ck_kernel[2], NULL, global_work_size, local_work_size);
    clFinish(cmd_queue);
    
    
    // int_sub level 2
    global_size = M_level_2*N_level_2 + (1024 - (M_level_2*N_level_2)%local_size);
    global_work_size[0] = global_size;
    
    ciErr = clSetKernelArg(ck_kernel[3], 0, sizeof(cl_mem), (void*)&buffer_level_2);
    ciErr = clSetKernelArg(ck_kernel[3], 1, sizeof(cl_mem), (void*)&buffer_level_2_reduced);
    ciErr = clSetKernelArg(ck_kernel[3], 2, sizeof(cl_int), (void*)&M_level_2);
    ciErr = clSetKernelArg(ck_kernel[3], 3, sizeof(cl_int), (void*)&N_level_2);
    ciErr = clSetKernelArg(ck_kernel[3], 4, sizeof(cl_int), (void*)&pad_size);
    
    enqueueBuffer(cmd_queue, ck_kernel[3], NULL, global_work_size, local_work_size);
    clFinish(cmd_queue);
    
    // adaptive smooth level 2
    global_size = M_level_2*N_level_2 + (1024 - (M_level_2*N_level_2)%local_size);
    global_work_size[0] = global_size;
    
    ciErr = clSetKernelArg(ck_kernel[0], 0, sizeof(cl_mem), (void*)&buffer_level_2);
    ciErr = clSetKernelArg(ck_kernel[0], 1, sizeof(cl_mem), (void*)&buffer_level_2_copy);
    ciErr = clSetKernelArg(ck_kernel[0], 2, sizeof(cl_int), (void*)&kernel_size);
    ciErr = clSetKernelArg(ck_kernel[0], 3, sizeof(cl_float), (void*)&SNR_target);
    ciErr = clSetKernelArg(ck_kernel[0], 4, sizeof(cl_int), (void*)&M_level_2);
    ciErr = clSetKernelArg(ck_kernel[0], 5, sizeof(cl_int), (void*)&N_level_2);
    ciErr = clSetKernelArg(ck_kernel[0], 6, sizeof(cl_int), (void*)&pad_size);
    
    ciErr = clSetKernelArg(ck_kernel[4], 0, sizeof(cl_mem), (void*)&buffer_level_2);
    ciErr = clSetKernelArg(ck_kernel[4], 1, sizeof(cl_mem), (void*)&buffer_level_2_copy);
    ciErr = clSetKernelArg(ck_kernel[4], 2, sizeof(cl_int), (void*)&M_level_2);
    ciErr = clSetKernelArg(ck_kernel[4], 3, sizeof(cl_int), (void*)&N_level_2);
    ciErr = clSetKernelArg(ck_kernel[4], 4, sizeof(cl_int), (void*)&pad_size);
    
    for (int i = 0; i < iter_limit; i++) { // Execute iteration, wait for it to finish, then do next iteration
        enqueueBuffer(cmd_queue, ck_kernel[0], NULL, global_work_size, local_work_size);
        enqueueBuffer(cmd_queue, ck_kernel[4], NULL, global_work_size, local_work_size);
		clFinish(cmd_queue);
    }
    
    // sub level 2
    ciErr = clSetKernelArg(ck_kernel[2], 0, sizeof(cl_mem), (void*)&buffer_level_1_reduced);
    ciErr = clSetKernelArg(ck_kernel[2], 1, sizeof(cl_mem), (void*)&buffer_level_2);
    ciErr = clSetKernelArg(ck_kernel[2], 2, sizeof(cl_int), (void*)&M_level_2);
    ciErr = clSetKernelArg(ck_kernel[2], 3, sizeof(cl_int), (void*)&N_level_2);
    ciErr = clSetKernelArg(ck_kernel[2], 4, sizeof(cl_int), (void*)&pad_size);
    
    enqueueBuffer(cmd_queue, ck_kernel[2], NULL, global_work_size, local_work_size);
    clFinish(cmd_queue);
    
    
    // int_sub level 1
    global_size = M*N + (1024 - (M*N)%local_size);
    global_work_size[0] = global_size;
    
    ciErr = clSetKernelArg(ck_kernel[3], 0, sizeof(cl_mem), (void*)&buffer_level_1);
    ciErr = clSetKernelArg(ck_kernel[3], 1, sizeof(cl_mem), (void*)&buffer_level_1_reduced);
    ciErr = clSetKernelArg(ck_kernel[3], 2, sizeof(cl_int), (void*)&M);
    ciErr = clSetKernelArg(ck_kernel[3], 3, sizeof(cl_int), (void*)&N);
    ciErr = clSetKernelArg(ck_kernel[3], 4, sizeof(cl_int), (void*)&pad_size);
    
    enqueueBuffer(cmd_queue, ck_kernel[3], NULL, global_work_size, local_work_size);
    clFinish(cmd_queue);
    
    // adaptive smooth level 1
    ciErr = clSetKernelArg(ck_kernel[0], 0, sizeof(cl_mem), (void*)&buffer_level_1);
    ciErr = clSetKernelArg(ck_kernel[0], 1, sizeof(cl_mem), (void*)&buffer_level_1_copy);
    ciErr = clSetKernelArg(ck_kernel[0], 2, sizeof(cl_int), (void*)&kernel_size);
    ciErr = clSetKernelArg(ck_kernel[0], 3, sizeof(cl_float), (void*)&SNR_target);
    ciErr = clSetKernelArg(ck_kernel[0], 4, sizeof(cl_int), (void*)&M);
    ciErr = clSetKernelArg(ck_kernel[0], 5, sizeof(cl_int), (void*)&N);
    ciErr = clSetKernelArg(ck_kernel[0], 6, sizeof(cl_int), (void*)&pad_size);
    
    ciErr = clSetKernelArg(ck_kernel[4], 0, sizeof(cl_mem), (void*)&buffer_level_1);
    ciErr = clSetKernelArg(ck_kernel[4], 1, sizeof(cl_mem), (void*)&buffer_level_1_copy);
    ciErr = clSetKernelArg(ck_kernel[4], 2, sizeof(cl_int), (void*)&M);
    ciErr = clSetKernelArg(ck_kernel[4], 3, sizeof(cl_int), (void*)&N);
    ciErr = clSetKernelArg(ck_kernel[4], 4, sizeof(cl_int), (void*)&pad_size);
    
    for (int i = 0; i < iter_limit; i++) { // Execute iteration, wait for it to finish, then do next iteration
        enqueueBuffer(cmd_queue, ck_kernel[0], NULL, global_work_size, local_work_size);
        enqueueBuffer(cmd_queue, ck_kernel[4], NULL, global_work_size, local_work_size);
		clFinish(cmd_queue);
    }
    
    // Retrieve final image from GPU
    writeDeviceToHost(cmd_queue, buffer_level_1, im_level_1, padded_size, CL_TRUE);
    
    float** out_im = init_array(N, M);
    
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            out_im[i][j] = im_level_1[(i+pad_size)*padded_width + (j+pad_size)];
        }
    }
    
    free_array(padded_im, padded_height);
    free(padded_im_concat);
    free(platforms);
    free(devices);
    
    // Free memory on GPU
    clReleaseMemObject(buffer_level_1);
    clReleaseMemObject(buffer_level_1_copy);
    clReleaseMemObject(buffer_level_2);
    clReleaseMemObject(buffer_level_2_copy);
    clReleaseMemObject(buffer_level_3);
    clReleaseMemObject(buffer_level_3_copy);
    clReleaseMemObject(buffer_level_1_reduced);
    clReleaseMemObject(buffer_level_2_reduced);
    
    // Free host-side buffers
    free(im_level_1);
    free(im_level_2);
    free(im_level_3);
    free(im_level_1_copy);
    free(im_level_2_copy);
    free(im_level_3_copy);
    free(im_level_1_reduced);
    free(im_level_2_reduced);
    
    // Free OpenCL program constructs
    for (int i = 0; i < num_kernels; i++) {
		freeOpenCLProgram(ck_kernel[i], cp_program[i], cmd_queue, context);
	}
    
    double duration_GPU_total = (clock() - start_GPU_total) / ((double)CLOCKS_PER_SEC);
    //printf("Execution time: %lf\n", duration_GPU_total);
    
    return out_im;
}
