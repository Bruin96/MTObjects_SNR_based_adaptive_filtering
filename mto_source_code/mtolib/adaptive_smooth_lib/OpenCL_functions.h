#ifndef OPENCL_FUNCTIONS_H
#define OPENCL_FUNCTIONS_H

#include <CL/cl.h>

cl_platform_id* getPlatforms(cl_uint* numPlatforms);
cl_device_id* getDevices(cl_uint* numDevices, cl_platform_id* platforms, cl_uint numPlatforms);
cl_context createContext(cl_device_id* devices, cl_uint numDevices);
cl_command_queue createCommandQueue(cl_device_id* devices, cl_context context);
cl_program createProgram(cl_context context, char* kernel_filename);
void buildProgram(cl_device_id* devices, cl_program cpProgram);
cl_mem createBuffer(cl_context context, cl_float* srcA, int numElems, cl_mem_flags read_write_flag);

cl_float* convertToCLfloat(float* A, int numElems);
cl_float* initialiseCLarray(int numElems);

void writeHostToDevice(cl_command_queue cmdQueue, cl_mem bufferA, cl_float* srcA, int numElems, cl_bool is_blocking);
void writeDeviceToHost(cl_command_queue cmdQueue, cl_mem bufferC, cl_float* srcC, int numElems, cl_bool is_blocking);

cl_kernel createKernel(cl_program cpProgram, char* function_name);
void enqueueBuffer(cl_command_queue cmdQueue, cl_kernel ckKernel, cl_event kernelevent,
    size_t* szGlobalWorkSize, size_t* szLocalWorkSize);

void freeBuffer(cl_mem buffer);
void freeOpenCLProgram(cl_kernel ckKernel, cl_program cpProgram, cl_command_queue cmdQueue, cl_context context);

#endif
