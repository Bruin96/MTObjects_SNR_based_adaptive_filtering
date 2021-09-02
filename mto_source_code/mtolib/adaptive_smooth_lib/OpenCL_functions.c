#include "OpenCL_functions.h"

#include <stdio.h>

cl_platform_id* getPlatforms(cl_uint *numPlatforms) {
    cl_platform_id* platforms = (cl_platform_id*)malloc(1000 * sizeof(cl_platform_id));

    cl_int status = clGetPlatformIDs(1000, platforms, numPlatforms);
    
    
    //printf("numPlatforms = %d\n", *numPlatforms);
    for (int i = 0; i < *numPlatforms; i++) {
        char vendor[1024];
        clGetPlatformInfo (platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
        //printf("platform vendor = %s\n", vendor);
    }
    

    if (status != CL_SUCCESS) {
        printf("Couldn't get platform IDs. Error = %d\n", status);
        return NULL;
    }

    return platforms;
}

cl_device_id* getDevices(cl_uint* numDevices, cl_platform_id *platforms, cl_uint numPlatforms) {
    cl_int status = clGetDeviceIDs(platforms[numPlatforms-1], CL_DEVICE_TYPE_GPU, 0,
        NULL, numDevices); // Discover the number of devices

    cl_device_id* devices = (cl_device_id*)malloc((*numDevices) * sizeof(cl_device_id));

    // Fill in devices
    status = clGetDeviceIDs(platforms[numPlatforms-1], CL_DEVICE_TYPE_GPU, *numDevices,
        devices, NULL);
        
    if (status != CL_SUCCESS) {
        printf("ERROR: Failed to create a device group. Error = %d\n", status);
        return NULL;
    }

    return devices;
}

cl_context createContext(cl_device_id* devices, cl_uint numDevices) {
    cl_int status;
    cl_context context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);

    if (!context) {
        printf("Error: Failed to create a compute context.\n");
        return NULL;
    }

    return context;
}

cl_command_queue createCommandQueue(cl_device_id* devices, cl_context context) {
    cl_int status;
    cl_command_queue cmdQueue = clCreateCommandQueue(context,
        devices[0], CL_QUEUE_PROFILING_ENABLE, &status);

    if (!cmdQueue) {
        printf("Error: Failed to create a command queue.\n");
    }

    return cmdQueue;
}

cl_program createProgram(cl_context context, char* kernel_filename) {
    FILE* programHandle = fopen(kernel_filename, "rb");
    if (programHandle == NULL) {
        printf("An error occurred while opening the file.\n");
        return NULL;
    }

    fseek(programHandle, 0, SEEK_END);
    size_t programSize = ftell(programHandle);
    rewind(programHandle);

    // Read kernel into program buffer
    char* programBuffer = (char*)malloc((programSize + 1) * sizeof(char));
    programBuffer[programSize] = '\0';
    int errno = fread(programBuffer, sizeof(char), programSize, programHandle);
    fclose(programHandle);

    cl_int ciErr;
    cl_program cpProgram = clCreateProgramWithSource(context, 1,
        (const char**)&programBuffer, &programSize, &ciErr);

    if (!cpProgram) {
        printf("Error: Failed to create compute program.\n");
        return NULL;
    }
    free(programBuffer);

    return cpProgram;
}

void buildProgram(cl_device_id* devices, cl_program cpProgram) {
    cl_int ciErr = clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);

    if (ciErr != CL_SUCCESS) {
        size_t len;
        char buffer[8192];
        printf("Error: Failed to build program executable.\n");
        clGetProgramBuildInfo(cpProgram, devices[0], CL_PROGRAM_BUILD_LOG,
            sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }
}

cl_mem createBuffer(cl_context context, cl_float* srcA, int numElems, cl_mem_flags read_write_flag) {
    cl_int status;
    size_t datasize = sizeof(cl_float) * numElems;

    cl_mem bufferA = clCreateBuffer(context, read_write_flag,
        datasize, NULL, &status);

    return bufferA;
}

cl_float* convertToCLfloat(float* A, int numElems) {
    cl_float* srcA = initialiseCLarray(numElems);

    for (int i = 0; i < numElems; ++i) {
        *((cl_float*)srcA + i) = A[i];
    }

    return srcA;
}

cl_float* initialiseCLarray(int numElems) {
    return (cl_float*)malloc(sizeof(cl_float) * numElems);
}

void writeHostToDevice(cl_command_queue cmdQueue, cl_mem bufferA, cl_float* srcA, int numElems, cl_bool is_blocking) {
    size_t datasize = sizeof(cl_float) * numElems;

    cl_int status = clEnqueueWriteBuffer(cmdQueue, bufferA, is_blocking,
        0, datasize, srcA, 0, NULL, NULL);

    if (status != CL_SUCCESS) {
        printf("Error: Failed to write host data A to device buffer.\n");
        exit(1);
    }
}

void writeDeviceToHost(cl_command_queue cmdQueue, cl_mem bufferC, cl_float* srcC, int numElems, cl_bool is_blocking) {
    size_t datasize = sizeof(cl_float) * numElems;
    cl_int ciErr = clEnqueueReadBuffer(cmdQueue, bufferC, is_blocking, 0, datasize,
        srcC, 0, NULL, NULL);
}

cl_kernel createKernel(cl_program cpProgram, char* function_name) {
    cl_int ciErr;
    cl_kernel ckKernel = clCreateKernel(cpProgram, function_name, &ciErr);

    if (!ckKernel || ciErr != CL_SUCCESS) {
        printf("Error: Failed to create compute kernel.\n");
        return NULL;
    }

    return ckKernel;
}

void enqueueBuffer(cl_command_queue cmdQueue, cl_kernel ckKernel, cl_event kernelevent, 
    size_t szGlobalWorkSize[], size_t szLocalWorkSize[]) {
    cl_int ciErr;
    ciErr = clEnqueueNDRangeKernel(cmdQueue, ckKernel, 2, NULL,
        szGlobalWorkSize, szLocalWorkSize, 0, NULL, &kernelevent);

    if (ciErr != CL_SUCCESS) {
        printf("ciErr = %d\n", ciErr);
        printf("Error launching kernel.\n");
        exit(1);
    }
}

void freeBuffer(cl_mem buffer) {
    if (buffer) clReleaseMemObject(buffer);
}

void freeOpenCLProgram(cl_kernel ckKernel, cl_program cpProgram, cl_command_queue cmdQueue, cl_context context) {
    if (ckKernel) clReleaseKernel(ckKernel);
    if (cpProgram) clReleaseProgram(cpProgram);
    if (cmdQueue) clReleaseCommandQueue(cmdQueue);
    if (context) clReleaseContext(context);
}
