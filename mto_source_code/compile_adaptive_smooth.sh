#!/bin/bash

if [ $# -eq 1 ] && [ $1 == "GPU" ]; then # Explicitly compile with GPU
    make -C mtolib/adaptive_smooth_lib adaptive_smooth_multigrid_GPU
elif [ $# -eq 1 ] && [ $1 == "CPU" ]; then # Explicitly compile with CPU
    make -C mtolib/adaptive_smooth_lib adaptive_smooth_multigrid_CPU
elif [ -f "/usr/lib/libOpenCL.so" ]; then # No command line directive, use GPU if library is found
    make -C mtolib/adaptive_smooth_lib adaptive_smooth_multigrid_GPU
else # Could not find OpenCL library, so compile with CPU
    make -C mtolib/adaptive_smooth_lib adaptive_smooth_multigrid_CPU
fi
