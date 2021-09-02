adaptive_smooth_multigrid_GPU: adaptive_smooth.c gauss_filter.c array_utils.c adaptive_smooth_multigrid.c adaptive_smooth_multigrid_GPU.c adaptive_smooth_execute_GPU.c OpenCL_functions.c
	gcc -std=gnu99 -shared -o adaptive_smooth.so adaptive_smooth.c gauss_filter.c array_utils.c adaptive_smooth_multigrid.c adaptive_smooth_multigrid_GPU.c adaptive_smooth_execute_GPU.c OpenCL_functions.c -O2 -fopenmp -lOpenCL -lm -fPIC

adaptive_smooth_multigrid_CPU: adaptive_smooth.c gauss_filter.c array_utils.c adaptive_smooth_multigrid.c adaptive_smooth_execute_CPU.c
	gcc -std=gnu99 -shared -o adaptive_smooth.so adaptive_smooth.c gauss_filter.c array_utils.c adaptive_smooth_multigrid.c adaptive_smooth_execute_CPU.c -O2 -fopenmp -lm -fPIC
