// Copyright (c) Microsoft Corporation.
// All rights reserved.

#include <cuda_runtime.h>
#include <cuda.h>

#include "probe_gpu_bandwidth.hpp"
#include <iostream>

int gpu_count() {
  int deviceCount;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  
  if (error_id != cudaSuccess) {
    std::cerr << "cudaGetDeviceCount returned " << error_id
              << ": " << cudaGetErrorString(error_id)
              << std::endl;;
    exit(1);
  }

  return deviceCount;
}


double probe_gpu_bandwidth_from_numa_node(int numa_node, int gpuA, int gpuB) {
  std::cout << "Probing bandwidth to " << gpuA << " and " << gpuB
            << " from " << numa_node
            << std::endl;
  std::cout << "(not implemented)" << std::endl;

  // plan:
  // - allocate buffers on each GPU
  // - allocate buffers for each GPU on numa node
  // - enqueue N kernels copying from GPU to buffer
  // - time kernels
  // - compute aggregate bandwidth of copies
  
  return 0.0;
}
