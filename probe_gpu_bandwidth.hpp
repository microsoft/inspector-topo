// Copyright (c) Microsoft Corporation.
// All rights reserved.

#ifndef __PROBE_GPU_BANDWIDTH__
#define __PROBE_GPU_BANDWIDTH__

#include <cuda.h>
#include <iostream>

template <typename T>
void check_cuda(T result, char const *const func, const char *const file,
                int const line) {
  if (result) {
    std::cout << "CUDA error at " << file << ":" << line
              << " code=" << static_cast<unsigned int>(result)
      //<< "(" << _cudaGetErrorEnum(result) << ")"
              << " \"" <<  func << "\""
              << std::endl;
    exit(1);
  }
}

#define CHECK_CUDA(val) check_cuda((val), #val, __FILE__, __LINE__)

double probe_gpu_bandwidth_from_numa_node(int nodeA, int gpuA, int nodeB, int gpuB);

#endif // __PROBE_GPU_BANDWIDTH__
