// Copyright (c) Microsoft Corporation.
// All rights reserved.

#include <cuda_runtime.h>
#include <cuda.h>

#include "probe_gpu_bandwidth.hpp"
#include <iostream>
#include <chrono>

#include <x86intrin.h>
#include <numa.h>
#include <gflags/gflags.h>

DECLARE_int64(length);
DEFINE_int32(bw_iters, 10, "Number of iterations to run when measuring GPU bandwidth.");

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


double probe_gpu_bandwidth_from_numa_node(int numa_nodeA, int gpuA, int numa_nodeB, int gpuB) {
  // std::cout << "Probing bandwidth with " <<
  // 	    << " for GPU doing " << gpuA << " on NUMA node " << numa_nodeA
  // 	    << " and GPU " << gpuB << " on NUMA node " << numa_nodeB
  //           << " from " << numa_node
  //           << std::endl;

  // plan:
  // - allocate buffers on each GPU
  // - allocate buffers for each GPU on numa node
  // - enqueue N kernels copying from GPU to buffer
  // - time kernels
  // - compute aggregate bandwidth of copies

  numa_run_on_node(numa_nodeA);  
  
  void * gpuA_host_buf = nullptr;
  void * gpuA_device_buf = nullptr;
  CHECK_CUDA(cudaSetDevice(gpuA));
  CHECK_CUDA(cudaHostAlloc(&gpuA_host_buf, FLAGS_length, cudaHostAllocMapped));
  CHECK_CUDA(cudaMalloc(&gpuA_device_buf, FLAGS_length));  

  cudaStream_t gpuA_stream;
  CHECK_CUDA(cudaStreamCreateWithFlags(&gpuA_stream, cudaStreamNonBlocking));

  //numa_run_on_node(numa_nodeB);
  
  void * gpuB_host_buf = nullptr;
  void * gpuB_device_buf = nullptr;
  CHECK_CUDA(cudaSetDevice(gpuB));
  CHECK_CUDA(cudaHostAlloc(&gpuB_host_buf, FLAGS_length, cudaHostAllocMapped));
  CHECK_CUDA(cudaMalloc(&gpuB_device_buf, FLAGS_length));

  cudaStream_t gpuB_stream;
  CHECK_CUDA(cudaStreamCreateWithFlags(&gpuB_stream, cudaStreamNonBlocking));

  // allow us to run anywhere
  //numa_run_on_node(-1);
    
  auto start_time = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < FLAGS_bw_iters; ++i) {
    CHECK_CUDA(cudaSetDevice(gpuA));  
    CHECK_CUDA(cudaMemcpyAsync(gpuA_host_buf, gpuA_device_buf, FLAGS_length, cudaMemcpyDeviceToHost, gpuA_stream));
    CHECK_CUDA(cudaSetDevice(gpuB));  
    //CHECK_CUDA(cudaMemcpyAsync(gpuB_host_buf, gpuB_device_buf, FLAGS_length, cudaMemcpyDeviceToHost, gpuB_stream));
    CHECK_CUDA(cudaMemcpyAsync(gpuB_device_buf, gpuB_host_buf, FLAGS_length, cudaMemcpyHostToDevice, gpuB_stream));    
  }

  CHECK_CUDA(cudaSetDevice(gpuA));
  CHECK_CUDA(cudaStreamSynchronize(gpuA_stream));
  CHECK_CUDA(cudaSetDevice(gpuB));
  CHECK_CUDA(cudaStreamSynchronize(gpuB_stream));
  auto end_time = std::chrono::high_resolution_clock::now();  

  // free memory
  CHECK_CUDA(cudaFreeHost(gpuA_host_buf));
  CHECK_CUDA(cudaFreeHost(gpuB_host_buf));
  CHECK_CUDA(cudaFree(gpuA_device_buf));
  CHECK_CUDA(cudaFree(gpuB_device_buf));
  
  uint64_t time_difference_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
  double bw = (double) FLAGS_length / (time_difference_ns / 1e9) / 1024 / 1024 / 1024 * FLAGS_bw_iters;
  std::cout << "Measured per-GPU bandwidth of " << bw
	    << " for GPU " << gpuA << " on NUMA node " << numa_nodeA << " doing DtoH" 
	    << " and GPU " << gpuB << " on NUMA node " << numa_nodeB << " doing HtoD" 
	    << std::endl;
  return bw;
}
