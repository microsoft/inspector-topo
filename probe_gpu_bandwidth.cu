// Copyright (c) Microsoft Corporation.
// All rights reserved.

#include <cuda_runtime.h>
#include <cuda.h>

#include "probe_gpu_bandwidth.hpp"
#include <iostream>
#include <chrono>

//#include <x86intrin.h> // TODO: nvcc doesn't like this on some CUDA versions with some GCC versions?
#include <numa.h>
#include <gflags/gflags.h>

DECLARE_int64(length);
DEFINE_int32(bw_iters, 10, "Number of iterations to run when measuring GPU bandwidth.");
DEFINE_int32(bw_warmup_iters, 1, "Number of warmup iterations to run when measuring GPU bandwidth.");

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


void * GPUBuffers::allocate_cpu_buffer(int numa_node) {
#ifdef DEBUG_LOG
  std::cout << "Allocating CPU buffer on node " << numa_node << "." << std::endl;
#endif
  
  // what NUMA node are we allocating from right now?
  int prev_preferred = -1; //numa_preferred();

  // set NUMA node for this allocation
  numa_set_preferred(numa_node);

  // allocate buffer
  void * host_buf = nullptr;
  if (use_write_combining) {
    CHECK_CUDA(cudaHostAlloc(&host_buf, FLAGS_length, cudaHostAllocWriteCombined));
  } else {
    CHECK_CUDA(cudaHostAlloc(&host_buf, FLAGS_length, cudaHostAllocMapped));
  }

  // restore previous NUMA policy
  numa_set_preferred(prev_preferred);

#ifdef DEBUG_LOG
  std::cout << "Allocated CPU buffer on node " << numa_node << " at " << host_buf << "." << std::endl;
#endif

  // return buffer
  return host_buf;
}

void GPUBuffers::free_cpu_buffer(void * buf) {
  CHECK_CUDA(cudaFreeHost(buf));
}

void * GPUBuffers::allocate_gpu_buffer(int gpu_id) {
#ifdef DEBUG_LOG
  std::cout << "Allocating GPU buffer on GPU " << gpu_id << "." << std::endl;
#endif
  
  void * device_buf = nullptr;
  CHECK_CUDA(cudaSetDevice(gpu_id));
  CHECK_CUDA(cudaMalloc(&device_buf, FLAGS_length));

#ifdef DEBUG_LOG
  std::cout << "Allocating GPU buffer at " << device_buf << " on GPU " << gpu_id << "." << std::endl;
#endif
  return device_buf;
}

void GPUBuffers::free_gpu_buffer(void * buf) {
  CHECK_CUDA(cudaFree(buf));
}

void GPUBuffers::allocate_gpu_stream(int gpu_id, cudaStream_t * stream_p) {
#ifdef DEBUG_LOG
  std::cout << "Allocating GPU stream on GPU " << gpu_id << "." << std::endl;
#endif

  CHECK_CUDA(cudaSetDevice(gpu_id));
  CHECK_CUDA(cudaStreamCreateWithFlags(stream_p, cudaStreamNonBlocking));

#ifdef DEBUG_LOG
  std::cout << "Allocated GPU stream " << *stream_p << " on GPU " << gpu_id << "." << std::endl;
#endif
}

void GPUBuffers::free_gpu_stream(cudaStream_t * stream_p) {
  CHECK_CUDA(cudaStreamDestroy(*stream_p));
  *stream_p = nullptr;
}


double GPUBuffers::double_memcpy_probe(int numa_nodeA, int gpuA, bool htod_or_dtohA, 
                                       int numa_nodeB, int gpuB, bool htod_or_dtohB) {
#ifdef DEBUG_LOG
  std::cout << "Doing memcpy "
	    << " between GPU " << gpuA << " buffer " << gpu_buffers[gpuA]
	    << " and NUMA node " << numa_nodeA << " buffer " << cpu_buffers[numa_nodeA]
	    << " doing " << (htod_or_dtohA ? "HtoD" : "DtoH")
	    << " and between GPU " << gpuB << " buffer " << gpu_buffers[gpuB]
	    << " and NUMA node " << numa_nodeB << " buffer " << cpu_buffers[numa_nodeB]
	    << " doing " << (htod_or_dtohB ? "HtoD" : "DtoH")
	    << std::endl;
#endif

  auto copy = [&] (int numa_node, int gpu, bool htod_or_dtoh) {
    CHECK_CUDA(cudaSetDevice(gpu));    
    if (htod_or_dtoh) {
      CHECK_CUDA(cudaMemcpyAsync(cpu_buffers[numa_node], gpu_buffers[gpu], FLAGS_length,
				 cudaMemcpyDeviceToHost, gpu_streams[gpu]));
    } else {
      CHECK_CUDA(cudaMemcpyAsync(gpu_buffers[gpu], cpu_buffers[numa_node], FLAGS_length,
				 cudaMemcpyHostToDevice, gpu_streams[gpu]));
    }
  };

  auto run = [&] (int iters) {
    for (int i = 0; i < iters; ++i) {
      copy(numa_nodeA, gpuA, htod_or_dtohA);
      copy(numa_nodeB, gpuB, htod_or_dtohB);
    }
    
    CHECK_CUDA(cudaSetDevice(gpuA));
    CHECK_CUDA(cudaStreamSynchronize(gpu_streams[gpuA]));
    CHECK_CUDA(cudaSetDevice(gpuB));
    CHECK_CUDA(cudaStreamSynchronize(gpu_streams[gpuB]));
  };
  
  // warmup iteration
  run(FLAGS_bw_warmup_iters);
  
  // timed run
  auto start_time = std::chrono::high_resolution_clock::now();
  run(FLAGS_bw_iters);
  auto end_time = std::chrono::high_resolution_clock::now();  

  uint64_t time_difference_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
  double bw = (double) FLAGS_length / (time_difference_ns / 1e9) / 1024 / 1024 / 1024 * FLAGS_bw_iters;
  // std::cout << "Measured per-GPU bandwidth of " << bw
  // 	    << " between GPU " << gpuA << " buffer " << gpu_buffers[gpuA]
  // 	    << " and NUMA node " << numa_nodeA << " buffer " << cpu_buffers[numa_nodeA]
  // 	    << " doing " << (htod_or_dtohA ? "HtoD" : "DtoH")
  // 	    << " and between GPU " << gpuB << " buffer " << gpu_buffers[gpuB]
  // 	    << " and NUMA node " << numa_nodeB << " buffer " << cpu_buffers[numa_nodeB]
  // 	    << " doing " << (htod_or_dtohB ? "HtoD" : "DtoH")
  // 	    << std::endl;

  std::cout << "WC:" << use_write_combining
	    << " GPUA:" << gpuA << " nodeA:" << numa_nodeA << " dirA:" << (htod_or_dtohA ? "HtoD" : "DtoH")
	    << " GPUB:" << gpuB << " nodeB:" << numa_nodeB << " dirB:" << (htod_or_dtohB ? "HtoD" : "DtoH")
	    << " BW:" << bw
  	    << std::endl;
  
  return bw;
}
