// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef __PROBE_GPU_BANDWIDTH__
#define __PROBE_GPU_BANDWIDTH__

#include <cuda.h>
#include <numa.h>
#include <iostream>
#include <vector>

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


class GPUBuffers {
private:
  int gpu_count;
  const int numa_node_count;
  const bool use_write_combining;
  
  std::vector<void *> cpu_buffers;
  std::vector<void *> gpu_buffers;
  std::vector<cudaStream_t> gpu_streams;

  void * allocate_cpu_buffer(int numa_node);
  void * allocate_gpu_buffer(int gpu_id);
  void allocate_gpu_stream(int gpu_id, cudaStream_t *);

  void free_cpu_buffer(void *);
  void free_gpu_buffer(void *);
  void free_gpu_stream(cudaStream_t *);
  
public:
  GPUBuffers(bool wc = false)
    : gpu_count(-1)
    , numa_node_count(numa_max_node()+1)
    , use_write_combining(wc)
    , cpu_buffers(numa_node_count)
    , gpu_buffers()
    , gpu_streams()
  {
    CHECK_CUDA(cudaGetDeviceCount(&gpu_count));

    // allocate one buffer on each NUMA node
    for (int cpu = 0; cpu < numa_node_count; ++cpu) {
      cpu_buffers[cpu] = allocate_cpu_buffer(cpu);
    }

    // allocate one buffer on each GPU
    gpu_buffers.resize(gpu_count);
    for (int gpu = 0; gpu < gpu_count; ++gpu) {
      gpu_buffers[gpu] = allocate_gpu_buffer(gpu);
    }

    // allocate a stream for each GPU
    gpu_streams.resize(gpu_count);
    for (int gpu = 0; gpu < gpu_count; ++gpu) {
      allocate_gpu_stream(gpu, &gpu_streams[gpu]);
    }
  }

  void free_all() {
    for (void * buf : cpu_buffers) {
      free_cpu_buffer(buf);
    }
    cpu_buffers.clear();
    
    for (void * buf : gpu_buffers) {
      free_gpu_buffer(buf);
    }
    gpu_buffers.clear();

    for (cudaStream_t & stream : gpu_streams) {
      free_gpu_stream(&stream);
    }
    gpu_streams.clear();
  }

  ~GPUBuffers() {
    free_all();
  }
  
  int get_gpu_count() const { return gpu_count; }
  int get_cpu_count() const { return numa_node_count; }

  void * get_cpu_buffer(int cpu_id) const { return cpu_buffers[cpu_id]; }
  void * get_gpu_buffer(int gpu_id) const { return gpu_buffers[gpu_id]; };

  double double_memcpy_probe(int numa_nodeA, int gpu_idA, bool htod_or_dtohA,
			     int numa_nodeB, int gpu_idB, bool htod_or_dtohB);
};



#endif // __PROBE_GPU_BANDWIDTH__
