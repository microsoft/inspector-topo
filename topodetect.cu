// Copyright (c) Microsoft Corporation.
// All rights reserved.

#include "probe_latency.hpp"
#include "LoopbackFlow.hpp"
#include "probe_gpu_bandwidth.hpp"

#include <gflags/gflags.h>
#include <numa.h>

#include <vector>
#include <limits>

DEFINE_int32(warmup,       100, "Number of warmup iterations to run before timing (defults to 100).");
DEFINE_int32(iters,       1000, "Number of timed iterations to run (defaults to 1000).");
DEFINE_int64(length, 1LL << 30, "Length of test buffers in bytes (defaults to 1 GB).");

int main(int argc, char * argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (-1 == numa_available()) {
    std::cerr << "NUMA not available. Cannot probe topology." << std::endl;
    exit(1);
  }

  // Measure latency on each NUMA node. Remember the lowest.
  double min_latency = std::numeric_limits<double>::max();
  int min_numa_node = 0;
  for (int node = 0; node <= numa_max_node(); ++node) {
    double latency = probe_latency_from_numa_node(node);
    if (latency < min_latency) {
      min_latency = latency;
      min_numa_node = node;
    }
  }
  
  std::cout << "NIC appears to be nearest to NUMA node " << min_numa_node << std::endl;
  

  // Now probe bandwith to all pairs of GPUs while NIC is processing a
  // loopback flow.
  int gpu_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&gpu_count));
  if (1 == gpu_count) {
    std::cout << "Simple case: single GPU. Not detecting NUMA node." << std::endl;
  } else {
    double gpu_pair_bw[gpu_count][gpu_count] = {0.0};

    // Track the pair with highest aggregate bandwidth.    
    double max_bw = std::numeric_limits<double>::min();
    int max_gpuA = 0;
    int max_gpuB = 1;

    // Track the pair with lowest aggregate bandwidth. This should be
    // the one shared with the NIC.
    double min_bw = std::numeric_limits<double>::max();
    int min_gpuA = 0;
    int min_gpuB = 1;

    { // Start loopback flow between NIC and the DRAM of the CPU it is nearest.
      LoopbackFlow flow(min_numa_node);
      
      for (int gpuA = 0; gpuA < gpu_count; ++gpuA) {
        for (int gpuB = gpuA + 1; gpuB < gpu_count; ++gpuB) {
          double bw = probe_gpu_bandwidth_from_numa_node(min_numa_node, gpuA, gpuB);
          gpu_pair_bw[gpuA][gpuB] = bw;
          
          // update best pair
          if (bw > max_bw) {
            max_bw = bw;
            max_gpuA = gpuA;
            max_gpuB = gpuB;
          }

          // update worst pair
          if (bw < min_bw) {
            min_bw = bw;
            min_gpuA = gpuA;
            min_gpuB = gpuB;
          }
        }
      }
    }

    std::cout << "Best GPU pair was " << max_gpuA << " and " << max_gpuB
              << " with a bandwidth of " << max_bw
              << std::endl;
    std::cout << "GPU pair shared with NIC appears to be " << min_gpuA << " and " << min_gpuB
              << " with a bandwidth of " << min_bw
              << std::endl;
  }

    
  std::cout << "Done." << std::endl;
  return 0;
}
