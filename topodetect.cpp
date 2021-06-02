// Copyright (c) Microsoft Corporation.
// All rights reserved.

#include "probe_latency.hpp"
#include "LoopbackFlow.hpp"

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

  // measure latency on each NUMA node. Remember the lowest.
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
  

  { // Start loopback flow between NIC and the DRAM of the CPU it is nearest
    LoopbackFlow flow(min_numa_node);

    // TODO: 
    sleep(5);
  }

    
  std::cout << "Done." << std::endl;
  return 0;
}
