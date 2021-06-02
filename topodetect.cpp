// Copyright (c) Microsoft Corporation.
// All rights reserved.

#include "probe.hpp"

#include <gflags/gflags.h>
#include <numa.h>

#include <vector>
#include <limits>

DEFINE_int32(warmup,         0, "Number of warmup iterations to run before timing.");
DEFINE_int32(iters,          1, "Number of timed iterations to run.");
DEFINE_int64(length, 1LL << 30, "Length of test buffers in bytes (defaults to 1 GB).");


int main(int argc, char * argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Plan:
  //
  // Step 1: identify which socket the NIC is connected to.
  // For each socket,
  //   Set up a loopback connection through the NIC and measure latency
  // The socket with the lowest latency should be the one the NIC is connected to.
  //
  // Step 2: ...

  if (-1 == numa_available()) {
    std::cerr << "NUMA not available. Cannot probe topology." << std::endl;
    exit(1);
  }

  // measure latency on each NUMA node. Track the lowest
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
  
  // double core0_latency  = probe_latency_from_core(0);
  // double core19_latency = probe_latency_from_core(19);

  // if (core0_latency < core19_latency) {
  //   std::cout << "NIC appears to be near core 0" << std::endl;
  // } else {
  //   std::cout << "NIC appears to be near core 19" << std::endl;
  // }
  



    
  std::cout << "Done." << std::endl;
  return 0;
}
