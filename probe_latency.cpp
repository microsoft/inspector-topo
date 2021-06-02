// Copyright (c) Microsoft Corporation.
// All rights reserved.

#include "probe_latency.hpp"

#include <gflags/gflags.h>

#include <iostream>
#include <vector>
#include <algorithm>

#include <numa.h>

#include <x86intrin.h>


DECLARE_int32(warmup);
DECLARE_int32(iters);
DECLARE_int64(length);

/// Run a lambda on a core. Lambda should return a double. Restore
/// previous CPU affinity after running.
template<typename F>
double run_on_core(int16_t core, F f) {
  
  // capture current CPU affinity mask
  cpu_set_t previous_set;
  CPU_ZERO(&previous_set);
  if (sched_getaffinity(getpid(), sizeof(previous_set), &previous_set) < 0) {
    std::cerr << "Error getting current CPU affinity" << std::endl;
    exit(1);
  }

  // set affinity mask for allocation
  cpu_set_t new_set;
  CPU_ZERO(&new_set);
  CPU_SET(core, &new_set);
  if (sched_setaffinity(getpid(), sizeof(new_set), &new_set) < 0) {
    std::cerr << "Error setting current CPU affinity" << std::endl;
    exit(1);
  }

  // run function
  std::cout << "Running on core " << core << std::endl;
  double result = f();
  
  // return affinity to normal
  if (sched_setaffinity(getpid(), sizeof(previous_set), &previous_set) < 0) {
    std::cerr << "Error resetting CPU affinity" << std::endl;
    exit(1);
  }

  return result;
}

/// Run a lambda on a particular NUMA node. Lambda should return a
/// double. Reset things to allow running on any NUMA node afterward.
template<typename F>
double run_on_numa_node(int node, F f) {
  // set NUMA node to that requested
  numa_run_on_node(node);

  // run function
  std::cout << "Running on NUMA node " << node << std::endl;
  double result = f();

  // run on any NUMA node
  numa_run_on_node(-1);

  return result;
}


/// Do 1-byte loopback RDMA READs to probe latency
double probe_latency() {
  // Initialize RDMA NIC
  Endpoint e;

  // Initialize loopback connection
  Loopback lp(e);
  
  // Allocate memory region. We copy to and from the same buffer,
  // since we care only about data movement, not the result.
  ibv_mr * mr = e.allocate(FLAGS_length);

  ibv_sge send_sge;
  send_sge.addr   = reinterpret_cast<uintptr_t>(mr->addr);
  send_sge.length = 1;
  send_sge.lkey   = mr->lkey;
    
  ibv_send_wr send_wr;
  send_wr.wr_id = 0;
  send_wr.next = nullptr;
  send_wr.sg_list = &send_sge;
  send_wr.num_sge = 1;
  send_wr.opcode = IBV_WR_RDMA_READ;
  send_wr.send_flags = IBV_SEND_SIGNALED;
  send_wr.wr.rdma.remote_addr = reinterpret_cast<uintptr_t>(mr->addr);
  send_wr.wr.rdma.rkey = mr->rkey;

  // do warmup iterations
  for (int i = 0; i < FLAGS_warmup; ++i) {
    // post send to start sending
    send_wr.wr_id = i; // set WR ID to iteration
    lp.post_send(&send_wr);

    // wait for completion
    ibv_wc completion;
    while (0 == lp.poll_cq(1, &completion)) {
      ; // just spin
    }

    // got completion; check that it's successful and continue
    if (completion.status != IBV_WC_SUCCESS) {
      std::cerr << "Got eror completion for " << (void*) completion.wr_id
                << " with status " << ibv_wc_status_str(completion.status)
                << std::endl;
      exit(1);
    }
  }
  
  // record the time whenever we complete 
  std::vector<uint64_t> send_times;
  send_times.reserve(FLAGS_iters + 1); // preallocate space for each probe
  
  // Record start time and do probes
  send_times.push_back(__rdtsc()); // record initial time
  for (int i = 0; i < FLAGS_iters; ++i) {
    // post send to start sending
    send_wr.wr_id = i; // set WR ID to iteration
    lp.post_send(&send_wr);

    // wait for completion
    ibv_wc completion;
    while (0 == lp.poll_cq(1, &completion)) {
      ; // just spin
    }

    // got completion; check that it's successful and continue
    if (completion.status != IBV_WC_SUCCESS) {
      std::cerr << "Got eror completion for " << (void*) completion.wr_id
                << " with status " << ibv_wc_status_str(completion.status)
                << std::endl;
      exit(1);
    }

    // record time
    send_times.push_back(__rdtsc());
  }


  // compute time taken for each send
  std::vector<double> time_differences_us;
  for (int i = 0; i < FLAGS_iters; ++i) {
    auto start_time = send_times[i];
    auto end_time = send_times[i+1];
    double time_difference_us = (end_time - start_time) / (e.get_ticks_per_sec() / 1.0e6);
    time_differences_us.push_back(time_difference_us);
  }

  // sort differences and extract latency metrics
  std::sort(time_differences_us.begin(), time_differences_us.end());
  double min_latency = time_differences_us.front();
  double max_latency = time_differences_us.back();
  double p99_latency = time_differences_us[99*time_differences_us.size()/100];
  double median_latency = time_differences_us[time_differences_us.size()/2];

  std::cout << "1-byte RDMA READ latency "
            << "min: " << min_latency
            << "us median: " << median_latency
            << "us p99: " << p99_latency
            << "us max: " << max_latency
            << "us" << std::endl;

  return (double) min_latency;
}

double probe_latency_from_core(int16_t core) {
  return run_on_core(core, []() -> double { return probe_latency(); });
}

double probe_latency_from_numa_node(int node) {
  return run_on_numa_node(node, []() -> double { return probe_latency(); });
}

