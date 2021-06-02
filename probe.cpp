// Copyright (c) Microsoft Corporation.
// All rights reserved.

#include "probe.hpp"
#include <iostream>

#include <gflags/gflags.h>

#include <iostream>
//#include <iomanip>
#include <chrono>
//#include <cmath>
#include <vector>
#include <algorithm>

#include <numa.h>

#include <x86intrin.h>


DECLARE_int32(warmup);
DECLARE_int32(iters);
DECLARE_int64(length);


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


double probe_latency() {
  // Initialize RDMA NIC
  Endpoint e;

  // Initialize loopback connection
  Loopback lp(e);
  lp.connect();
  
  // Allocate memory regions
  ibv_mr * source_mr = e.allocate(FLAGS_length);
  ibv_mr * dest_mr   = e.allocate(FLAGS_length);

  // ibv_sge recv_sge;
  // recv_sge.addr   = reinterpret_cast<uintptr_t>(dest_mr->addr);
  // recv_sge.length = FLAGS_length;
  // recv_sge.lkey   = dest_mr->lkey;

  // ibv_recv_wr recv_wr;
  // recv_wr.wr_id = 0;
  // recv_wr.next = nullptr;
  // recv_wr.sg_list = nullptr;
  // recv_wr.num_sge = 0;

  ibv_sge send_sge;
  send_sge.addr   = reinterpret_cast<uintptr_t>(dest_mr->addr);
  send_sge.length = 1; //FLAGS_length;
  send_sge.lkey   = dest_mr->lkey;
    
  ibv_send_wr send_wr;
  send_wr.wr_id = 0;
  send_wr.next = nullptr;
  send_wr.sg_list = &send_sge;
  send_wr.num_sge = 1;
  send_wr.opcode = IBV_WR_RDMA_READ;
  send_wr.send_flags = IBV_SEND_SIGNALED;
  send_wr.wr.rdma.remote_addr = reinterpret_cast<uintptr_t>(source_mr->addr);
  send_wr.wr.rdma.rkey = source_mr->rkey;
  send_wr.imm_data = 0;

  // record the time whenever we complete 
  std::vector<uint64_t> send_times;
  send_times.reserve(FLAGS_iters + 1); // preallocate space for each probe
  
  // Record start time and do probes
  send_times.push_back(__rdtsc()); // record initial time
  for (int i = 0; i < FLAGS_iters; ++i) {
    // post send to start sending
    send_wr.wr_id = i; // set WR ID to 
    Loopback::post_send(lp.queue_pair, &send_wr);

    // wait for completion
    ibv_wc completion;
    while (0 == Loopback::poll_cq(lp.completion_queue, 1, &completion)) {
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


