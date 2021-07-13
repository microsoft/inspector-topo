// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "LoopbackFlow.hpp"
#include "Loopback.hpp"

#include <gflags/gflags.h>

#include <iostream>
#include <vector>

#include <numa.h>

DECLARE_int64(length);
DEFINE_int32(concurrent_reads, 4, "Number of concurrent reads for loopback flow bandwidth probe.");

void LoopbackFlow::thread_body(int numa_node) {
#ifdef DEBUG_LOG
  std::cout << "Loopback flow thread starting on NUMA node " << numa_node << std::endl;
#endif
  // set NUMA node to that requested
  numa_run_on_node(numa_node);

  // Initialize RDMA NIC
  Endpoint e;

  // Initialize loopback connection
  Loopback lp(e);
  
  // Allocate memory region. We copy to and from the same buffer,
  // since we care only about data movement, not the result.
  ibv_mr * mr = e.allocate(FLAGS_length);

  // reuse same SGE for all reads
  ibv_sge send_sge;
  send_sge.addr   = reinterpret_cast<uintptr_t>(mr->addr);
  send_sge.length = FLAGS_length;
  send_sge.lkey   = mr->lkey;
    
  const int concurrent_reads = FLAGS_concurrent_reads;
  ibv_send_wr send_wr[concurrent_reads];
  for (int i = 0; i < concurrent_reads; ++i) {
    send_wr[i].wr_id = i;
    send_wr[i].next = nullptr;
    send_wr[i].sg_list = &send_sge;
    send_wr[i].num_sge = 1;
    send_wr[i].opcode = IBV_WR_RDMA_READ;
    send_wr[i].send_flags = IBV_SEND_SIGNALED;
    send_wr[i].wr.rdma.remote_addr = reinterpret_cast<uintptr_t>(mr->addr);
    send_wr[i].wr.rdma.rkey = mr->rkey;
  }

  // post all outstanding requests
  int outstanding_reads = 0;
  for (int i = 0; i < concurrent_reads; ++i) {
    lp.post_send(&send_wr[i]);
    ++outstanding_reads;
#ifdef DEBUG_LOG
    std::cout << "Loopback flow posted read " << i << "." << std::endl;
#endif
  }
    
  // keep outstanding requests going
  bool done = false;
  while (!done) {
    // see if any reads have completed
    ibv_wc completions[concurrent_reads];
    int completed_count = lp.poll_cq(concurrent_reads, &completions[0]);

    // decrement outstanding count, whether reads completed successfully or not
    outstanding_reads -= completed_count;

    // check done flag here to avoid reposting unnecessary reads
    done = done_flag.load();
      
    // process completions
    for (int i = 0; i < completed_count; ++i) {
      if (completions[i].status != IBV_WC_SUCCESS) {
        std::cerr << "Got eror completion for " << (void*) completions[i].wr_id
                  << " with status " << ibv_wc_status_str(completions[i].status)
                  << std::endl;
        exit(1);
      } else {
#ifdef DEBUG_LOG
        std::cout << "Loopback flow read " << completions[i].wr_id << " completed; reposting..." << std::endl;
#endif
        // repost read that completed
        lp.post_send(&send_wr[completions[i].wr_id]);
        ++outstanding_reads;
      }
    }
  }

  // stop was requested; wait for current requests to complete
  while (outstanding_reads > 0) {
    ibv_wc completions[concurrent_reads];
    int completed_count = lp.poll_cq(concurrent_reads, &completions[0]);
    outstanding_reads -= completed_count;
#ifdef DEBUG_LOG
    for (int i = 0; i < completed_count; ++i) {
      std::cout << "Loopback flow read " << completions[i].wr_id << " completed." << std::endl;
    }
#endif
  }
    
#ifdef DEBUG_LOG
  std::cout << "Loopback flow thread exiting." << std::endl;
#endif
}

LoopbackFlow::~LoopbackFlow() {
  if (thread.joinable()) {
    stop();
  }
}
    
void LoopbackFlow::stop() {
#ifdef DEBUG_LOG
  std::cout << "Loopback flow stop requested." << std::endl;
#endif
  done_flag.store(true);
  thread.join();
}

  
