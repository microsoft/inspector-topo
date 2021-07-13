// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef __LOOPBACK_FLOW__
#define __LOOPBACK_FLOW__

#include <thread>
#include <atomic>

class LoopbackFlow {
private:
  /// flag to indicate we should stop
  std::atomic<bool> done_flag;

  /// thread that actually does the work
  std::thread thread;

  /// body of thread
  void thread_body(int numa_node);
  
public:
  /// starts loopback thread as soon as constructed. Defaults to run
  /// on any NUMA node, but a specific one can be requested.
  LoopbackFlow(int numa_node = -1)
    : done_flag(false)
    , thread([this, numa_node] { this->thread_body(numa_node); })
  { }

  ~LoopbackFlow();
  
  void stop();
};

#endif // __LOOPBACK_FLOW__
