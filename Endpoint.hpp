// Copyright (c) Microsoft Corporation.
// All rights reserved.

#ifndef __ENDPOINT__
#define __ENDPOINT__

#include <infiniband/verbs.h>
#include <cstring>
#include <iostream>
#include <unordered_set>
#include <endian.h>
#include <gflags/gflags.h>

class Loopback;

class Endpoint {
private:
  /// list of Verbs-capable devices
  ibv_device ** devices;
  int num_devices;

  /// info about chosen device
  ibv_device * device;
  const char * device_name;
  uint64_t device_guid; // big-endian
  ibv_device_attr device_attributes;
  
  /// info about chosen port
  uint8_t port; // port is generally 1-indexed
  ibv_port_attr port_attributes;
    
  /// GID of port
  uint8_t gid_index; // 0: RoCEv1 with MAC-based GID, 1:RoCEv2 with MAC-based GID, 2: RoCEv1 with IP-based GID, 3: RoCEv2 with IP-based GID
  ibv_gid gid;

  /// device context, used for most Verbs operations
  ibv_context * context;

  /// protection domain to go with context
  ibv_pd * protection_domain;

  /// until we can use NIC timestamps, store the CPU timestamp counter tick rate
  uint64_t ticks_per_sec;

  /// remember memory regions to free on exit
  std::unordered_set<ibv_mr *> memory_regions;

  /// Provide access to Loopback class for queue pair setup
  friend class Loopback;
  
public:
  Endpoint();
  ~Endpoint();

  /// register an already-allocated memory region.
  ibv_mr * register_region(void * address, size_t length);
  
  /// deregister a memory region.
  void deregister_region(ibv_mr * mr);
  
  /// allocate and register a memory region. Use huge pages if
  /// requested. Set requested_address to non-null to allocate at a
  /// specific virtual address. Returns null if the allocation fails.
  ibv_mr * allocate(size_t length,
                    bool use_hugepages = false,
                    void * requested_address = nullptr);

  // /// allocate and register a memory region on a particular NUMA
  // /// domain. Return null if the allocation is not possible.
  // ibv_mr * allocate_on_numa_domain(void * requested_address, size_t length);

  /// allocate and register a memory region from a particular
  /// core. Return null if the allocation is not possible.
  ibv_mr * allocate_from_core(int16_t core, size_t length);

  /// free a memory region allocated by one of the above calls.
  void free(ibv_mr * mr);

  /// accessor for ticks_per_sec
  inline uint64_t get_ticks_per_sec() const { return ticks_per_sec; }
};


#endif //  __ENDPOINT__
