// Copyright (c) Microsoft Corporation.
// All rights reserved.

#ifndef __PROBE_LATENCY__
#define __PROBE_LATENCY__

#include "Endpoint.hpp"
#include "Loopback.hpp"

/// Does 1-byte loopback RDMA READs to probe latency
double probe_latency();
double probe_latency_from_core(int16_t core);
double probe_latency_from_numa_node(int node);


#endif // __PROBE_LATENCY__
