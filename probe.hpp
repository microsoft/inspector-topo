// Copyright (c) Microsoft Corporation.
// All rights reserved.

#include "Endpoint.hpp"
#include "Loopback.hpp"

double probe_latency();
double probe_latency_from_core(int16_t core);
double probe_latency_from_numa_node(int node);

