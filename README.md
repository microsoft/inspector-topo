
topodetect: a PCIe topology detection tool
==========================================

This utility probes GPUs and RDMA NICs over PCIe to estimate the system interconnect topology. This is for use in sitations where virtualization obscures information that would normally be reported by topology query tools like lstopo.

The goal is to answer these questions:
 * which socket is each RDMA NIC connected to?
 * which socket is each GPU connected to?
 * which NICs and GPUs share a PCIe switch, and thus share bandwidth to the CPU?
 
Requirements
------------

The tool currently depends on these libraries:
* ibverbs
* numa
* hugetlbfs
* gflags
