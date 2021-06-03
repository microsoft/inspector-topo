
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
* ibverbs (but is likely already installed)
* numa
* hugetlbfs
* gflags

On Ubuntu, you can install them like this:
```
sudo apt install libnuma-dev libhugetlbfs-dev libgflags-dev
```

Building
--------

Just run ```make```.

Running
-------

GPU IDs are not yet handled in a smart way. You MUST set ```CUDA_VISIBLE_DEVICES``` before running to get results that make sense. For now on the 8-GPU ND40v2 nodes, use a command like this:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./topodetect
```

For the GPU bandwidth tests, the tool defaults to copying a 1 GB buffer 10 times. This is probably fine, but you can change the buffer size with the ```--length``` flag and the GPU bandwidth iteration count with the ```--bw_iters``` flag if you want.



