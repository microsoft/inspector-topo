
inspector-topo: a PCIe topology detection tool
==========================================

This utility probes GPUs and RDMA NICs over PCIe to estimate the system interconnect topology. This is for use in sitations where virtualization obscures information that would normally be reported by topology query tools like lstopo. This is true for some Azure GPU SKUs.

The goal is to answer these questions:
 * which socket is each RDMA NIC connected to?
 * which socket is each GPU connected to?
 * which NICs and GPUs share a PCIe switch, and thus share bandwidth to the CPU?

This is a work in progress!

Requirements
------------

The tool currently depends on these libraries:
* ibverbs (specifically libibverbs-dev on Ubuntu, but if it's not already installed you probably have IB/RDMA driver problems)
* numa (specifically libnuma-dev)

On Ubuntu, you can install them like this:
```
sudo apt install libnuma-dev
```

Building
--------

Just run ```make```.

Installing
----------

Run ```make install``` to place the inspector-topo binary in /usr/local/bin.

Running
-------

GPU IDs are not yet handled in a smart way. You MUST set ```CUDA_VISIBLE_DEVICES``` before running to get results that make sense. For now on the 8-GPU ND40v2 nodes, use a command like this:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./topodetect
```

For the GPU bandwidth tests, the tool defaults to copying a 1 GB buffer 10 times. This is probably fine, but you can change the buffer size with the ```--length``` flag and the GPU bandwidth iteration count with the ```--bw_iters``` flag if you want.



Contributing
------------

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.


Trademarks
----------

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
