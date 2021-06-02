// Copyright (c) Microsoft Corporation.
// All rights reserved.

#include "Endpoint.hpp"
#include <sys/mman.h>

#include <x86intrin.h>
#include <thread>
#include <unistd.h>

extern "C" {
#include <hugetlbfs.h>
}

DEFINE_string(device, "mlx5_0", "Name of Verbs device");
DEFINE_int32(device_port, 1, "Port on Verbs device (usually 1-indexed, so should usually be 1)");
DEFINE_int32(gid_index, 3, "Verbs device GID index. 0: RoCEv1 with MAC-based GID, 1: RoCEv2 with MAC-based GID, 2: RoCEv1 with IP-based GID, 3: RoCEv2 with IP-based GIDPort on Verbs device");

Endpoint::Endpoint()
  : devices(nullptr)
  , num_devices(0)
  , device(nullptr)
  , device_name(nullptr)
  , device_guid(0)
  , device_attributes() // clear later
  , port(FLAGS_device_port) // port is generally 1-indexed
  , port_attributes() // clear later
  , gid_index(FLAGS_gid_index) // use RoCEv2 with IP-based GID
  , gid({.global = {0, 0}})
  , context(nullptr)
  , protection_domain(nullptr)
  , ticks_per_sec(0)
  , memory_regions()
{
  std::memset(&device_attributes, 0, sizeof(ibv_device_attr));
  std::memset(&port_attributes, 0, sizeof(ibv_port_attr));


  // get device list
  devices = ibv_get_device_list(&num_devices);
  if (!devices)  {
    std::cerr << "Didn't find any Verbs-capable devices!";
    exit(1);
  }

  // search for device
  for(int i = 0; i < num_devices; ++i) {
#ifdef DEBUG_LOG
      std::cout << "Found Verbs device " << ibv_get_device_name(devices[i]) 
                << " with guid " << (void*) be64toh(ibv_get_device_guid(devices[i])) 
                << std::endl;
#endif
    if ((num_devices == 1) || (FLAGS_device == ibv_get_device_name(devices[i])))  {
      // choose this device
      device = devices[i];
      device_name = ibv_get_device_name(device);
      device_guid = be64toh(ibv_get_device_guid(device) );
    }
  }
  
  // ensure we found a device
  if (!device)  {
    std::cerr << "Didn't find device " << FLAGS_device << "\n";
    exit(1);
  } 
#ifdef DEBUG_LOG
  else {
      std::cout << "Chose Verbs device " << ibv_get_device_name(device) << " gid index " << (int) gid_index << "\n";
  }
#endif


  // open device context and get device attributes
  context = ibv_open_device(device);
  if (!context)  {
    std::cerr << "Failed to get context for device " << device_name << "\n";
    exit(1);
  }
  int retval = ibv_query_device(context, &device_attributes);
  if (retval < 0)  {
    perror("Error getting device attributes");
    exit(1);
  }

  // choose a port on the device and get port attributes
#ifdef DEBUG_LOG
  if (device_attributes.phys_port_cnt > 1)  {
      std::cout << (int) device_attributes.phys_port_cnt << " ports detected; using port " << (int) FLAGS_device_port << std::endl;
  }
#endif
  if (device_attributes.phys_port_cnt < FLAGS_device_port)  {
    std::cerr << "expected " << (int) FLAGS_device_port << " ports, but found " << (int) device_attributes.phys_port_cnt;
    exit(1);
  }
  port = FLAGS_device_port;
  retval = ibv_query_port(context, port, &port_attributes);
  if (retval < 0)  {
    perror("Error getting port attributes");
    exit(1);
  }

  // print GIDs
  for (int i = 0; i < port_attributes.gid_tbl_len; ++i) {
    retval = ibv_query_gid(context, port, i, &gid);
    if (retval < 0)  {
      perror("Error getting GID");
      exit(1);
    }
#ifdef DEBUG_LOG
    if (gid.global.subnet_prefix != 0 || gid.global.interface_id !=0) {
        std::cout << "GID " << i << " is "
                  << (void*) gid.global.subnet_prefix << " " << (void*) gid.global.interface_id
                  << "\n";
    }
#endif
  }

  // get selected gid
  retval = ibv_query_gid(context, port, FLAGS_gid_index, &gid);
  if (retval < 0)  {
    perror("Error getting GID");
    exit(1);
  }
  if (0 == gid.global.subnet_prefix && 0 == gid.global.interface_id) {
    std::cerr << "Selected GID " << gid_index << " was all zeros; is interface down? Maybe try RoCEv1 GID index?" << std::endl;
    exit(1);
  }

  // create protection domain
  protection_domain = ibv_alloc_pd(context);
  if (!protection_domain)  {
    std::cerr << "Error getting protection domain!\n";
    exit(1);
  }

  /// until we can use NIC timestamps, store the CPU timestamp counter tick rate
  uint64_t start_ticks = __rdtsc();
  std::this_thread::sleep_for(std::chrono::seconds(1));
  uint64_t end_ticks = __rdtsc();
  ticks_per_sec = end_ticks - start_ticks;
#ifdef DEBUG_LOG
    std::cout << "Timestamp counter appears to count at " << ticks_per_sec << " ticks per second." << std::endl;
#endif
}

Endpoint::~Endpoint() {
  // free any memory regions that haven't been freed yet
  while (!memory_regions.empty()) {
    free(*memory_regions.begin());
  }

  if (protection_domain)  {
    int retval = ibv_dealloc_pd(protection_domain);
    if (retval < 0)  {
      perror("Error deallocating protection domain");
      exit(1);
    }
    protection_domain = nullptr;
  }

  if (context)  {
    int retval = ibv_close_device(context);
    if (retval < 0)  {
      perror("Error closing device context");
      exit(1);
    }
    context = nullptr;
  }

  if (devices)  {
    ibv_free_device_list(devices);
    devices = nullptr;
  }

  if (device)  {
    device = nullptr;
  }

}

ibv_mr * Endpoint::register_region(void * address, size_t length) {
  // register
  ibv_mr * mr = ibv_reg_mr(protection_domain, address, length,
                           (IBV_ACCESS_LOCAL_WRITE |
                            IBV_ACCESS_REMOTE_READ |
                            IBV_ACCESS_REMOTE_WRITE));
  if (!mr) {
      perror("Error registering memory region");
      exit(1);
  }

  // return memory region
  return mr;
}

void Endpoint::deregister_region(ibv_mr * mr) {
  int retval = ibv_dereg_mr(mr);
  if (retval < 0)  {
      perror("Error deregistering memory region");
      exit(1);
  }
}

ibv_mr * Endpoint::allocate(size_t length,
                            bool use_hugepages,
                            void * requested_address) {
  // round up to default huge page size
  size_t hugepagesize = gethugepagesize();
  if (hugepagesize < 0) {
    std::cerr << "Error getting default huge page size" << std::endl;
    exit(1);
  }
  length = (length + (hugepagesize-1)) & ~(hugepagesize-1);

  // do allocation
  void * buf = mmap(requested_address, length,
                    PROT_READ | PROT_WRITE,
                    (MAP_PRIVATE | MAP_ANONYMOUS |
                     (use_hugepages ? MAP_HUGETLB : 0) |
                     (requested_address != nullptr ? MAP_FIXED : 0)),
                    -1, 0);
  if (MAP_FAILED == buf) {
      perror("Error allocating memory region");
      exit(1);
  } else if (requested_address != nullptr && requested_address != buf) {
      perror("Error allocating memory region at requested address");
      exit(1);
  }

#ifdef DEBUG_LOG
    std::cout << "Buffer " << buf << " length " << length << " allocated." << std::endl;
#endif
  
  // register
  ibv_mr * mr = register_region(buf, length);

  // add memory region to list to be deallocated at teardown
  memory_regions.emplace(mr);
  
  // return newly-allocated memory region
  return mr;
}

ibv_mr * Endpoint::allocate_from_core(int16_t core, size_t length) {
  // capture current CPU affinity mask
  cpu_set_t previous_set;
  CPU_ZERO(&previous_set);
  int retval = sched_getaffinity(getpid(), sizeof(previous_set), &previous_set);
  if (retval < 0) {
    std::cerr << "Error getting current CPU affinity" << std::endl;
    exit(1);
  }

  // set affinity mask for allocation
  cpu_set_t new_set;
  CPU_ZERO(&new_set);
  CPU_SET(core, &new_set);

  // allocate
  ibv_mr * mr = allocate(length);
    
  // restore previous CPU affinity
  retval = sched_setaffinity(getpid(), sizeof(previous_set), &previous_set);
  if (retval < 0) {
    std::cerr << "Error getting current CPU affinity" << std::endl;
    exit(1);
  }

  // return newly-allocated memory region
  return mr;
}

void Endpoint::free(ibv_mr * mr) {
  // extract pointer from MR
  auto buf = mr->addr;
  auto len = mr->length;
  
  // deregister MR
  deregister_region(mr);
  
  // free MR
  int retval = munmap(buf, len);
  if (retval < 0)  {
      perror("Error freeing memory region");
      exit(1);
  }

  // remove from list
  memory_regions.erase(mr);

#ifdef DEBUG_LOG
    std::cout << "Buffer " << buf << " length " << len << " freed." << std::endl;
#endif
}
