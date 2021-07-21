// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "Options.hpp"

#include <cstdlib>
#include <getopt.h>
#include <iostream>

/// Actual storage for global options pointer
/// Initialize with defaults
std::unique_ptr<Options> Options::options = std::make_unique<Options>();

void Options::print_help_and_exit() {
  std::cout << "inspector-topo options:\n"
            << "  --device/-d (Name of Verbs device) type: string default: \"mlx5_0\"\n"
            << "  --device_port/-p (Port on Verbs device (usually 1-indexed, so should usually be\n"
            << "     1)) type: int32 default: 1\n"
            << "  --gid_index/-g (Verbs device GID index. 0: RoCEv1 with MAC-based GID, 1: RoCEv2\n"
            << "     with MAC-based GID, 2: RoCEv1 with IP-based GID, 3: RoCEv2 with IP-based\n"
            << "     GIDPort on Verbs device) type: int32 default: 3\n"
            << "  --concurrent_reads/-r (Number of concurrent reads for loopback flow bandwidth\n"
            << "     probe.) type: int32 default: 4\n"
            << "  --bw_iters/-b (Number of iterations to run when measuring GPU bandwidth.)\n"
            << "     type: int32 default: 10\n"
            << "  --bw_warmup_iters/-t (Number of warmup iterations to run when measuring GPU\n"
            << "     bandwidth.) type: int32 default: 1\n"
            << "  --iters/-i (Number of timed iterations to run (defaults to 1000).) type: int32\n"
            << "     default: 1000\n"
            << "  --length/-s (Length of test buffers in bytes (defaults to 1 GB).) type: int64\n"
            << "     default: 1073741824\n"
            << "  --warmup/-w (Number of warmup iterations to run before timing (defults to\n"
            << "     100).) type: int32 default: 100\n"
            << "  --help/-h (Print this help.)"
            << std::endl;
  std::exit(1);
}


/// default constructor just initializes defaults
Options::Options()
  : device("mlx5_0")
  , device_port(1)
  , gid_index(3)
  , warmup(100)
  , iters(1000)
  , length(1LL << 30)
  , concurrent_reads(4)
  , bw_iters(10)
  , bw_warmup_iters(1) {
}


/// Parse options from command line
Options::Options(int * argc_p, char ** argv_p[])
  : device("mlx5_0")
  , device_port(1)
  , gid_index(3)
  , warmup(100)
  , iters(1000)
  , length(1LL << 30)
  , concurrent_reads(4)
  , bw_iters(10)
  , bw_warmup_iters(1) {
    
  const char* const short_opts = "d:p:g:r:b:t:i:l:w:h";

  const struct option long_opts[] = {
    {"device",           required_argument, nullptr, 'd'},
    {"device_port",      required_argument, nullptr, 'p'},
    {"gid_index",        required_argument, nullptr, 'g'},
    {"concurrent_reads", required_argument, nullptr, 'r'},
    {"bw_iters",         required_argument, nullptr, 'b'},
    {"bw_warmup_iters",  required_argument, nullptr, 't'},
    {"iters",            required_argument, nullptr, 'i'},
    {"length",           required_argument, nullptr, 'l'},
    {"warmup",           required_argument, nullptr, 'w'},
    {"help",             no_argument,       nullptr, 'h'},
    {nullptr,            no_argument,       nullptr, 0}
  };
    
  // parse options
  while (true) {
    const auto opt = getopt_long(*argc_p, *argv_p, short_opts, long_opts, nullptr);
      
    if (-1 == opt) {
      break;
    }
      
    switch (opt) {
    case 'd':
      device = std::string(optarg);
      break;
        
    case 'p':
      device_port = std::stoi(optarg);
      break;
        
    case 'g':
      gid_index = std::stoi(optarg);
      break;
        
    case 'r':
      concurrent_reads = std::stoi(optarg);
      break;
        
    case 'b':
      bw_iters = std::stoi(optarg);
      break;
        
    case 't':
      bw_warmup_iters = std::stoi(optarg);
      break;
        
    case 'i':
      iters = std::stoi(optarg);
      break;
        
    case 'l':
      length = std::stoll(optarg);
      break;
        
    case 'w':
      warmup = std::stoi(optarg);
      break;
        
    case 'h': // -h or --help
    case '?': // Unrecognized option
    default:
      print_help_and_exit();
      break;
    }
  }
}
