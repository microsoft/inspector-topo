// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef __OPTIONS__
#define __OPTIONS__

#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <memory>

class Options {
private:
  void print_help_and_exit();

public:
  //Options(int * argc_p, char ** argv_p[]);
  Options(int * argc_p, char ** argv_p[]);
  Options();

  // endpoint-related options
  std::string device;
  int device_port;
  int gid_index;

  // loopback related options
  int concurrent_reads;

  // GPU bandwidth related options
  int bw_iters;
  int bw_warmup_iters;
  
  // top-level options
  int iters;
  int length;
  int warmup;

  /// Global options pointer
  static std::unique_ptr<Options> options;
};


#endif //  __OPTIONS__
