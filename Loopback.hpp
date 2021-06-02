// Copyright (c) Microsoft Corporation.
// All rights reserved.

#ifndef __LOOPBACK__
#define __LOOPBACK__

#include <infiniband/verbs.h>
#include <cstring>
#include <iostream>
#include <endian.h>
#include <gflags/gflags.h>
#include <vector>
#include <string>
#include <sstream>

#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>

#include <errno.h>
#include <unistd.h>

#include "Endpoint.hpp"

DECLARE_int32(packet_size);

class Loopback {
private:

  /// information about local NIC
  Endpoint & endpoint;

  /// constants for initializing queues
  static const int completion_queue_depth       = 256;  // need to handle concurrent send and receive completions on each queue
  static const int send_queue_depth             = 128;  // need to be able to post a send before we've processed the previous one's completion
  static const int receive_queue_depth          = 128;  // need to be able to receive immediate value notifications
  static const int scatter_gather_element_count = 1;    // how many SGE's do we allow per operation?
  static const int max_inline_data              = 0;   // how much inline data should we support?
  static const int max_dest_rd_atomic           = 1;    // how many outstanding reads/atomic ops are allowed? (remote end of qp, limited by card)
  static const int max_rd_atomic                = 1;    // how many outstanding reads/atomic ops are allowed? (local end of qp, limited by card)
  static const int min_rnr_timer                = 0x12; // from Mellanox RDMA-Aware Programming manual, for RC only; probably don't need to touch
  static const int timeout                      = 14;   // from Mellanox RDMA-Aware Programming manual, for RC only; probably don't need to touch
  static const int retry_count                  = 7;    // from Mellanox RDMA-Aware Programming manual, for RC only; probably don't need to touch
  static const int rnr_retry                    = 7;    // from Mellanox RDMA-Aware Programming manual, for RC only; probably don't need to touch

  // functions to connect queue pairs
  void initialize_queue_pairs();
  void connect_queue_pairs();
  ibv_cq * create_completion_queue();
  ibv_qp * create_queue_pair(ibv_cq * completion_queue);

  void move_to_init(ibv_qp *);
  void move_to_rtr(ibv_qp *);
  void move_to_rts(ibv_qp *);
  
public:
  Loopback(Endpoint & e)
    : endpoint(e)
    , completion_queue(nullptr)
    , queue_pair(nullptr)
  {
    // do this later to support server overload
    //set_up_queue_pairs();
  }

  /// completion queue
  ibv_cq * completion_queue;

  /// loopback QP
  ibv_qp * queue_pair;
  
  void connect() {
    initialize_queue_pairs();
    connect_queue_pairs();
  }

  static void post_recv(ibv_qp * qp, ibv_recv_wr * wr) {
    ibv_recv_wr * bad_wr = nullptr;
    
    int retval = ibv_post_recv(qp, wr, &bad_wr);
    if (retval < 0) {
      std::cerr << "Error " << retval << " posting receive WR startgin at WR " << wr << " id " << (void*) wr->wr_id << std::endl;
      perror( "Error posting receive WR" );
      throw;
      exit(1);
    }
    
    if (bad_wr) {
      std::cerr << "Error posting receive WR at WR " << bad_wr << " id " << (void*) bad_wr->wr_id << std::endl;
      throw;
      exit(1);
    }
  }

  static void post_send(ibv_qp * qp, ibv_send_wr * wr) {
    ibv_send_wr * bad_wr = nullptr;

    int retval = ibv_post_send(qp, wr, &bad_wr);
    if (retval < 0) {
      std::cerr << "Error " << retval
                << " posting send WR starting at WR " << wr
                << " id " << (void*) wr->wr_id
                << ": " << strerror(errno)
                << std::endl;
      throw;
      exit(1);
    }
    
    if (bad_wr) {
      std::cerr << "Hmm. Error posting send WR at WR " << bad_wr
                << " id " << (void*) bad_wr->wr_id
                << " starting at WR " << wr
                << std::endl;
      throw;
      exit(1);
    }
  }

  static int poll_cq(ibv_cq * cq, int max_entries, ibv_wc completions[]) {
    int retval = ibv_poll_cq(cq, max_entries, &completions[0]);

    if (retval < 0) {
      std::cerr << "Failed polling completion queue with status " << retval << std::endl;
      exit(1);
    }

    return retval;
  }

};

#endif //  __LOOPBACK__
