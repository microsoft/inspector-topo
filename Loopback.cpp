// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "Loopback.hpp"
#include <chrono>

void Loopback::initialize_queue_pairs() {
  // See section 3.5 of the RDMA Aware Networks Programming User
  // Manual for more details on queue pair bringup.

  // Create shared completion queue
#ifdef DEBUG_LOG
    std::cout << "Creating completion queue...\n";
#endif
  completion_queue = create_completion_queue();

#ifdef DEBUG_LOG
    std::cout << "Creating queue pair...\n";
#endif
  queue_pair = create_queue_pair(completion_queue);

  // Move queue pair to INIT. This generates a local queue pair number.
#ifdef DEBUG_LOG
    std::cout << "Moving queue pair to INIT...\n";
#endif
  move_to_init(queue_pair);

  // At this point we can post receive buffers. In theory, we
  // *should* do so before we move queues to RTR, but as long as we
  // have some other syncronization mechanism that will keep other
  // parties from sending before we're ready, it's okay not to.

  // Next, we must exchange queue pair information and finish
  // connecting the queue pairs.
}

void Loopback::connect_queue_pairs() {
  // See section 3.5 of the RDMA Aware Networks Programming User
  // Manual for more details on queue pair bringup.

  // Move queue pairs to RTR. After this, we're ready to receive.
#ifdef DEBUG_LOG
    std::cout << "Moving queue pair to RTR...\n";
#endif
  move_to_rtr(queue_pair);
  
  // Move queue pairs to RTS. After this, we're ready to send.
#ifdef DEBUG_LOG
    std::cout << "Moving queue pair to RTS...\n";
#endif
  move_to_rts(queue_pair);
}

// create shared completion queues, one per core.
ibv_cq * Loopback::create_completion_queue() {
  ibv_cq * completion_queue = ibv_create_cq(endpoint.context,
                                            completion_queue_depth,
                                            NULL,  // no user context
                                            NULL,  // no completion channel
                                            0);    // no completion channel vector
  if (!completion_queue) {
    std::cerr << "Error creating completion queue!\n";
    exit(1);
  }
  
  return completion_queue;
}

// first, create queue pair (starts in RESET state)
ibv_qp * Loopback::create_queue_pair(ibv_cq * completion_queue) {
  ibv_qp_init_attr init_attributes;
  std::memset(&init_attributes, 0, sizeof(ibv_qp_init_attr));
  
  // use shared completion queue
  init_attributes.send_cq = completion_queue;
  init_attributes.recv_cq = completion_queue;
  
  // use whatever type of queue pair we selected
  init_attributes.qp_type = IBV_QPT_RC;
      
  // only issue send completions if requested
  init_attributes.sq_sig_all = false;
      
  // set queue depths and WR parameters accoring to constants declared earlier
  init_attributes.cap.max_send_wr     = send_queue_depth;
  init_attributes.cap.max_recv_wr     = receive_queue_depth;
  init_attributes.cap.max_send_sge    = scatter_gather_element_count;
  init_attributes.cap.max_recv_sge    = scatter_gather_element_count;
  init_attributes.cap.max_inline_data = max_inline_data;
  
  // create queue pair
  ibv_qp * queue_pair = ibv_create_qp(endpoint.protection_domain, &init_attributes);
  if (!queue_pair) {
    std::cerr << "Error creating queue pair!\n";
    exit(1);
  }

#ifdef DEBUG_LOG
    std::cout << "Created queue pair " << queue_pair
              << " QPN 0x" << std::hex << queue_pair->qp_num << std::dec
              << ".\n";
#endif
  
  return queue_pair;
}

// then, move queue pair to INIT
void Loopback::move_to_init(ibv_qp * qp) {
  ibv_qp_attr attributes;
  std::memset(&attributes, 0, sizeof(attributes));
  attributes.qp_state = IBV_QPS_INIT;
  attributes.port_num = endpoint.port;
  attributes.pkey_index = 0;
  attributes.qp_access_flags = (IBV_ACCESS_LOCAL_WRITE |
                                IBV_ACCESS_REMOTE_READ |
                                IBV_ACCESS_REMOTE_WRITE);

#ifdef DEBUG_LOG
    std::cout << "Moving queue pair " << qp << " to INIT...\n";
#endif
  int retval = ibv_modify_qp(qp, &attributes,
                             IBV_QP_STATE |
                             IBV_QP_PORT |
                             IBV_QP_PKEY_INDEX |
                             IBV_QP_ACCESS_FLAGS);
    if (retval < 0) {
    perror("Error setting queue pair to INIT");
    exit(1);
  }
}

// now, move queue pair to Ready-To-Receive (RTR) state
void Loopback::move_to_rtr(ibv_qp * qp) {
  ibv_qp_attr attributes;
  std::memset(&attributes, 0, sizeof(attributes));
  attributes.qp_state           = IBV_QPS_RTR;
  attributes.dest_qp_num        = qp->qp_num;
  attributes.rq_psn             = qp->qp_num; // use QPN as initial PSN
  attributes.max_dest_rd_atomic = max_dest_rd_atomic;
  attributes.min_rnr_timer      = min_rnr_timer;
  
  // what packet size do we want?
  attributes.path_mtu = endpoint.port_attributes.active_mtu;
  
  attributes.ah_attr.is_global     = 1; 
  attributes.ah_attr.dlid          = endpoint.port_attributes.lid; // not really necessary since using RoCE, not IB, and is_global is set
  attributes.ah_attr.sl            = 0;
  attributes.ah_attr.src_path_bits = 0;
  attributes.ah_attr.port_num      = endpoint.port;
  
  attributes.ah_attr.grh.dgid                      = endpoint.gid;
  attributes.ah_attr.grh.sgid_index                = endpoint.gid_index;
  attributes.ah_attr.grh.flow_label                = 0;
  attributes.ah_attr.grh.hop_limit                 = 0xFF;
  attributes.ah_attr.grh.traffic_class             = 1;
  
  int retval = ibv_modify_qp(qp, &attributes,
                             IBV_QP_STATE |
                             IBV_QP_PATH_MTU |
                             IBV_QP_DEST_QPN |
                             IBV_QP_RQ_PSN |
                             IBV_QP_AV |
                             IBV_QP_MAX_DEST_RD_ATOMIC |
                             IBV_QP_MIN_RNR_TIMER);
  if (retval < 0) {
    perror("Error setting queue pair to RTR");
    exit(1);
  }
}

// finally, move queue to Ready-To-Send (RTS) state
void Loopback::move_to_rts(ibv_qp * qp) {
  ibv_qp_attr attributes;
  std::memset(&attributes, 0, sizeof(attributes));
  attributes.qp_state = IBV_QPS_RTS;
  attributes.sq_psn = qp->qp_num; // use QPN as initial PSN
  attributes.timeout = timeout;             // used only for RC
  attributes.retry_cnt = retry_count;       // used only for RC
  attributes.rnr_retry = rnr_retry;         // used only for RC
  attributes.max_rd_atomic = max_rd_atomic; // used only for RC
  int retval = ibv_modify_qp(qp, &attributes,
                             IBV_QP_STATE |
                             IBV_QP_SQ_PSN |
                             IBV_QP_TIMEOUT |
                             IBV_QP_RETRY_CNT |
                             IBV_QP_RNR_RETRY |
                             IBV_QP_MAX_QP_RD_ATOMIC);
  if (retval < 0) {
    perror("Error setting queue pair to RTR");
    exit(1);
  }
}
