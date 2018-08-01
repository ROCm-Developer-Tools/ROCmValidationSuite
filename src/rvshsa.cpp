/********************************************************************************
 *
 * Copyright (c) 2018 ROCm Developer Tools
 *
 * MIT LICENSE:
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do
 * so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include "rvshsa.h"

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "rvs_util.h"
#include "rvsloglp.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

// ptr to singletone instance
rvs::hsa* rvs::hsa::pDsc;

void rvs::hsa::Init() {
  if (pDsc == nullptr) {
    pDsc = new rvs::hsa();
    pDsc->InitAgents();
  }
}

void rvs::hsa::Terminate() {
  if (pDsc != nullptr) {
    delete pDsc;
    pDsc = nullptr;
  }
}

rvs::hsa* rvs::hsa::Get() {
  return pDsc;
}

//! Default constructor
rvs::hsa::hsa() {
}

//! Default destructor
rvs::hsa::~hsa() {
}

void rvs::hsa::print_hsa_status(const std::string& file,
                                int line,
                                const std::string& function,
                                hsa_status_t st) {
  string log_msg = "pqt   " + std::string(file)+ "  " + function + ":"
  + std::to_string(line);

  print_hsa_status(log_msg, st);
}

// TODO(dmatichdl) add info
void rvs::hsa::print_hsa_status(string message, hsa_status_t st) {
  string log_msg = message;
  // skip successfull messages
  if (st == HSA_STATUS_SUCCESS) {
    return;
  }
  switch (st) {
    case HSA_STATUS_SUCCESS : {
      log_msg = message + " The function has been executed successfully.";
      break;
    };
    case HSA_STATUS_INFO_BREAK : {
      log_msg = message + " A traversal over a list of elements has been " +
      "interrupted by the application before completing.";
      break;
    };
    case HSA_STATUS_ERROR : {
      log_msg = message + " A generic error has occurred.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_ARGUMENT : {
      log_msg = message + " One of the actual arguments does not meet a " +
      "precondition stated in the documentation of the corresponding formal " +
      "argument.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_QUEUE_CREATION : {
      log_msg = message + " The requested queue creation is not valid.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_ALLOCATION : {
      log_msg = message + " The requested allocation is not valid.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_AGENT : {
      log_msg = message + " The agent is invalid.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_REGION : {
      log_msg = message + " The memory region is invalid.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_SIGNAL : {
      log_msg = message + " The signal is invalid.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_QUEUE : {
      log_msg = message + " The queue is invalid.";
      break;
    };
    case HSA_STATUS_ERROR_OUT_OF_RESOURCES : {
      log_msg = message + " The HSA runtime failed to allocate the necessary " +
      "resources. This error may also occur when the HSA runtime needs to " +
      "spawn threads or create internal OS-specific events.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_PACKET_FORMAT : {
      log_msg = message + " The AQL packet is malformed.";
      break;
    };
    case HSA_STATUS_ERROR_RESOURCE_FREE : {
      log_msg = message +
      " An error has been detected while releasing a resource.";
      break;
    };
    case HSA_STATUS_ERROR_NOT_INITIALIZED : {
      log_msg = message +
      " An API other than ::hsa_init has been invoked while the reference " +
      "count of the HSA runtime is 0.";
      break;
    };
    case HSA_STATUS_ERROR_REFCOUNT_OVERFLOW : {
      log_msg = message + " The maximum reference count for the object has " +
      "been reached.";
      break;
    };
    case HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS : {
      log_msg = message + " The arguments passed to a functions are not " +
      "compatible.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_INDEX : {
      log_msg = message + " The index is invalid.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_ISA : {
      log_msg = message + " The instruction set architecture is invalid.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_ISA_NAME : {
      log_msg = message + " The instruction set architecture name is invalid.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_CODE_OBJECT : {
      log_msg = message + " The code object is invalid.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_EXECUTABLE : {
      log_msg = message + " The executable is invalid.";
      break;
    };
    case HSA_STATUS_ERROR_FROZEN_EXECUTABLE : {
      log_msg = message + " The executable is frozen.";
      break;
    };
    case HSA_STATUS_ERROR_INVALID_SYMBOL_NAME : {
      log_msg = message + " There is no symbol with the given name.";
      break;
    };
    case HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED : {
      log_msg = message + " The variable is already defined.";
      break;
    };
    case HSA_STATUS_ERROR_VARIABLE_UNDEFINED : {
      log_msg = message + " The variable is undefined.";
      break;
    };
    case HSA_STATUS_ERROR_EXCEPTION : {
      log_msg = message +
      " An HSAIL operation resulted on a hardware exception.";
      break;
    };
    default : {
      log_msg = message + " Unknown error.";
      break;
    }
  }
  rvs::lp::Log(log_msg, rvs::logdebug);
}


/**
 * @brief Fetch all hsa_agents
 *
 * Functionality:
 *
 * Fetch all CPUs and GPUs present in the system.
 *
 * @return void
 *
 * */
void rvs::hsa::InitAgents() {
  hsa_status_t status;
  string log_msg;

  // Initialize Roc Runtime
  rvs::lp::Log("[PQT] Before hsa_init ...", rvs::logdebug);
  status = hsa_init();
  rvs::lp::Log("[PQT] After hsa_init ...", rvs::logdebug);
  print_hsa_status("[PQT] InitAgents - hsa_init()", status);

  // Initialize profiling
  rvs::lp::Log("[PQT] Before hsa_amd_profiling_async_copy_enable ...",
               rvs::logdebug);
  status = hsa_amd_profiling_async_copy_enable(true);
  rvs::lp::Log("[PQT] Before hsa_amd_profiling_async_copy_enable ...",
               rvs::logdebug);
  print_hsa_status("[PQT] InitAgents - hsa_amd_profiling_async_copy_enable()",
                   status);

  // Populate the lists of agents
  rvs::lp::Log("[PQT] Before hsa_iterate_agents ...", rvs::logdebug);
  status = hsa_iterate_agents(ProcessAgent, &agent_list);
  rvs::lp::Log("[PQT] After hsa_iterate_agents ...", rvs::logdebug);
  print_hsa_status("[PQT] InitAgents - hsa_iterate_agents()", status);

  for (uint32_t i = 0; i < agent_list.size(); i++) {
    rvs::lp::Log("[PQT] ============================", rvs::logdebug);
    log_msg = "[PQT] InitAgents - agent with name = "  +
      agent_list[i].agent_name + " and device_type = " +
      agent_list[i].agent_device_type;
    rvs::lp::Log(log_msg.c_str(), rvs::logdebug);
    rvs::lp::Log("[PQT] ============================", rvs::logdebug);

    // Populate the list of memory pools
    rvs::lp::Log("[PQT] Before hsa_amd_agent_iterate_memory_pools ...",
                 rvs::logdebug);
    status = hsa_amd_agent_iterate_memory_pools(agent_list[i].agent,
                                                ProcessMemPool, &agent_list[i]);
    rvs::lp::Log("[PQT] After hsa_amd_agent_iterate_memory_pools ...",
                 rvs::logdebug);
    print_hsa_status("[PQT] InitAgents - hsa_amd_agent_iterate_memory_pools()",
                     status);

    // separate the lists
    if (agent_list[i].agent_device_type == "CPU") {
      cpu_list.push_back(agent_list[i]);
    } else if (agent_list[i].agent_device_type == "GPU") {
      gpu_list.push_back(agent_list[i]);
    }
  }

  // Initialize the list of buffer sizes to use in copy/read/write operations
  // For All Copy operations use only one buffer size
  if (size_list.size() == 0) {
    uint32_t size_len = sizeof(DEFAULT_SIZE_LIST)/sizeof(uint32_t);
    for (uint32_t idx = 0; idx < size_len; idx++) {
      size_list.push_back(DEFAULT_SIZE_LIST[idx]);
    }
  } else {
    uint32_t size_len = size_list.size();
    for (uint32_t idx = 0; idx < size_len; idx++) {
      size_list[idx] = size_list[idx] * 1024 * 1024;
    }
  }
  std::sort(size_list.begin(), size_list.end());
}

/**
 * @brief Process individual hsa_agent
 *
 * Functionality:
 *
 * Process a single CPUs or GPUs hsa_agent_t
 *
 * @return hsa_status_t
 *
 * */
hsa_status_t rvs::hsa::ProcessAgent(hsa_agent_t agent, void* data) {
  hsa_status_t status;
  char agent_name[64];
  hsa_device_type_t device_type;
  string log_msg, log_agent_name;
  uint32_t node;
  AgentInformation agent_info;

  // get agent list
  vector<AgentInformation>* agent_l =
  reinterpret_cast<vector<AgentInformation>*>(data);

  rvs::lp::Log("[PQT] Called ProcessAgent() ...", rvs::logdebug);

  // Get the name of the agent
  status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, agent_name);
  print_hsa_status("[PQT] HSA_AGENT_INFO_NAME", status);
  rvs::lp::Log(string("pqt agent_name: ") + agent_name, rvs::logdebug);

  // Get device type
  status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
  print_hsa_status("[PQT] HSA_AGENT_INFO_DEVICE", status);

  status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NODE, &node);
  print_hsa_status("[PQT] HSA_AGENT_INFO_NODE", status);
  agent_info.node = node;
  rvs::lp::Log("pqt node: " + std::to_string(node), rvs::logdebug);

  log_agent_name = agent_name;
  log_msg = "[PQT] Found agent with name = "  + log_agent_name +
    " and device_type = ";
  switch (device_type) {
    case HSA_DEVICE_TYPE_CPU : {
      agent_info.agent_device_type = "CPU";
      log_msg = log_msg + "CPU.";
      break;
    };
    case HSA_DEVICE_TYPE_GPU : {
      agent_info.agent_device_type = "GPU";
      log_msg = log_msg + "GPU.";
      break;
    };
    case HSA_DEVICE_TYPE_DSP : {
      agent_info.agent_device_type = "DSP";
      log_msg = log_msg + "DSP.";
      break;
    };
  }
  rvs::lp::Log(log_msg.c_str(), rvs::logdebug);
  // add agent to list
  agent_info.agent = agent;
  agent_info.agent_name = log_agent_name;
  agent_l->push_back(agent_info);

  return HSA_STATUS_SUCCESS;
}

/**
 * @brief Process hsa_agent memory pool
 *
 * Functionality:
 *
 * Process agents memory pools
 *
 * @return hsa_status_t
 *
 * */
hsa_status_t rvs::hsa::ProcessMemPool(hsa_amd_memory_pool_t pool, void* data) {
  hsa_status_t status;

  // get current agents memory pools
  AgentInformation* agent_info = reinterpret_cast<AgentInformation*>(data);

  // Query pools' segment, report only pools from global segment
  hsa_amd_segment_t segment;
  status = hsa_amd_memory_pool_get_info(pool,
                                        HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                                        &segment);
  RVSDEBUG_
  print_hsa_status(__FILE__, __LINE__, __func__, status);
  if (HSA_AMD_SEGMENT_GLOBAL != segment) {
    return HSA_STATUS_SUCCESS;
  }

  // Determine if allocation is allowed in this pool
  // Report only pools that allow an alloction by user
  bool alloc = false;
  status = hsa_amd_memory_pool_get_info(pool,
                            HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
                            &alloc);
  print_hsa_status(__FILE__, __LINE__, __func__, status);
  if (alloc != true) {
    return HSA_STATUS_SUCCESS;
  }

  // Query the max allocatable size
  size_t max_size = 0;
  status = hsa_amd_memory_pool_get_info(pool,
                                        HSA_AMD_MEMORY_POOL_INFO_SIZE,
                                        &max_size);
  print_hsa_status(__FILE__, __LINE__, __func__, status);
  agent_info->max_size_list.push_back(max_size);

  // Determine if the pools is accessible to all agents
  bool access_to_all = false;
  status = hsa_amd_memory_pool_get_info(pool,
                                HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL,
                                &access_to_all);
  print_hsa_status(__FILE__, __LINE__, __func__, status);

  // Determine type of access to owner agent
  hsa_amd_memory_pool_access_t owner_access;
  hsa_agent_t agent = agent_info->agent;
  status = hsa_amd_agent_memory_pool_get_info(agent, pool,
                                      HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS,
                                      &owner_access);
  print_hsa_status(__FILE__, __LINE__, __func__, status);

  // Determine if the pool is fine-grained or coarse-grained
  uint32_t flag = 0;
  status = hsa_amd_memory_pool_get_info(pool,
                                        HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
                                        &flag);
  print_hsa_status(__FILE__, __LINE__, __func__, status);
  bool is_kernarg = (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT & flag);

  // Update the pool handle for system memory if kernarg is true
  rvs::lp::Log("[PQT] ****************************************", rvs::logdebug);
  if (is_kernarg) {
    agent_info->sys_pool = pool;
    rvs::lp::Log("[PQT] Found system memory region", rvs::logdebug);
  } else if (owner_access != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
    agent_info->mem_pool_list.push_back(pool);
    rvs::lp::Log("[PQT] Found regular memory region", rvs::logdebug);
  }
  rvs::lp::Log("[PQT] ****************************************", rvs::logdebug);

  return HSA_STATUS_SUCCESS;
}

const int rvs::hsa::FindAgent(const uint32_t Node) {
  for (size_t i = 0; i < agent_list.size(); i++) {
    if (agent_list[i].node == Node)
      return i;
  }
  return -1;
}

// TODO(dmatichdl) info
double rvs::hsa::GetCopyTime(bool bidirectional,
                             hsa_signal_t signal_fwd, hsa_signal_t signal_rev) {
  hsa_status_t status;
  // Obtain time taken for forward copy
  hsa_amd_profiling_async_copy_time_t async_time_fwd = {0};
  status = hsa_amd_profiling_get_async_copy_time(signal_fwd, &async_time_fwd);
  print_hsa_status(
    "[PQT] GetCopyTime - hsa_amd_profiling_get_async_copy_time(forward)",
                   status);
  if (bidirectional == false) {
    RVSDEBUG_
    return(async_time_fwd.end - async_time_fwd.start);
  }

  hsa_amd_profiling_async_copy_time_t async_time_rev = {0};
  status = hsa_amd_profiling_get_async_copy_time(signal_rev, &async_time_rev);
  print_hsa_status(
    "[PQT] GetCopyTime - hsa_amd_profiling_get_async_copy_time(backward)",
                   status);
  double start = std::min(async_time_fwd.start, async_time_rev.start);
  double end = std::max(async_time_fwd.end, async_time_rev.end);
  RVSDEBUG_
  return(end - start);
}

int rvs::hsa::Allocate(int SrcAgent, int DstAgent, size_t Size,
                     hsa_amd_memory_pool_t* pSrcPool, void** SrcBuff,
                     hsa_amd_memory_pool_t* pDstPool, void** DstBuff) {
  hsa_status_t status;
  void* srcbuff = nullptr;
  void* dstbuff = nullptr;

  // iterate over src pools
  for (size_t i = 0; i < agent_list[SrcAgent].mem_pool_list.size(); i++) {
    // size too small, continue
    if (Size > agent_list[SrcAgent].max_size_list[i])
      continue;

    // try allocating source buffer
    status = hsa_amd_memory_pool_allocate(
      agent_list[SrcAgent].mem_pool_list[i], Size, 0, &srcbuff);
    if (status != HSA_STATUS_SUCCESS)
      continue;


    // iterate over dst pools
    for (size_t j = 0; j < agent_list[DstAgent].mem_pool_list.size(); j++) {
      // size too small, continue
      if (Size > agent_list[DstAgent].max_size_list[j])
        continue;

      // check if src agent has access to this dst agent's pool
      hsa_amd_memory_pool_access_t access =
        HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
      status = hsa_amd_agent_memory_pool_get_info(
        agent_list[SrcAgent].agent,
        agent_list[DstAgent].mem_pool_list[j],
        HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS,
        &access);

      if (access == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED)
        continue;

      // try allocating destination buffer
      status = hsa_amd_memory_pool_allocate(
        agent_list[DstAgent].mem_pool_list[j], Size, 0, &dstbuff);
      if (status != HSA_STATUS_SUCCESS)
        continue;

      // grant src agen access to destination buffer
      status = hsa_amd_agents_allow_access(1,
                                           &agent_list[SrcAgent].agent,
                                           NULL,
                                           dstbuff);
      if (status != HSA_STATUS_SUCCESS) {
        // cannot grant access, release buffer and skip to the next pool
        hsa_amd_memory_pool_free(dstbuff);
        dstbuff = nullptr;
        continue;
      }

      // all OK, set output parameters:
      *pSrcPool = agent_list[SrcAgent].mem_pool_list[i];
      *pDstPool = agent_list[DstAgent].mem_pool_list[j];
      *SrcBuff = srcbuff;
      *DstBuff = dstbuff;

      return 0;
    }

    // suitable destination buffer not foud, deallocate src buff and exit
    hsa_amd_memory_pool_free(srcbuff);

    RVSDEBUG_
    return -1;
  }

  RVSDEBUG_
  return -1;
}

  // TODO(mlucinhdl) add info
int rvs::hsa::SendTraffic(uint32_t SrcNode, uint32_t DstNode,
                              size_t Size, bool bidirectional,
                              double* Duration) {
  hsa_status_t status;
  int sts;

  int32_t src_ix_fwd;
  int32_t dst_ix_fwd;
  hsa_amd_memory_pool_t src_pool_fwd;
  hsa_amd_memory_pool_t dst_pool_fwd;
  void* src_ptr_fwd = nullptr;
  void* dst_ptr_fwd = nullptr;
  hsa_signal_t signal_fwd;

  int32_t src_ix_rev;
  int32_t dst_ix_rev;
  hsa_amd_memory_pool_t src_pool_rev;
  hsa_amd_memory_pool_t dst_pool_rev;
  void* src_ptr_rev = nullptr;
  void* dst_ptr_rev = nullptr;
  hsa_signal_t signal_rev;

  RVSDEBUG_

  // given NUMA nodes, find agent indexes
  src_ix_fwd = FindAgent(SrcNode);
  dst_ix_fwd = FindAgent(DstNode);
  if (src_ix_fwd < 0 || dst_ix_fwd < 0) {
    RVSDEBUG_
    return -1;
  }

  // allocate buffers and grant permissions for forward transfer
  sts = Allocate(src_ix_fwd, dst_ix_fwd, Size,
           &src_pool_fwd, &src_ptr_fwd,
           &dst_pool_fwd, &dst_ptr_fwd);
  if (sts) {
    RVSDEBUG_
    return -1;
  }

  // Create a signal to wait on copy operation
  status = hsa_signal_create(1, 0, NULL, &signal_fwd);
  if (status != HSA_STATUS_SUCCESS) {
      hsa_amd_memory_pool_free(src_ptr_fwd);
      hsa_amd_memory_pool_free(dst_ptr_fwd);
      RVSDEBUG_
      return -1;
  }

  if (bidirectional) {
    src_ix_rev = dst_ix_fwd;
    dst_ix_rev = src_ix_fwd;

    // allocate buffers and grant permissions for reverse transfer
    sts = Allocate(src_ix_rev, dst_ix_rev, Size,
            &src_pool_rev, &src_ptr_rev,
            &dst_pool_rev, &dst_ptr_rev);

    if (sts) {
      hsa_amd_memory_pool_free(src_ptr_fwd);
      hsa_amd_memory_pool_free(dst_ptr_fwd);
      RVSDEBUG_
      return -1;
    }

    // Create a signal to wait on for reverse copy operation
    status = hsa_signal_create(1, 0, NULL, &signal_rev);
    if (status != HSA_STATUS_SUCCESS) {
      hsa_amd_memory_pool_free(src_ptr_fwd);
      hsa_amd_memory_pool_free(dst_ptr_fwd);
      hsa_amd_memory_pool_free(src_ptr_rev);
      hsa_amd_memory_pool_free(dst_ptr_rev);
      hsa_signal_destroy(signal_fwd);
      RVSDEBUG_
      return -1;
    }
  }

  // initiate forward transfer
  hsa_signal_store_relaxed(signal_fwd, 1);
  status = hsa_amd_memory_async_copy(dst_ptr_fwd, agent_list[dst_ix_fwd].agent,
                                     src_ptr_fwd, agent_list[src_ix_fwd].agent,
                                     Size,
                                     0, NULL, signal_fwd);
  if (bidirectional) {
    RVSDEBUG_
    // initiate reverse transfer
    hsa_signal_store_relaxed(signal_rev, 1);
    status = hsa_amd_memory_async_copy(
        dst_ptr_rev, agent_list[dst_ix_rev].agent,
        src_ptr_rev, agent_list[src_ix_rev].agent, Size,
        0, NULL, signal_rev);
  }

  // wait for transfer to complete
  while (hsa_signal_wait_acquire(signal_fwd, HSA_SIGNAL_CONDITION_LT,
    1, uint64_t(-1), HSA_WAIT_STATE_ACTIVE)) {}

  // if bidirectional, also wait for reverse transfer to complete
  if (bidirectional == true) {
    while (hsa_signal_wait_acquire(signal_rev, HSA_SIGNAL_CONDITION_LT,
    1, uint64_t(-1), HSA_WAIT_STATE_ACTIVE)) {}
    RVSDEBUG_
  }

  // get transfer duration
  *Duration = GetCopyTime(bidirectional, signal_fwd, signal_rev)/1000000000;

  hsa_amd_memory_pool_free(src_ptr_fwd);
  hsa_amd_memory_pool_free(dst_ptr_fwd);
  hsa_signal_destroy(signal_fwd);

  if (bidirectional) {
    hsa_amd_memory_pool_free(src_ptr_rev);
    hsa_amd_memory_pool_free(dst_ptr_rev);
    hsa_signal_destroy(signal_rev);
  }

  return 0;
}

// TODO(dmatichdl) info
/**
 * @brief Process hsa_agent memory pool
 *
 * Functionality:
 *
 * Process agents memory pools
 *
 * @return hsa_status_t
 *
 * */
/*
double rvs::hsa::send_traffic(hsa_agent_t src_agent, hsa_agent_t dst_agent,
                              hsa_amd_memory_pool_t src_buff,
                              hsa_amd_memory_pool_t dst_buff,
                              bool bidirectional, size_t curr_size) {
  hsa_status_t status;
  void* src_pool_pointer_fwd;
  void* dst_pool_pointer_fwd;
  void* src_pool_pointer_rev;
  void* dst_pool_pointer_rev;
  string log_msg;
  hsa_signal_t signal_fwd, signal_rev;
  char s_buff[256];
  uint64_t total_size = 0;
  double curr_time;
  double bandwidth;

  rvs::lp::Log("[PQT] ++++++++++++++++++++++++++++++++++++++", rvs::logdebug);
  rvs::lp::Log("[PQT] send_traffic called ... ", rvs::logdebug);
  rvs::lp::Log("[PQT] ++++++++++++++++++++++++++++++++++++++", rvs::logdebug);

  snprintf(s_buff, sizeof(s_buff), "%lX", src_agent.handle);
  rvs::lp::Log(std::string("src_agent = ") + s_buff, rvs::logdebug);

  snprintf(s_buff, sizeof(s_buff), "%lX", dst_agent.handle);
  rvs::lp::Log(std::string("dst_agent = ") + s_buff, rvs::logdebug);

  // print current size
  rvs::lp::Log("[PQT] ---------------------------------------", rvs::logdebug);
  log_msg = "[PQT] send_traffic - curr_size = " +
    std::to_string(curr_size) + " Bytes";
  rvs::lp::Log(log_msg.c_str(), rvs::logdebug);
  rvs::lp::Log("[PQT] ---------------------------------------", rvs::logdebug);

  // Allocate buffers in src and dst pools
  status = hsa_amd_memory_pool_allocate(src_buff, curr_size, 0,
                                    static_cast<void**>(&src_pool_pointer_fwd));
  print_hsa_status("[PQT] send_traffic - hsa_amd_memory_pool_allocate(SRC)",
                   status);
  snprintf(s_buff, sizeof(s_buff), "%p", src_pool_pointer_fwd);
  rvs::lp::Log(std::string("src_pool_pointer_fwd = ") + s_buff, rvs::logdebug);

  status = hsa_amd_memory_pool_allocate(dst_buff, curr_size, 0,
                                  static_cast<void**>(&dst_pool_pointer_fwd));
  print_hsa_status("[PQT] send_traffic - hsa_amd_memory_pool_allocate(DST)",
                   status);
  snprintf(s_buff, sizeof(s_buff), "%p", dst_pool_pointer_fwd);
  rvs::lp::Log(std::string("dst_pool_pointer_fwd = ") + s_buff, rvs::logdebug);

  if (bidirectional == true) {
    status = hsa_amd_memory_pool_allocate(src_buff, curr_size, 0,
                                  static_cast<void**>(&src_pool_pointer_rev));
    print_hsa_status(
      "[PQT] send_traffic BIDIRECTIONAL - hsa_amd_memory_pool_allocate(SRC)",
                     status);
    snprintf(s_buff, sizeof(s_buff), "%p", src_pool_pointer_rev);
    rvs::lp::Log(std::string("src_pool_pointer_rev = ") + s_buff,
                 rvs::logdebug);

    status = hsa_amd_memory_pool_allocate(dst_buff, curr_size, 0,
                                  static_cast<void**>(&dst_pool_pointer_rev));
    print_hsa_status(
      "[PQT] send_traffic BIDIRECTIONAL - hsa_amd_memory_pool_allocate(DST)",
                     status);
    snprintf(s_buff, sizeof(s_buff), "%p", dst_pool_pointer_rev);
    rvs::lp::Log(std::string("dst_pool_pointer_rev = ") + s_buff,
                 rvs::logdebug);
  }

  // Create a signal to wait on copy operation
  status = hsa_signal_create(1, 0, NULL, &signal_fwd);
  print_hsa_status("[PQT] send_traffic - hsa_signal_create()", status);

  // get agent access
  status = hsa_amd_agents_allow_access(1, &src_agent,
                                       NULL, dst_pool_pointer_fwd);
  print_hsa_status(
    "[PQT] send_traffic - hsa_amd_agents_allow_access(SRC)", status);

  status = hsa_amd_agents_allow_access(1, &dst_agent,
                                       NULL, src_pool_pointer_fwd);
  print_hsa_status(
    "[PQT] send_traffic - hsa_amd_agents_allow_access(DST)", status);

  // store signal
  hsa_signal_store_relaxed(signal_fwd, 1);

  if (bidirectional == true) {
    status = hsa_signal_create(1, 0, NULL, &signal_rev);
    print_hsa_status(
      "[PQT] send_traffic BIDIRECTIONAL - hsa_signal_create()", status);

    status = hsa_amd_agents_allow_access(1, &src_agent,
                                         NULL, dst_pool_pointer_rev);
    print_hsa_status(
      "[PQT] send_traffic BIDIRECTIONAL - hsa_amd_agents_allow_access(SRC)",
                     status);

    status = hsa_amd_agents_allow_access(1, &dst_agent,
                                         NULL, src_pool_pointer_rev);
    print_hsa_status(
      "[PQT] send_traffic BIDIRECTIONAL - hsa_amd_agents_allow_access(DST)",
                     status);

    hsa_signal_store_relaxed(signal_rev, 1);
  }

  // Determine if accessibility to dst pool for src agent is not denied
  hsa_amd_memory_pool_access_t access;
  status = hsa_amd_agent_memory_pool_get_info(src_agent, dst_buff,
        HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access);
  print_hsa_status(
    "[PQT] send_traffic - hsa_amd_agent_memory_pool_get_info(SRC->DST)",
                   status);

  if (access == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
    log_msg = std::string("[PQT] send_traffic - HSA_AMD_MEMORY_POOL_ACCESS_") +
    "NEVER_ALLOWED for SRC -> DST), so skip it ...";
    rvs::lp::Log(log_msg.c_str(), rvs::logdebug);
    return 0;
  }

  if (bidirectional == true) {
    status = hsa_amd_agent_memory_pool_get_info(dst_agent, src_buff,
                                HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access);
    print_hsa_status(std::string("[PQT] send_traffic BIDIRECTIONAL - ") +
    "hsa_amd_agent_memory_pool_get_info(DST->SRC)", status);

    if (access == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
      log_msg = std::string("[PQT] send_traffic BIDIRECTIONAL - HSA_AMD_") +
      "MEMORY_POOL_ACCESS_NEVER_ALLOWED for DST -> SRC), so skip it ...";
      rvs::lp::Log(log_msg.c_str(), rvs::logdebug);
      return 0;
    }
  }

  // Add current transfer size
  total_size += curr_size;

  // Copy from src into dst buffer
  status = hsa_amd_memory_async_copy(dst_pool_pointer_fwd, dst_agent,
                                     src_pool_pointer_fwd, src_agent, curr_size,
                                     0, NULL, signal_fwd);
  print_hsa_status(
    "[PQT] send_traffic - hsa_amd_memory_async_copy(SRC -> DST)", status);

  if (bidirectional == true) {
    status = hsa_amd_memory_async_copy(src_pool_pointer_rev, src_agent,
                                       dst_pool_pointer_rev, dst_agent,
                                       curr_size, 0, NULL, signal_rev);
    print_hsa_status(
      "[PQT] send_traffic BIDIRECTIONAL - hsa_amd_memory_async_copy(DST -> SRC)"
      , status);
  }

  // Wait for the forward copy operation to complete
  log_msg = std::string("[PQT] send_traffic - ") +
      "hsa_signal_wait_acquire(SRC -> DST) before ...";
  rvs::lp::Log(log_msg.c_str(), rvs::logdebug);
  while (hsa_signal_wait_acquire(signal_fwd, HSA_SIGNAL_CONDITION_LT,
    1, uint64_t(-1), HSA_WAIT_STATE_ACTIVE)) {}
  log_msg =
    "[PQT] send_traffic - hsa_signal_wait_acquire(SRC -> DST) after ...";
  rvs::lp::Log(log_msg.c_str(), rvs::logdebug);

  if (bidirectional == true) {
    log_msg = std::string("[PQT] send_traffic BIDIRECTIONAL - ") +
    "hsa_signal_wait_acquire(DST -> SRC) before ...";
    rvs::lp::Log(log_msg.c_str(), rvs::logdebug);
    while (hsa_signal_wait_acquire(signal_rev, HSA_SIGNAL_CONDITION_LT,
      1, uint64_t(-1), HSA_WAIT_STATE_ACTIVE)) {}
    log_msg =
      std::string("[PQT] send_traffic BIDIRECTIONAL - ") +
      "hsa_signal_wait_acquire(DST -> SRC) after ...";
    rvs::lp::Log(log_msg.c_str(), rvs::logdebug);
  }

  curr_time = GetCopyTime(bidirectional, signal_fwd, signal_rev)/1000000000;
  log_msg = "[PQT] send_traffic - curr_time = " + std::to_string(curr_time);
  rvs::lp::Log(log_msg.c_str(), rvs::logdebug);

  // convert to GB/s
  if (bidirectional == true) {
    curr_size *= 2;
  }
  bandwidth = (curr_size / curr_time);
  bandwidth /= (1024*1024*1024);
  log_msg = "[PQT] send_traffic - curr_size = " + std::to_string(curr_size) +
    " Bytes and curr_time = " + std::to_string(curr_time) + " bandwidth = " +
    std::to_string(bandwidth) + " GBytes/s";
  rvs::lp::Log(log_msg.c_str(), rvs::logdebug);

  return bandwidth;
}
*/
