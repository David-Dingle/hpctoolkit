// -*-Mode: C++;-*- // technically C99

// * BeginRiceCopyright *****************************************************
//
// --------------------------------------------------------------------------
// Part of HPCToolkit (hpctoolkit.org)
//
// Information about sources of support for research and development of
// HPCToolkit is at 'hpctoolkit.org' and in 'README.Acknowledgments'.
// --------------------------------------------------------------------------
//
// Copyright ((c)) 2002-2019, Rice University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of Rice University (RICE) nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// This software is provided by RICE and contributors "as is" and any
// express or implied warranties, including, but not limited to, the
// implied warranties of merchantability and fitness for a particular
// purpose are disclaimed. In no event shall RICE or contributors be
// liable for any direct, indirect, incidental, special, exemplary, or
// consequential damages (including, but not limited to, procurement of
// substitute goods or services; loss of use, data, or profits; or
// business interruption) however caused and on any theory of liability,
// whether in contract, strict liability, or tort (including negligence
// or otherwise) arising in any way out of the use of this software, even
// if advised of the possibility of such damage.
//
// ******************************************************* EndRiceCopyright *

//******************************************************************************
// macros
//******************************************************************************

#define UNIT_TEST 0

#define DEBUG 0

#include "../gpu-print.h"

//******************************************************************************
// local includes
//******************************************************************************

#include "sanitizer-buffer.h"
#include "sanitizer-api.h"

#include <stddef.h>
#include <gpu-patch.h>
#include <redshow.h>

#include <hpcrun/gpu/nvidia/cubin-id-map.h>
#include <hpcrun/gpu/gpu-function-id-map.h>
#include <hpcrun/memory/hpcrun-malloc.h>

#include "sanitizer-buffer-channel.h"
#include "../gpu-channel-item-allocator.h"


//******************************************************************************
// type declarations
//******************************************************************************

typedef struct sanitizer_buffer_t {
  s_element_t next;

  uint32_t thread_id;
  uint32_t cubin_id;
  uint32_t mod_id;
  int32_t kernel_id;
  uint64_t host_op_id;
  uint32_t stream_id;
  uint32_t type;
  gpu_patch_buffer_t *gpu_patch_buffer;
} sanitizer_buffer_t;

//******************************************************************************
// interface operations 
//******************************************************************************

void
sanitizer_buffer_process
(
 sanitizer_buffer_t *b
)
{
  uint32_t thread_id = b->thread_id;
  uint32_t cubin_id = b->cubin_id;
  uint32_t mod_id = b->mod_id;
  int32_t kernel_id = b->kernel_id;
  uint64_t host_op_id = b->host_op_id;
  uint32_t stream_id = b->stream_id;

  gpu_patch_buffer_t *gpu_patch_buffer = b->gpu_patch_buffer;
  
  redshow_analyze(thread_id, cubin_id, mod_id, kernel_id, host_op_id, stream_id, gpu_patch_buffer);
}


sanitizer_buffer_t *
sanitizer_buffer_alloc
(
 sanitizer_buffer_channel_t *channel
)
{
  return channel_item_alloc(channel, sanitizer_buffer_t);
}


void
sanitizer_buffer_produce
(
 sanitizer_buffer_t *b,
 uint32_t thread_id,
 uint32_t cubin_id,
 uint32_t mod_id,
 int32_t kernel_id,
 uint64_t host_op_id,
 uint32_t stream_id,
 uint32_t type,
 size_t num_records,
 atomic_uint *balance,
 bool async
)
{
  b->thread_id = thread_id;
  b->cubin_id = cubin_id;
  b->mod_id = mod_id;
  b->kernel_id = kernel_id;
  b->host_op_id = host_op_id;
  b->stream_id = stream_id;
  b->type = type;

  // Increase balance
  atomic_fetch_add(balance, 1);
  if (b->gpu_patch_buffer == NULL) {
    // Spin waiting
    while (atomic_load(balance) >= sanitizer_buffer_pool_size_get()) {
      if (!async) {
        b->gpu_patch_buffer = NULL;
        return ;
      }
      sanitizer_process_signal();
    }
//    if (type == GPU_PATCH_TYPE_DEFAULT) {
//      size_t num_records = sanitizer_gpu_patch_record_num_get();
//      b->gpu_patch_buffer = (gpu_patch_buffer_t *) hpcrun_malloc_safe(sizeof(gpu_patch_buffer_t));
//      b->gpu_patch_buffer->records = hpcrun_malloc_safe(num_records * sizeof(gpu_patch_record_t));
//      PRINT("Sanitizer-> Allocate gpu_patch_record_t buffer size %lu\n", num_records * sizeof(gpu_patch_record_t));
//    } else if (type == GPU_PATCH_TYPE_ADDRESS_PATCH) {
//      size_t num_records = sanitizer_gpu_patch_record_num_get();
//      b->gpu_patch_buffer = (gpu_patch_buffer_t *) hpcrun_malloc_safe(sizeof(gpu_patch_buffer_t));
//      b->gpu_patch_buffer->records = hpcrun_malloc_safe(num_records * sizeof(gpu_patch_record_address_t));
//      PRINT("Sanitizer-> Allocate gpu_patch_record_address_t buffer size %lu\n", num_records * sizeof(gpu_patch_record_address_t));
//    } else if (type == GPU_PATCH_TYPE_ADDRESS_ANALYSIS) {
//      size_t num_records = sanitizer_gpu_analysis_record_num_get();
//      b->gpu_patch_buffer = (gpu_patch_buffer_t *) hpcrun_malloc_safe(sizeof(gpu_patch_buffer_t));
//      b->gpu_patch_buffer->records = hpcrun_malloc_safe(num_records * sizeof(gpu_patch_analysis_address_t));
//      PRINT("Sanitizer-> Allocate gpu_patch_analysis_address_t buffer size %lu\n", num_records * sizeof(gpu_patch_analysis_address_t));
//    }

/**
 * Fixed Code
 * */
    CUcontext context = My_Get_Context();
    if (type == GPU_PATCH_TYPE_DEFAULT) {
      size_t num_records = sanitizer_gpu_patch_record_num_get();
      sanitizerAllocHost(context, (void **)(&(b->gpu_patch_buffer)), sizeof(gpu_patch_buffer_t));
      sanitizerAllocHost(context, (void **)(&(b->gpu_patch_buffer->records)), num_records * sizeof(gpu_patch_record_t));
      PRINT("Sanitizer-> Allocate gpu_patch_record_t buffer size %lu\n", num_records * sizeof(gpu_patch_record_t));
    } else if (type == GPU_PATCH_TYPE_ADDRESS_PATCH) {
      size_t num_records = sanitizer_gpu_patch_record_num_get();
      sanitizerAllocHost(context, (void **)(&(b->gpu_patch_buffer)), sizeof(gpu_patch_buffer_t));
      sanitizerAllocHost(context, (void **)(&(b->gpu_patch_buffer->records)), num_records * sizeof(gpu_patch_record_address_t));
      PRINT("Sanitizer-> Allocate gpu_patch_record_address_t buffer size %lu\n", num_records * sizeof(gpu_patch_record_address_t));
    } else if (type == GPU_PATCH_TYPE_ADDRESS_ANALYSIS) {
      size_t num_records = sanitizer_gpu_analysis_record_num_get();
      sanitizerAllocHost(context, (void **)(&(b->gpu_patch_buffer)), sizeof(gpu_patch_buffer_t));
      sanitizerAllocHost(context, (void **)(&(b->gpu_patch_buffer->records)), num_records * sizeof(gpu_patch_analysis_address_t));
      PRINT("Sanitizer-> Allocate gpu_patch_analysis_address_t buffer size %lu\n", num_records * sizeof(gpu_patch_analysis_address_t));
    }
  } else {
    PRINT("Sanitizer-> Reuse buffer\n");
  }
}


void
sanitizer_buffer_free
(
 sanitizer_buffer_channel_t *channel, 
 sanitizer_buffer_t *b,
 atomic_uint *balance
)
{
  channel_item_free(channel, b);
  atomic_fetch_add(balance, -1);
}


gpu_patch_buffer_t *
sanitizer_buffer_entry_gpu_patch_buffer_get
(
 sanitizer_buffer_t *b
)
{
  return b->gpu_patch_buffer;
}
