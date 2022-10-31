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

//***************************************************************************
//
// File:
//   sanitizer-api.c
//
// Purpose:
//   implementation of wrapper around NVIDIA's Sanitizer API
//  
//***************************************************************************

//***************************************************************************
// system includes
//***************************************************************************

#include <stdio.h>
#include <errno.h>     // errno
#include <fcntl.h>     // open
#include <limits.h>    // PATH_MAX
#include <stdio.h>     // sprintf
#include <unistd.h>
#include <sys/stat.h>  // mkdir
#include <string.h>    // strstr
#include <pthread.h>
#include <time.h>

#ifndef HPCRUN_STATIC_LINK
#include <dlfcn.h>
#undef _GNU_SOURCE
#define _GNU_SOURCE
#include <link.h>          // dl_iterate_phdr
#include <linux/limits.h>  // PATH_MAX
#include <string.h>        // strstr
#endif

#include <sanitizer.h>
#include <gpu-patch.h>
#include <redshow.h>
#include <vector_types.h>  // dim3

#include <lib/prof-lean/spinlock.h>
#include <lib/prof-lean/stdatomic.h>

#include <hpcrun/cct2metrics.h>
#include <hpcrun/files.h>
#include <hpcrun/hpcrun_stats.h>
#include <hpcrun/module-ignore-map.h>
#include <hpcrun/main.h>
#include <hpcrun/safe-sampling.h>
#include <hpcrun/sample_event.h>
#include <hpcrun/thread_data.h>
#include <hpcrun/threadmgr.h>
#include <hpcrun/utilities/hpcrun-nanotime.h>

#include <hpcrun/gpu/gpu-application-thread-api.h>
#include <hpcrun/gpu/gpu-monitoring-thread-api.h>
#include <hpcrun/gpu/gpu-correlation-id.h>
#include <hpcrun/gpu/gpu-op-placeholders.h>
#include <hpcrun/gpu/gpu-metrics.h>

#include <hpcrun/sample-sources/libdl.h>
#include <hpcrun/sample-sources/nvidia.h>

#include "cuda-api.h"
#include "cubin-id-map.h"
#include "cubin-hash-map.h"
#include "sanitizer-api.h"
#include "sanitizer-context-map.h"
#include "sanitizer-stream-map.h"
#include "sanitizer-op-map.h"
#include "sanitizer-buffer.h"
#include "sanitizer-buffer-channel.h"
#include "sanitizer-buffer-channel-set.h"
#include "sanitizer-function-list.h"

#define SANITIZER_API_DEBUG 1

#define FUNCTION_NAME_LENGTH 1024

#if SANITIZER_API_DEBUG
#define PRINT(...) fprintf(stderr, __VA_ARGS__)
#else
#define PRINT(...)
#endif

#define MIN2(m1, m2) m1 > m2 ? m2 : m1

#define SANITIZER_FN_NAME(f) DYN_FN_NAME(f)

#define SANITIZER_FN(fn, args) \
  static SanitizerResult (*SANITIZER_FN_NAME(fn)) args

#define HPCRUN_SANITIZER_CALL(fn, args) \
{ \
  SanitizerResult status = SANITIZER_FN_NAME(fn) args; \
  if (status != SANITIZER_SUCCESS) { \
    sanitizer_error_report(status, #fn); \
  } \
}

#define HPCRUN_SANITIZER_CALL_NO_CHECK(fn, args) \
{ \
  SANITIZER_FN_NAME(fn) args; \
}

#define DISPATCH_CALLBACK(fn, args) if (fn) fn args

#define FORALL_SANITIZER_ROUTINES(macro)   \
  macro(sanitizerSubscribe)                \
  macro(sanitizerUnsubscribe)              \
  macro(sanitizerEnableAllDomains)         \
  macro(sanitizerEnableDomain)             \
  macro(sanitizerAlloc)                    \
  macro(sanitizerMemset)                   \
  macro(sanitizerMemcpyDeviceToHost)       \
  macro(sanitizerMemcpyHostToDeviceAsync)  \
  macro(sanitizerSetCallbackData)          \
  macro(sanitizerStreamSynchronize)        \
  macro(sanitizerAddPatchesFromFile)       \
  macro(sanitizerGetFunctionPcAndSize)     \
  macro(sanitizerPatchInstructions)        \
  macro(sanitizerPatchModule)              \
  macro(sanitizerGetResultString)          \
  macro(sanitizerGetStreamHandle)          \
  macro(sanitizerUnpatchModule)


typedef void (*sanitizer_error_callback_t)
(
 const char *type,
 const char *fn,
 const char *error_string
);

typedef cct_node_t *(*sanitizer_correlation_callback_t)
(
 uint64_t id,
 uint32_t skip_frames
);

static cct_node_t *
sanitizer_correlation_callback_dummy // __attribute__((unused))
(
 uint64_t id,
 uint32_t skip_frames
);

static void
sanitizer_error_callback_dummy // __attribute__((unused))
(
 const char *type,
 const char *fn,
 const char *error_string
);

typedef struct {
  pthread_t thread;
  pthread_mutex_t mutex;
  pthread_cond_t cond;
} sanitizer_thread_t;

typedef struct {
  bool flag;
  int32_t persistent_id;
  uint64_t correlation_id;
  uint64_t start;
  uint64_t end;
} sanitizer_memory_register_delegate_t;

typedef struct stream_to_integer {
  CUstream stream;
  uint32_t stream_id;
  struct stream_to_integer* next;
} stream_to_integer_t;

static stream_to_integer_t* stream_to_integer_linkedlist = NULL;

// only subscribed by the main thread
static Sanitizer_SubscriberHandle sanitizer_subscriber_handle;
// Single background process thread, can be extended
static sanitizer_thread_t sanitizer_thread;

// Configurable variables
static int sanitizer_buffer_pool_size = 0;
static int sanitizer_pc_views = 0;
static int sanitizer_mem_views = 0;

static redshow_approx_level_t sanitizer_approx_level = REDSHOW_APPROX_NONE;
static redshow_data_type_t sanitizer_data_type = REDSHOW_DATA_FLOAT;

// CPU async
static bool sanitizer_analysis_async = false;

// Patch info (GPU)
static size_t sanitizer_gpu_patch_record_num = 0;
static size_t sanitizer_gpu_patch_record_size = 0;
static uint32_t sanitizer_gpu_patch_type = GPU_PATCH_TYPE_DEFAULT;

// Analysis info (GPU)
static int sanitizer_gpu_analysis_record_num = 0;
static size_t sanitizer_gpu_analysis_record_size = 0;
static uint32_t sanitizer_gpu_analysis_blocks = 0;
static uint32_t sanitizer_gpu_analysis_type = GPU_PATCH_TYPE_ADDRESS_ANALYSIS;
static bool sanitizer_read_trace_ignore = false;
static bool sanitizer_data_flow_hash = false;
static bool sanitizer_liveness_ongpu = false;
static bool sanitizer_torch_analysis = false;
static bool sanitizer_torch_analysis_ongpu = false;

static __thread bool sanitizer_stop_flag = false;
static __thread bool sanitizer_context_creation_flag = false;
static __thread uint32_t sanitizer_thread_id_self = (1 << 30);
static __thread uint32_t sanitizer_thread_id_local = 0;
static __thread CUcontext sanitizer_thread_context = NULL;

static __thread sanitizer_memory_register_delegate_t sanitizer_memory_register_delegate = {
  .flag = false
};

// Host buffers are per-thread
static __thread gpu_patch_buffer_t *sanitizer_gpu_patch_buffer_host = NULL;
static __thread gpu_patch_buffer_t *sanitizer_gpu_patch_buffer_addr_read_host = NULL;
static __thread gpu_patch_buffer_t *sanitizer_gpu_patch_buffer_addr_write_host = NULL;
static __thread gpu_patch_aux_address_dict_t *sanitizer_gpu_patch_aux_addr_dict_host = NULL;
static __thread gpu_patch_aux_address_dict_t *sanitizer_gpu_patch_torch_aux_addr_dict_host = NULL;

// Reset and device buffers are per-context
static __thread gpu_patch_buffer_t *sanitizer_gpu_patch_buffer_reset = NULL;
static __thread gpu_patch_buffer_t *sanitizer_gpu_patch_buffer_addr_read_reset = NULL;
static __thread gpu_patch_buffer_t *sanitizer_gpu_patch_buffer_addr_write_reset = NULL;
static __thread gpu_patch_aux_address_dict_t *sanitizer_gpu_patch_aux_addr_dict_reset = NULL;
static __thread gpu_patch_aux_address_dict_t *sanitizer_gpu_patch_torch_aux_addr_dict_reset = NULL;

static __thread gpu_patch_buffer_t *sanitizer_gpu_patch_buffer_device = NULL;
static __thread gpu_patch_buffer_t *sanitizer_gpu_patch_buffer_addr_read_device = NULL;
static __thread gpu_patch_buffer_t *sanitizer_gpu_patch_buffer_addr_write_device = NULL;
static __thread gpu_patch_aux_address_dict_t *sanitizer_gpu_patch_aux_addr_dict_device = NULL;
static __thread gpu_patch_aux_address_dict_t *sanitizer_gpu_patch_torch_aux_addr_dict_device = NULL;

static sanitizer_correlation_callback_t sanitizer_correlation_callback =
  sanitizer_correlation_callback_dummy;

static sanitizer_error_callback_t sanitizer_error_callback =
  sanitizer_error_callback_dummy;

static atomic_uint sanitizer_thread_id = ATOMIC_VAR_INIT(0);
static atomic_uint sanitizer_process_thread_counter = ATOMIC_VAR_INIT(0);
static atomic_bool sanitizer_process_awake_flag = ATOMIC_VAR_INIT(0);
static atomic_bool sanitizer_process_stop_flag = ATOMIC_VAR_INIT(0);

static sanitizer_function_list_entry_t *sanitizer_whitelist = NULL;
static sanitizer_function_list_entry_t *sanitizer_blacklist = NULL;

//----------------------------------------------------------
// sanitizer function pointers for late binding
//----------------------------------------------------------

SANITIZER_FN
(
 sanitizerSubscribe,
 (
  Sanitizer_SubscriberHandle* subscriber,
  Sanitizer_CallbackFunc callback,
  void* userdata
 )
);


SANITIZER_FN
(
 sanitizerUnsubscribe,
 (
  Sanitizer_SubscriberHandle subscriber
 )
);


SANITIZER_FN
(
 __attribute__((unused)) sanitizerEnableAllDomains,
 (
  uint32_t enable,
  Sanitizer_SubscriberHandle subscriber
 ) 
);


SANITIZER_FN
(
 sanitizerEnableDomain,
 (
  uint32_t enable,
  Sanitizer_SubscriberHandle subscriber,
  Sanitizer_CallbackDomain domain
 ) 
);


SANITIZER_FN
(
 sanitizerAlloc,
 (
  CUcontext ctx,
  void** devPtr,
  size_t size
 )
);


SANITIZER_FN
(
 sanitizerMemset,
 (
  void* devPtr,
  int value,
  size_t count,
  Sanitizer_StreamHandle stream
 )
);


SANITIZER_FN
(
 sanitizerStreamSynchronize,
 (
  Sanitizer_StreamHandle stream
 )
);


SANITIZER_FN
(
 sanitizerMemcpyDeviceToHost,
 (
  void* dst,
  void* src,
  size_t count,
  Sanitizer_StreamHandle stream
 ) 
);


SANITIZER_FN
(
 sanitizerMemcpyHostToDeviceAsync,
 (
  void* dst,
  void* src,
  size_t count,
  Sanitizer_StreamHandle stream
 ) 
);


SANITIZER_FN
(
 sanitizerSetCallbackData,
 (
  CUfunction function,
  const void* userdata
 ) 
);


SANITIZER_FN
(
 sanitizerAddPatchesFromFile,
 (
  const char* filename,
  CUcontext ctx
 ) 
);


SANITIZER_FN
(
 sanitizerPatchInstructions,
 (
  const Sanitizer_InstructionId instructionId,
  CUmodule module,
  const char* deviceCallbackName
 ) 
);


SANITIZER_FN
(
 sanitizerPatchModule,
 (
  CUmodule module
 )
);


SANITIZER_FN
(
 __attribute__((unused)) sanitizerUnpatchModule,
 (
  CUmodule module
 )
);


SANITIZER_FN
(
 sanitizerGetResultString,
 (
  SanitizerResult result,
  const char **str
 )
);


SANITIZER_FN
(
 sanitizerGetFunctionPcAndSize,
 (
  CUmodule module,
  const char *functionName,
  uint64_t* pc,
  uint64_t* size
 )
);


SANITIZER_FN
(
 sanitizerGetStreamHandle,
 (
  CUcontext ctx,
  CUstream stream,
  Sanitizer_StreamHandle *hStream
 )
);

//******************************************************************************
// forward declaration operations
//******************************************************************************

static Sanitizer_StreamHandle sanitizer_priority_stream_get(CUcontext context);

static Sanitizer_StreamHandle sanitizer_kernel_stream_get(CUcontext context);

static void sanitizer_kernel_launch(CUcontext context);

//******************************************************************************
// private operations
//******************************************************************************
static uint32_t
sanitizer_stearm_id_query
(
  CUstream stream
)
{
  uint32_t stream_id;
  if (stream_to_integer_linkedlist == NULL) {
    // @Lin-Mao: no need to free when using hpcrun_malloc_safe?
    stream_to_integer_linkedlist = (stream_to_integer_t*) hpcrun_malloc_safe(sizeof(stream_to_integer_t));
    stream_to_integer_linkedlist->stream = stream;
    stream_to_integer_linkedlist->stream_id = 0;
    stream_to_integer_linkedlist->next = NULL;
    stream_id = stream_to_integer_linkedlist->stream_id;
  } else {
    stream_to_integer_t* p = stream_to_integer_linkedlist;
    stream_to_integer_t* q = p;
    uint32_t count = 0;
    bool not_fount = true;
    while (p) {
      if (p->stream == stream) {
        stream_id = p->stream_id;
        not_fount = false;
        break;
      } else {
        count = p->stream_id;
      }
      q = p;
      p = p->next;
    }
    if (not_fount) {
      stream_to_integer_t* node = (stream_to_integer_t*) hpcrun_malloc_safe(sizeof(stream_to_integer_t));
      node->stream = stream;
      node->stream_id = count + 1;
      node->next = q->next;
      q->next = node;
      stream_id = node->stream_id;
    }
  }
  return stream_id;

}


static cct_node_t *
sanitizer_correlation_callback_dummy // __attribute__((unused))
(
 uint64_t id,
 uint32_t skip_frames
)
{
  return NULL;
}


static void
sanitizer_error_callback_dummy // __attribute__((unused))
(
 const char *type,
 const char *fn,
 const char *error_string
)
{
  PRINT("Sanitizer-> %s: function %s failed with error %s\n", type, fn, error_string);
  exit(-1);
}


static void
sanitizer_error_report
(
 SanitizerResult error,
 const char *fn
)
{
  const char *error_string;
  SANITIZER_FN_NAME(sanitizerGetResultString)(error, &error_string);
  sanitizer_error_callback("Sanitizer result error", fn, error_string);
}


static void
sanitizer_log_data_callback
(
 int32_t kernel_id,
 gpu_patch_buffer_t *trace_data
)
{
}


static void
sanitizer_dtoh
(
 uint64_t host,
 uint64_t device,
 uint64_t len
)
{
  Sanitizer_StreamHandle priority_stream = sanitizer_priority_stream_get(sanitizer_thread_context);

  // sanitizerMemcpyDeviceToHost is thread safe
  HPCRUN_SANITIZER_CALL(sanitizerMemcpyDeviceToHost,
    ((void *)host, (void *)device, len, priority_stream));
}


static void
sanitizer_record_data_callback
(
 uint32_t cubin_id,
 int32_t kernel_id,
 redshow_record_data_t *record_data
)
{
  gpu_activity_t ga;

  ga.kind = GPU_ACTIVITY_REDUNDANCY;

  if (record_data->analysis_type == REDSHOW_ANALYSIS_SPATIAL_REDUNDANCY &&
    record_data->access_type == REDSHOW_ACCESS_READ) {
    ga.details.redundancy.type = GPU_RED_SPATIAL_READ_RED;
  } else if (record_data->analysis_type == REDSHOW_ANALYSIS_SPATIAL_REDUNDANCY &&
    record_data->access_type == REDSHOW_ACCESS_WRITE) {
    ga.details.redundancy.type = GPU_RED_SPATIAL_WRITE_RED;
  } else if (record_data->analysis_type == REDSHOW_ANALYSIS_TEMPORAL_REDUNDANCY &&
    record_data->access_type == REDSHOW_ACCESS_READ) {
    ga.details.redundancy.type = GPU_RED_TEMPORAL_READ_RED;
  } else if (record_data->analysis_type == REDSHOW_ANALYSIS_TEMPORAL_REDUNDANCY &&
    record_data->access_type == REDSHOW_ACCESS_WRITE) {
    ga.details.redundancy.type = GPU_RED_TEMPORAL_WRITE_RED;
  } else {
    assert(0);
  }
  cstack_ptr_set(&(ga.next), 0);
  
  uint32_t num_views = record_data->num_views;
  uint32_t i;
  for (i = 0; i < num_views; ++i) {
    uint32_t function_index = record_data->views[i].function_index;
    uint64_t pc_offset = record_data->views[i].pc_offset;
    uint64_t red_count = record_data->views[i].red_count;
    uint64_t access_count = record_data->views[i].access_count;

    ip_normalized_t ip = cubin_id_transform(cubin_id, function_index, pc_offset);
    sanitizer_op_map_entry_t *entry = sanitizer_op_map_lookup(kernel_id);
    if (entry != NULL) {
      cct_node_t *host_op_node = sanitizer_op_map_op_get(entry);
      ga.cct_node = hpcrun_cct_insert_ip_norm(host_op_node, ip);
      ga.details.redundancy.red_count = red_count;
      ga.details.redundancy.access_count = access_count;
      // Associate record_data with calling context (kernel_id)
      gpu_metrics_attribute(&ga);
    } else {
      PRINT("Sanitizer-> NULL cct_node with kernel_id %d\n", kernel_id);
    }
  }
}



#ifndef HPCRUN_STATIC_LINK
static const char *
sanitizer_path
(
 void
)
{
  const char *path = "libsanitizer-public.so";
    
  static char buffer[PATH_MAX];
  buffer[0] = 0;

  // open an NVIDIA library to find the CUDA path with dl_iterate_phdr
  // note: a version of this file with a more specific name may 
  // already be loaded. thus, even if the dlopen fails, we search with
  // dl_iterate_phdr.
  void *h = monitor_real_dlopen("libcudart.so", RTLD_LOCAL | RTLD_LAZY);

  if (cuda_path(buffer)) {
    // invariant: buffer contains CUDA home 
    strcat(buffer, "compute-sanitizer/libsanitizer-public.so");
    path = buffer;
  }

  if (h) monitor_real_dlclose(h);

  return path;
}

#endif

//******************************************************************************
// asynchronous process thread
//******************************************************************************

void
sanitizer_process_signal
(
)
{
  pthread_cond_t *cond = &(sanitizer_thread.cond);
  pthread_mutex_t *mutex = &(sanitizer_thread.mutex);

  pthread_mutex_lock(mutex);

  atomic_store(&sanitizer_process_awake_flag, true);

  pthread_cond_signal(cond);

  pthread_mutex_unlock(mutex);
}


static void
sanitizer_process_await
(
)
{
  pthread_cond_t *cond = &(sanitizer_thread.cond);
  pthread_mutex_t *mutex = &(sanitizer_thread.mutex);

  pthread_mutex_lock(mutex);

  while (!atomic_load(&sanitizer_process_awake_flag)) {
    pthread_cond_wait(cond, mutex);
  }

  atomic_store(&sanitizer_process_awake_flag, false);

  pthread_mutex_unlock(mutex);
}


static void*
sanitizer_process_thread
(
 void *arg
)
{
  pthread_cond_t *cond = &(sanitizer_thread.cond);
  pthread_mutex_t *mutex = &(sanitizer_thread.mutex);

  while (!atomic_load(&sanitizer_process_stop_flag)) {
    redshow_analysis_begin();
    sanitizer_buffer_channel_set_consume();
    redshow_analysis_end();
    sanitizer_process_await();
  }

  // Last records
  sanitizer_buffer_channel_set_consume();

  // Create thread data
  thread_data_t* td = NULL;
  int id = sanitizer_thread_id_self;
  hpcrun_threadMgr_non_compact_data_get(id, NULL, &td);
  hpcrun_set_thread_data(td);
  
  atomic_fetch_add(&sanitizer_process_thread_counter, -1);

  pthread_mutex_destroy(mutex);
  pthread_cond_destroy(cond);

  return NULL;
}


//******************************************************************************
// Cubins
//******************************************************************************

static void
sanitizer_buffer_init
(
 CUcontext context
)
{

  if (sanitizer_gpu_patch_buffer_device != NULL) {
    // All entries have been initialized
    return;
  }

  // Get cached entry
  sanitizer_context_map_entry_t *entry = sanitizer_context_map_init(context);
  Sanitizer_StreamHandle priority_stream = sanitizer_priority_stream_get(context);

  sanitizer_gpu_patch_buffer_device = sanitizer_context_map_entry_buffer_device_get(entry);
  sanitizer_gpu_patch_buffer_addr_read_device = sanitizer_context_map_entry_buffer_addr_read_device_get(entry);
  sanitizer_gpu_patch_buffer_addr_write_device = sanitizer_context_map_entry_buffer_addr_write_device_get(entry);
  sanitizer_gpu_patch_aux_addr_dict_device = sanitizer_context_map_entry_aux_addr_dict_device_get(entry);
  sanitizer_gpu_patch_torch_aux_addr_dict_device = sanitizer_context_map_entry_torch_aux_addr_dict_device_get(entry);

  sanitizer_gpu_patch_buffer_reset = sanitizer_context_map_entry_buffer_reset_get(entry);
  sanitizer_gpu_patch_buffer_addr_read_reset = sanitizer_context_map_entry_buffer_addr_read_reset_get(entry);
  sanitizer_gpu_patch_buffer_addr_write_reset = sanitizer_context_map_entry_buffer_addr_write_reset_get(entry);
  sanitizer_gpu_patch_aux_addr_dict_reset = sanitizer_context_map_entry_aux_addr_dict_reset_get(entry);
  sanitizer_gpu_patch_torch_aux_addr_dict_reset = sanitizer_context_map_entry_torch_aux_addr_dict_reset_get(entry);

  if (sanitizer_gpu_patch_buffer_device == NULL) {
    // Allocated buffer
    void *gpu_patch_records = NULL;
    // gpu_patch_buffer
    HPCRUN_SANITIZER_CALL(sanitizerAlloc, (context, (void **)(&(sanitizer_gpu_patch_buffer_device)), sizeof(gpu_patch_buffer_t)));
    HPCRUN_SANITIZER_CALL(sanitizerMemset, (sanitizer_gpu_patch_buffer_device, 0, sizeof(gpu_patch_buffer_t), priority_stream));

    PRINT("Sanitizer-> Allocate gpu_patch_buffer %p, size %zu\n", sanitizer_gpu_patch_buffer_device, sizeof(gpu_patch_buffer_t));

    // gpu_patch_buffer_t->records
    HPCRUN_SANITIZER_CALL(sanitizerAlloc,
      (context, &gpu_patch_records, sanitizer_gpu_patch_record_num * sanitizer_gpu_patch_record_size));
    HPCRUN_SANITIZER_CALL(sanitizerMemset,
      (gpu_patch_records, 0, sanitizer_gpu_patch_record_num * sanitizer_gpu_patch_record_size, priority_stream));
    PRINT("Sanitizer-> Allocate gpu_patch_records %p, size %zu\n", \
      gpu_patch_records, sanitizer_gpu_patch_record_num * sanitizer_gpu_patch_record_size);

    // Allocate reset record
    sanitizer_gpu_patch_buffer_reset = (gpu_patch_buffer_t *)hpcrun_malloc_safe(sizeof(gpu_patch_buffer_t));
    sanitizer_gpu_patch_buffer_reset->full = 0;
    sanitizer_gpu_patch_buffer_reset->analysis = 0;
    sanitizer_gpu_patch_buffer_reset->head_index = 0;
    sanitizer_gpu_patch_buffer_reset->tail_index = 0;
    sanitizer_gpu_patch_buffer_reset->size = sanitizer_gpu_patch_record_num;
    sanitizer_gpu_patch_buffer_reset->num_threads = 0;
    sanitizer_gpu_patch_buffer_reset->type = sanitizer_gpu_patch_type;
    sanitizer_gpu_patch_buffer_reset->flags = GPU_PATCH_NONE;
    sanitizer_gpu_patch_buffer_reset->aux = NULL;
    sanitizer_gpu_patch_buffer_reset->records = gpu_patch_records;

    if (sanitizer_gpu_analysis_blocks != 0) {
      sanitizer_gpu_patch_buffer_reset->flags |= GPU_PATCH_ANALYSIS;
    }

    if (sanitizer_read_trace_ignore) {
      void *gpu_patch_aux = NULL;
      // Use a dict to filter read trace
      HPCRUN_SANITIZER_CALL(sanitizerAlloc, (context, (void **)(&(gpu_patch_aux)), sizeof(gpu_patch_aux_address_dict_t)));
      HPCRUN_SANITIZER_CALL(sanitizerMemset, (gpu_patch_aux, 0, sizeof(gpu_patch_aux_address_dict_t), priority_stream));

      PRINT("Sanitizer-> Allocate gpu_patch_aux %p, size %zu\n", gpu_patch_aux, sizeof(gpu_patch_aux_address_dict_t));

      // Update map
      sanitizer_gpu_patch_buffer_reset->aux = gpu_patch_aux;
      sanitizer_context_map_aux_addr_dict_device_update(context, gpu_patch_aux);
    }

    if (sanitizer_liveness_ongpu) {
      void *gpu_patch_liveness_aux = NULL;
      // Used for liveness analysis on gpu
      HPCRUN_SANITIZER_CALL(sanitizerAlloc, (context, (void **)(&(gpu_patch_liveness_aux)), sizeof(gpu_patch_aux_address_dict_t)));
      HPCRUN_SANITIZER_CALL(sanitizerMemset, (gpu_patch_liveness_aux, 0, sizeof(gpu_patch_aux_address_dict_t), priority_stream));

      PRINT("Sanitizer-> Allocate gpu_patch_liveness_aux %p, size %zu\n", gpu_patch_liveness_aux, sizeof(gpu_patch_aux_address_dict_t));

      // Update map
      sanitizer_gpu_patch_buffer_reset->aux = gpu_patch_liveness_aux;
      sanitizer_context_map_aux_addr_dict_device_update(context, gpu_patch_liveness_aux);
    }

    if (sanitizer_torch_analysis_ongpu) {
      void *gpu_patch_liveness_aux = NULL;
      void *gpu_patch_torch_liveness_aux = NULL;

      // for memory liveness analysis on GPU
      HPCRUN_SANITIZER_CALL(sanitizerAlloc, (context, (void **)(&(gpu_patch_liveness_aux)), sizeof(gpu_patch_aux_address_dict_t)));
      HPCRUN_SANITIZER_CALL(sanitizerMemset, (gpu_patch_liveness_aux, 0, sizeof(gpu_patch_aux_address_dict_t), priority_stream));
      PRINT("Sanitizer-> Allocate gpu_patch_liveness_aux %p, size %zu\n", gpu_patch_liveness_aux, sizeof(gpu_patch_aux_address_dict_t));

      // for torch sub-memory analysis on GPU
      HPCRUN_SANITIZER_CALL(sanitizerAlloc, (context, (void **)(&(gpu_patch_torch_liveness_aux)), sizeof(gpu_patch_aux_address_dict_t)));
      HPCRUN_SANITIZER_CALL(sanitizerMemset, (gpu_patch_torch_liveness_aux, 0, sizeof(gpu_patch_aux_address_dict_t), priority_stream));
      PRINT("Sanitizer-> Allocate gpu_patch_torch_liveness_aux %p, size %zu\n", gpu_patch_torch_liveness_aux, sizeof(gpu_patch_aux_address_dict_t));
      
      // Update map
      sanitizer_gpu_patch_buffer_reset->aux = gpu_patch_liveness_aux;
      sanitizer_gpu_patch_buffer_reset->torch_aux = gpu_patch_torch_liveness_aux;
      sanitizer_context_map_aux_addr_dict_device_update(context, gpu_patch_liveness_aux);
      sanitizer_context_map_torch_aux_addr_dict_device_update(context, gpu_patch_torch_liveness_aux);
      
    }

    HPCRUN_SANITIZER_CALL(sanitizerMemcpyHostToDeviceAsync,
      (sanitizer_gpu_patch_buffer_device, sanitizer_gpu_patch_buffer_reset, sizeof(gpu_patch_buffer_t), priority_stream));

    // Update map
    sanitizer_context_map_buffer_device_update(context, sanitizer_gpu_patch_buffer_device);
    sanitizer_context_map_buffer_reset_update(context, sanitizer_gpu_patch_buffer_reset);

    if (sanitizer_gpu_analysis_blocks != 0) {
      // Read
      HPCRUN_SANITIZER_CALL(sanitizerAlloc, (context, (void **)(&(sanitizer_gpu_patch_buffer_addr_read_device)),
          sizeof(gpu_patch_buffer_t)));
      HPCRUN_SANITIZER_CALL(sanitizerMemset, (sanitizer_gpu_patch_buffer_addr_read_device, 0,
          sizeof(gpu_patch_buffer_t), priority_stream));
      PRINT("Sanitizer-> Allocate sanitizer_gpu_patch_buffer_addr_read_device %p, size %zu\n", \
        sanitizer_gpu_patch_buffer_addr_read_device, sizeof(gpu_patch_buffer_t));

      // gpu_patch_buffer_t->records
      HPCRUN_SANITIZER_CALL(sanitizerAlloc,
        (context, &gpu_patch_records, sanitizer_gpu_analysis_record_num * sanitizer_gpu_analysis_record_size));
      HPCRUN_SANITIZER_CALL(sanitizerMemset,
        (gpu_patch_records, 0, sanitizer_gpu_analysis_record_num * sanitizer_gpu_analysis_record_size, priority_stream));
      PRINT("Sanitizer-> Allocate gpu_patch_records %p, size %zu\n", \
        gpu_patch_records, sanitizer_gpu_analysis_record_num * sanitizer_gpu_analysis_record_size);

      sanitizer_gpu_patch_buffer_addr_read_reset = (gpu_patch_buffer_t *)hpcrun_malloc_safe(sizeof(gpu_patch_buffer_t));
      sanitizer_gpu_patch_buffer_addr_read_reset->full = 0;
      sanitizer_gpu_patch_buffer_addr_read_reset->analysis = 0;
      sanitizer_gpu_patch_buffer_addr_read_reset->head_index = 0;
      sanitizer_gpu_patch_buffer_addr_read_reset->tail_index = 0;
      sanitizer_gpu_patch_buffer_addr_read_reset->size = sanitizer_gpu_analysis_record_num;
      sanitizer_gpu_patch_buffer_addr_read_reset->num_threads = GPU_PATCH_ANALYSIS_THREADS;
      sanitizer_gpu_patch_buffer_addr_read_reset->type = GPU_PATCH_TYPE_ADDRESS_ANALYSIS;
      sanitizer_gpu_patch_buffer_addr_read_reset->flags = GPU_PATCH_READ | GPU_PATCH_ANALYSIS;
      sanitizer_gpu_patch_buffer_addr_read_reset->aux = NULL;
      sanitizer_gpu_patch_buffer_addr_read_reset->records = gpu_patch_records;

      HPCRUN_SANITIZER_CALL(sanitizerMemcpyHostToDeviceAsync, (sanitizer_gpu_patch_buffer_addr_read_device,
          sanitizer_gpu_patch_buffer_addr_read_reset, sizeof(gpu_patch_buffer_t), priority_stream));

      // Write
      HPCRUN_SANITIZER_CALL(sanitizerAlloc, (context, (void **)(&(sanitizer_gpu_patch_buffer_addr_write_device)),
          sizeof(gpu_patch_buffer_t)));
      HPCRUN_SANITIZER_CALL(sanitizerMemset, (sanitizer_gpu_patch_buffer_addr_write_device, 0,
          sizeof(gpu_patch_buffer_t), priority_stream));
      PRINT("Sanitizer-> Allocate sanitizer_gpu_patch_buffer_addr_write_device %p, size %zu\n",
        sanitizer_gpu_patch_buffer_addr_write_device, sizeof(gpu_patch_buffer_t));

      // gpu_patch_buffer_t->records
      HPCRUN_SANITIZER_CALL(sanitizerAlloc,
        (context, &gpu_patch_records, sanitizer_gpu_analysis_record_num * sanitizer_gpu_analysis_record_size));
      HPCRUN_SANITIZER_CALL(sanitizerMemset,
        (gpu_patch_records, 0, sanitizer_gpu_analysis_record_num * sanitizer_gpu_analysis_record_size, priority_stream));
      PRINT("Sanitizer-> Allocate gpu_patch_records %p, size %zu\n", \
        gpu_patch_records, sanitizer_gpu_analysis_record_num * sanitizer_gpu_analysis_record_size);

      sanitizer_gpu_patch_buffer_addr_write_reset = (gpu_patch_buffer_t *)hpcrun_malloc_safe(sizeof(gpu_patch_buffer_t));
      sanitizer_gpu_patch_buffer_addr_write_reset->full = 0;
      sanitizer_gpu_patch_buffer_addr_write_reset->analysis = 0;
      sanitizer_gpu_patch_buffer_addr_write_reset->head_index = 0;
      sanitizer_gpu_patch_buffer_addr_write_reset->tail_index = 0;
      sanitizer_gpu_patch_buffer_addr_write_reset->size = sanitizer_gpu_analysis_record_num;
      sanitizer_gpu_patch_buffer_addr_write_reset->num_threads = GPU_PATCH_ANALYSIS_THREADS;
      sanitizer_gpu_patch_buffer_addr_write_reset->type = GPU_PATCH_TYPE_ADDRESS_ANALYSIS;
      sanitizer_gpu_patch_buffer_addr_write_reset->flags = GPU_PATCH_WRITE | GPU_PATCH_ANALYSIS;
      sanitizer_gpu_patch_buffer_addr_write_reset->aux = NULL;
      sanitizer_gpu_patch_buffer_addr_write_reset->records = gpu_patch_records;

      HPCRUN_SANITIZER_CALL(sanitizerMemcpyHostToDeviceAsync, (sanitizer_gpu_patch_buffer_addr_write_device,
          sanitizer_gpu_patch_buffer_addr_write_reset, sizeof(gpu_patch_buffer_t), priority_stream));

      // Update map
      sanitizer_context_map_buffer_addr_read_device_update(context, sanitizer_gpu_patch_buffer_addr_read_device);
      sanitizer_context_map_buffer_addr_write_device_update(context, sanitizer_gpu_patch_buffer_addr_write_device);
      sanitizer_context_map_buffer_addr_read_reset_update(context, sanitizer_gpu_patch_buffer_addr_read_reset);
      sanitizer_context_map_buffer_addr_write_reset_update(context, sanitizer_gpu_patch_buffer_addr_write_reset);
    }

    // Ensure data copy is done
    HPCRUN_SANITIZER_CALL(sanitizerStreamSynchronize, (priority_stream));
  }
}


// case == SANITIZER_CBID_RESOURCE_MODULE_LOADED
static void
sanitizer_load_callback
(
 CUcontext context,
 CUmodule module, 
 const void *cubin, 
 size_t cubin_size
)
{
  hpctoolkit_cumod_st_t *cumod = (hpctoolkit_cumod_st_t *)module;
  uint32_t cubin_id = cumod->cubin_id;
  uint32_t mod_id = cumod->mod_id;

  // Compute hash for cubin and store it into a map
  cubin_hash_map_entry_t *cubin_hash_entry = cubin_hash_map_lookup(cubin_id);
  unsigned char *hash;
  unsigned int hash_len;
  if (cubin_hash_entry == NULL) {
    cubin_hash_map_insert(cubin_id, cubin, cubin_size);
    cubin_hash_entry = cubin_hash_map_lookup(cubin_id);
  }
  hash = cubin_hash_map_entry_hash_get(cubin_hash_entry, &hash_len);

  // Create file name
  char file_name[PATH_MAX];
  size_t i;
  size_t used = 0;
  used += sprintf(&file_name[used], "%s", hpcrun_files_output_directory());
  used += sprintf(&file_name[used], "%s", "/cubins/");
  mkdir(file_name, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  for (i = 0; i < hash_len; ++i) {
    used += sprintf(&file_name[used], "%02x", hash[i]);
  }
  used += sprintf(&file_name[used], "%s", ".cubin");
  PRINT("Sanitizer-> cubin_id %d hash %s\n", cubin_id, file_name);

  uint32_t hpctoolkit_module_id;
  load_module_t *load_module = NULL;
  hpcrun_loadmap_lock();
  if ((load_module = hpcrun_loadmap_findByName(file_name)) == NULL) {
    hpctoolkit_module_id = hpcrun_loadModule_add(file_name);
  } else {
    hpctoolkit_module_id = load_module->id;
  }
  hpcrun_loadmap_unlock();
  PRINT("Sanitizer-> <cubin_id %d, mod_id %d> -> hpctoolkit_module_id %d\n", cubin_id, mod_id, hpctoolkit_module_id);

  // Compute elf vector
  Elf_SymbolVector *elf_vector = computeCubinFunctionOffsets(cubin, cubin_size);

  // Register cubin module
  cubin_id_map_insert(cubin_id, hpctoolkit_module_id, elf_vector);

  // Query cubin function offsets
  uint64_t *addrs = (uint64_t *)hpcrun_malloc_safe(sizeof(uint64_t) * elf_vector->nsymbols);
  for (i = 0; i < elf_vector->nsymbols; ++i) {
    addrs[i] = 0;
    if (elf_vector->symbols[i] != 0) {
      uint64_t pc;
      uint64_t size;
      // do not check error
      HPCRUN_SANITIZER_CALL_NO_CHECK(sanitizerGetFunctionPcAndSize, (module, elf_vector->names[i], &pc, &size));
      addrs[i] = pc;
    }
  }
  redshow_cubin_cache_register(cubin_id, mod_id, elf_vector->nsymbols, addrs, file_name);

  PRINT("Sanitizer-> Context %p Patch CUBIN: \n", context);
  PRINT("Sanitizer-> %s\n", HPCTOOLKIT_GPU_PATCH);
  // patch binary
  if (sanitizer_gpu_patch_type == GPU_PATCH_TYPE_ADDRESS_PATCH) {
    if (sanitizer_liveness_ongpu) {
      HPCRUN_SANITIZER_CALL(sanitizerAddPatchesFromFile, (HPCTOOLKIT_GPU_PATCH "gpu-patch-aux.fatbin", context));
      HPCRUN_SANITIZER_CALL(sanitizerPatchInstructions,
      (SANITIZER_INSTRUCTION_GLOBAL_MEMORY_ACCESS, module, "sanitizer_global_memory_access_callback"));

    } else if (sanitizer_torch_analysis_ongpu) {
      HPCRUN_SANITIZER_CALL(sanitizerAddPatchesFromFile, (HPCTOOLKIT_GPU_PATCH "gpu-patch-torch-aux.fatbin", context));
      HPCRUN_SANITIZER_CALL(sanitizerPatchInstructions,
      (SANITIZER_INSTRUCTION_GLOBAL_MEMORY_ACCESS, module, "sanitizer_global_memory_access_callback"));

    } else {
      // Only analyze global memory
      HPCRUN_SANITIZER_CALL(sanitizerAddPatchesFromFile, (HPCTOOLKIT_GPU_PATCH "gpu-patch-address.fatbin", context));
      HPCRUN_SANITIZER_CALL(sanitizerPatchInstructions,
      (SANITIZER_INSTRUCTION_GLOBAL_MEMORY_ACCESS, module, "sanitizer_global_memory_access_callback"));
    }
  } else {
    HPCRUN_SANITIZER_CALL(sanitizerAddPatchesFromFile, (HPCTOOLKIT_GPU_PATCH "gpu-patch.fatbin", context));
    HPCRUN_SANITIZER_CALL(sanitizerPatchInstructions,
      (SANITIZER_INSTRUCTION_GLOBAL_MEMORY_ACCESS, module, "sanitizer_global_memory_access_callback"));
    HPCRUN_SANITIZER_CALL(sanitizerPatchInstructions,
      (SANITIZER_INSTRUCTION_SHARED_MEMORY_ACCESS, module, "sanitizer_shared_memory_access_callback"));
    HPCRUN_SANITIZER_CALL(sanitizerPatchInstructions,
      (SANITIZER_INSTRUCTION_LOCAL_MEMORY_ACCESS, module, "sanitizer_local_memory_access_callback"));
    HPCRUN_SANITIZER_CALL(sanitizerPatchInstructions,
      (SANITIZER_INSTRUCTION_BLOCK_ENTER, module, "sanitizer_block_enter_callback"));
  }
  HPCRUN_SANITIZER_CALL(sanitizerPatchInstructions,
    (SANITIZER_INSTRUCTION_BLOCK_EXIT, module, "sanitizer_block_exit_callback"));
  HPCRUN_SANITIZER_CALL(sanitizerPatchModule, (module));

  sanitizer_buffer_init(context);
}


static void
sanitizer_unload_callback
(
 const void *module,
 const void *cubin, 
 size_t cubin_size
)
{
  hpctoolkit_cumod_st_t *cumod = (hpctoolkit_cumod_st_t *)module;
  cuda_unload_callback(cumod->cubin_id);

  // We cannot unregister cubins
  //redshow_cubin_unregister(cumod->cubin_id, cumod->mod_id);
}

//******************************************************************************
// record handlers
//******************************************************************************

static void __attribute__((unused))
dim3_id_transform
(
 dim3 dim,
 uint32_t flat_id,
 uint32_t *id_x,
 uint32_t *id_y,
 uint32_t *id_z
)
{
  *id_z = flat_id / (dim.x * dim.y);
  *id_y = (flat_id - (*id_z) * dim.x * dim.y) / (dim.x);
  *id_x = (flat_id - (*id_z) * dim.x * dim.y - (*id_y) * dim.x);
}


static Sanitizer_StreamHandle
sanitizer_priority_stream_get
(
 CUcontext context
)
{
  sanitizer_context_map_entry_t *entry = sanitizer_context_map_init(context);

  Sanitizer_StreamHandle priority_stream_handle =
    sanitizer_context_map_entry_priority_stream_handle_get(entry);

  if (priority_stream_handle == NULL) {
    // First time
    // Update priority stream
    CUstream priority_stream = sanitizer_context_map_entry_priority_stream_get(entry);
    HPCRUN_SANITIZER_CALL(sanitizerGetStreamHandle, (context, priority_stream, &priority_stream_handle)); 

    sanitizer_context_map_priority_stream_handle_update(context, priority_stream_handle);
  }

  return priority_stream_handle;
}


static Sanitizer_StreamHandle
sanitizer_kernel_stream_get
(
 CUcontext context
)
{
  sanitizer_context_map_entry_t *entry = sanitizer_context_map_init(context);

  Sanitizer_StreamHandle kernel_stream_handle =
    sanitizer_context_map_entry_kernel_stream_handle_get(entry);

  if (kernel_stream_handle == NULL) {
    // First time
    // Update kernel stream
    CUstream kernel_stream = sanitizer_context_map_entry_kernel_stream_get(entry);
    HPCRUN_SANITIZER_CALL(sanitizerGetStreamHandle, (context, kernel_stream, &kernel_stream_handle)); 

    sanitizer_context_map_kernel_stream_handle_update(context, kernel_stream_handle);
  }

  return kernel_stream_handle;
}


static void
sanitizer_module_load
(
 CUcontext context
)
{
  CUmodule analysis_module = NULL;
  CUfunction analysis_function = NULL;
  cuda_module_load(&analysis_module, HPCTOOLKIT_GPU_PATCH "gpu-analysis.fatbin");
  cuda_module_function_get(&analysis_function, analysis_module, "gpu_analysis_interval_merge");
  
  sanitizer_context_map_analysis_function_update(context, analysis_function);
  
  PRINT("Sanitizer-> context %p load function gpu_analysis_interval_merge %p\n", context, analysis_function);
}


// cbid == SANITIZER_CBID_LAUNCH_AFTER_SYSCALL_SETUP
static void
sanitizer_kernel_launch
(
 CUcontext context
)
{
  sanitizer_context_map_entry_t *entry = sanitizer_context_map_init(context);

  // Get raw priority stream and function
  CUfunction analysis_function = sanitizer_context_map_entry_analysis_function_get(entry);
  CUstream kernel_stream = sanitizer_context_map_entry_kernel_stream_get(entry);

  // First time get function
  if (analysis_function == NULL) {
    sanitizer_module_load(context);
  }

  // Launch analysis function
  if (analysis_function != NULL) {
    void *args[] = { (void *)&sanitizer_gpu_patch_buffer_device, (void *)&sanitizer_gpu_patch_buffer_addr_read_device,
      (void *)&sanitizer_gpu_patch_buffer_addr_write_device };

    cuda_kernel_launch(analysis_function, sanitizer_gpu_analysis_blocks, 1, 1,
      GPU_PATCH_ANALYSIS_THREADS, 1, 1, 0, kernel_stream, args);

    PRINT("Sanitizer-> context %p launch function gpu_analysis_interval_merge %p <%u, %u>\n", \
      context, analysis_function, sanitizer_gpu_analysis_blocks, GPU_PATCH_ANALYSIS_THREADS);
  }
}


static void
buffer_analyze
(
 int32_t persistent_id,
 uint64_t correlation_id,
 uint32_t stream_id,
 uint32_t cubin_id,
 uint32_t mod_id,
 uint32_t gpu_patch_type,
 size_t record_size,
 gpu_patch_buffer_t *gpu_patch_buffer_host,
 gpu_patch_buffer_t *gpu_patch_buffer_device,
 Sanitizer_StreamHandle priority_stream
)
{
  sanitizer_buffer_t *sanitizer_buffer = sanitizer_buffer_channel_produce(
    sanitizer_thread_id_local, cubin_id, mod_id, persistent_id, correlation_id, stream_id, gpu_patch_type,
    sanitizer_gpu_patch_record_num, sanitizer_analysis_async);
  gpu_patch_buffer_t *gpu_patch_buffer = sanitizer_buffer_entry_gpu_patch_buffer_get(sanitizer_buffer);

  // If sync mode and not enough buffer, empty current buffer
  if (gpu_patch_buffer == NULL && !sanitizer_analysis_async) {
    sanitizer_buffer_channel_t *channel = sanitizer_buffer_channel_get(gpu_patch_type);
    sanitizer_buffer_channel_consume(channel);
    // Get it again
    sanitizer_buffer = sanitizer_buffer_channel_produce(
      sanitizer_thread_id_local, cubin_id, mod_id, persistent_id, correlation_id, stream_id, gpu_patch_type,
      sanitizer_gpu_patch_record_num, sanitizer_analysis_async);
    gpu_patch_buffer = sanitizer_buffer_entry_gpu_patch_buffer_get(sanitizer_buffer);
  }

  // Move host buffer to a cache
  memcpy(gpu_patch_buffer, gpu_patch_buffer_host, offsetof(gpu_patch_buffer_t, records));

  // Copy all records to the cache
  size_t num_records = gpu_patch_buffer_host->head_index;
  void *gpu_patch_record_device = gpu_patch_buffer_host->records;
  if (num_records != 0) {
    HPCRUN_SANITIZER_CALL(sanitizerMemcpyDeviceToHost,
      (gpu_patch_buffer->records, gpu_patch_record_device, record_size * num_records, priority_stream));
    PRINT("Sanitizer-> copy num_records %zu\n", num_records);
  }

  // Tell kernel to continue
  // Do not need to sync stream.
  // The function will return once the pageable buffer has been copied to the staging memory.
  // for DMA transfer to device memory, but the DMA to final destination may not have completed.
  // Only copy the first field because other fields are being updated by the GPU.
  gpu_patch_buffer_host->full = 0;
  HPCRUN_SANITIZER_CALL(sanitizerMemcpyHostToDeviceAsync, (gpu_patch_buffer_device, gpu_patch_buffer_host,
     sizeof(gpu_patch_buffer_host->full), priority_stream));

  sanitizer_buffer_channel_push(sanitizer_buffer, gpu_patch_type);
}


static void
aux_buffer_analyze
(
 int32_t persistent_id,
 uint64_t correlation_id,
 uint32_t stream_id,
 uint32_t cubin_id,
 uint32_t mod_id,
 uint32_t gpu_patch_type,
 size_t record_size,
 gpu_patch_buffer_t *gpu_patch_buffer_host,
 gpu_patch_buffer_t *gpu_patch_buffer_device,
 Sanitizer_StreamHandle priority_stream
)
{
  sanitizer_buffer_t *sanitizer_buffer = sanitizer_buffer_channel_produce(
    sanitizer_thread_id_local, cubin_id, mod_id, persistent_id, correlation_id, stream_id, gpu_patch_type,
    sanitizer_gpu_patch_record_num, sanitizer_analysis_async);
  gpu_patch_buffer_t *gpu_patch_buffer = sanitizer_buffer_entry_gpu_patch_buffer_get(sanitizer_buffer);


  // Move host buffer to a cache
  memcpy(gpu_patch_buffer, gpu_patch_buffer_host, offsetof(gpu_patch_buffer_t, records));

  // copy back aux buffer
  HPCRUN_SANITIZER_CALL(sanitizerMemcpyDeviceToHost,
      (sanitizer_gpu_patch_aux_addr_dict_host, sanitizer_gpu_patch_buffer_host->aux, \
        sizeof(gpu_patch_aux_address_dict_t), priority_stream));

  if (sanitizer_torch_analysis_ongpu) {
    HPCRUN_SANITIZER_CALL(sanitizerMemcpyDeviceToHost,
      (sanitizer_gpu_patch_torch_aux_addr_dict_host, sanitizer_gpu_patch_buffer_host->torch_aux, \
        sizeof(gpu_patch_aux_address_dict_t), priority_stream));
    gpu_patch_buffer->torch_aux = sanitizer_gpu_patch_torch_aux_addr_dict_host;
  }

  gpu_patch_buffer->aux = sanitizer_gpu_patch_aux_addr_dict_host;

  sanitizer_buffer_channel_push(sanitizer_buffer, gpu_patch_type);

}

static void
sanitizer_kernel_analyze
(
 int32_t persistent_id,
 uint64_t correlation_id,
 uint32_t stream_id,
 uint32_t cubin_id,
 uint32_t mod_id,
 Sanitizer_StreamHandle priority_stream,
 Sanitizer_StreamHandle kernel_stream,
 bool analysis_end
)
{
  if (analysis_end) {
    HPCRUN_SANITIZER_CALL(sanitizerMemcpyDeviceToHost,
      (sanitizer_gpu_patch_buffer_addr_read_host, sanitizer_gpu_patch_buffer_addr_read_device,
       sizeof(gpu_patch_buffer_t), priority_stream));

    HPCRUN_SANITIZER_CALL(sanitizerMemcpyDeviceToHost,
      (sanitizer_gpu_patch_buffer_addr_write_host, sanitizer_gpu_patch_buffer_addr_write_device,
       sizeof(gpu_patch_buffer_t), priority_stream));

    while (sanitizer_gpu_patch_buffer_addr_read_host->num_threads != 0) {
      if (sanitizer_gpu_patch_buffer_addr_read_host->full != 0) {
        PRINT("Sanitizer-> read analysis address\n");
        buffer_analyze(persistent_id, correlation_id, stream_id, cubin_id, mod_id, GPU_PATCH_TYPE_ADDRESS_ANALYSIS,
          sanitizer_gpu_analysis_record_size, sanitizer_gpu_patch_buffer_addr_read_host,
          sanitizer_gpu_patch_buffer_addr_read_device, priority_stream);
      }

      if (sanitizer_gpu_patch_buffer_addr_write_host->full != 0) {
        PRINT("Sanitizer-> write analysis address\n");
        buffer_analyze(persistent_id, correlation_id, stream_id, cubin_id, mod_id, GPU_PATCH_TYPE_ADDRESS_ANALYSIS,
          sanitizer_gpu_analysis_record_size, sanitizer_gpu_patch_buffer_addr_write_host,
          sanitizer_gpu_patch_buffer_addr_write_device, priority_stream);
      }

      HPCRUN_SANITIZER_CALL(sanitizerMemcpyDeviceToHost,
        (sanitizer_gpu_patch_buffer_addr_read_host, sanitizer_gpu_patch_buffer_addr_read_device,
         sizeof(gpu_patch_buffer_t), priority_stream));

      HPCRUN_SANITIZER_CALL(sanitizerMemcpyDeviceToHost,
        (sanitizer_gpu_patch_buffer_addr_write_host, sanitizer_gpu_patch_buffer_addr_write_device,
         sizeof(gpu_patch_buffer_t), priority_stream));
    }

    // To ensure analysis is done
    HPCRUN_SANITIZER_CALL(sanitizerStreamSynchronize, (kernel_stream));

    // Last analysis
    PRINT("Sanitizer-> read analysis address\n");
    buffer_analyze(persistent_id, correlation_id, stream_id, cubin_id, mod_id, GPU_PATCH_TYPE_ADDRESS_ANALYSIS,
      sanitizer_gpu_analysis_record_size, sanitizer_gpu_patch_buffer_addr_read_host,
      sanitizer_gpu_patch_buffer_addr_read_device, priority_stream);

    PRINT("Sanitizer-> write analysis address\n");
    buffer_analyze(persistent_id, correlation_id, stream_id, cubin_id, mod_id, GPU_PATCH_TYPE_ADDRESS_ANALYSIS,
      sanitizer_gpu_analysis_record_size, sanitizer_gpu_patch_buffer_addr_write_host,
      sanitizer_gpu_patch_buffer_addr_write_device, priority_stream);

    // Do not enter later code
    PRINT("Sanitizer-> analysis gpu done\n");

    return;
  }

  HPCRUN_SANITIZER_CALL(sanitizerMemcpyDeviceToHost,
    (sanitizer_gpu_patch_buffer_addr_read_host, sanitizer_gpu_patch_buffer_addr_read_device,
     sizeof(gpu_patch_buffer_t), priority_stream));

  if (sanitizer_gpu_patch_buffer_addr_read_host->full != 0) {
    PRINT("Sanitizer-> read analysis address\n");
    buffer_analyze(persistent_id, correlation_id, stream_id, cubin_id, mod_id, GPU_PATCH_TYPE_ADDRESS_ANALYSIS,
      sanitizer_gpu_analysis_record_size, sanitizer_gpu_patch_buffer_addr_read_host,
      sanitizer_gpu_patch_buffer_addr_read_device, priority_stream);

    PRINT("Sanitizer-> analysis gpu in process\n");
  }

  HPCRUN_SANITIZER_CALL(sanitizerMemcpyDeviceToHost,
    (sanitizer_gpu_patch_buffer_addr_write_host, sanitizer_gpu_patch_buffer_addr_write_device,
     sizeof(gpu_patch_buffer_t), priority_stream));

  if (sanitizer_gpu_patch_buffer_addr_write_host->full != 0) {
    PRINT("Sanitizer-> write analysis address\n");
    buffer_analyze(persistent_id, correlation_id, stream_id, cubin_id, mod_id, GPU_PATCH_TYPE_ADDRESS_ANALYSIS,
      sanitizer_gpu_analysis_record_size, sanitizer_gpu_patch_buffer_addr_write_host,
      sanitizer_gpu_patch_buffer_addr_write_device, priority_stream);

    PRINT("Sanitizer-> analysis gpu in process\n");
  }
}

// cbid == SANITIZER_CBID_LAUNCH_END
static void
sanitizer_kernel_launch_sync
(
 int32_t persistent_id,
 uint64_t correlation_id,
 uint32_t stream_id,
 CUcontext context,
 CUmodule module,
 CUfunction function,
 Sanitizer_StreamHandle priority_stream,
 Sanitizer_StreamHandle kernel_stream,
 dim3 grid_size,
 dim3 block_size
)
{
  // Look up module id
  hpctoolkit_cumod_st_t *cumod = (hpctoolkit_cumod_st_t *)module;
  uint32_t cubin_id = cumod->cubin_id;
  uint32_t mod_id = cumod->mod_id;

  // TODO(Keren): correlate metrics with api_node

  int block_sampling_frequency = sanitizer_block_sampling_frequency_get();
  int grid_dim = grid_size.x * grid_size.y * grid_size.z;
  int block_dim = block_size.x * block_size.y * block_size.z;
  uint64_t num_threads = grid_dim * block_dim;
  size_t num_left_threads = 0;

  // If block sampling is set
  if (block_sampling_frequency != 0) {
    // Uniform sampling
    int sampling_offset = sanitizer_gpu_patch_buffer_reset->block_sampling_offset;
    int mod_blocks = grid_dim % block_sampling_frequency;
    int sampling_blocks = 0;
    if (mod_blocks == 0) {
      sampling_blocks = (grid_dim - 1) / block_sampling_frequency + 1;
    } else {
      sampling_blocks = (grid_dim - 1) / block_sampling_frequency + (sampling_offset >= mod_blocks ? 0 : 1);
    }
    num_left_threads = num_threads - sampling_blocks * block_dim;
  }

  // Init a buffer on host
  if (sanitizer_gpu_patch_buffer_host == NULL) {
    sanitizer_gpu_patch_buffer_host = (gpu_patch_buffer_t *)hpcrun_malloc_safe(sizeof(gpu_patch_buffer_t));

    if (sanitizer_gpu_analysis_blocks != 0) {
      sanitizer_gpu_patch_buffer_addr_read_host = (gpu_patch_buffer_t *)hpcrun_malloc_safe(sizeof(gpu_patch_buffer_t));
      sanitizer_gpu_patch_buffer_addr_write_host = (gpu_patch_buffer_t *)hpcrun_malloc_safe(sizeof(gpu_patch_buffer_t));
    }
  }

  // Reserve for debugging correctness
  //PRINT("head_index %u, tail_index %u, num_left_threads %lu\n",
  //  sanitizer_gpu_patch_buffer_host->head_index, sanitizer_gpu_patch_buffer_host->tail_index, num_threads);

  while (true) {
    // Copy buffer
    HPCRUN_SANITIZER_CALL(sanitizerMemcpyDeviceToHost,
      (sanitizer_gpu_patch_buffer_host, sanitizer_gpu_patch_buffer_device, sizeof(gpu_patch_buffer_t), priority_stream));

    // size_t num_records = sanitizer_gpu_patch_buffer_host->head_index;

    // Reserve for debugging correctness
    // PRINT("head_index %u, tail_index %u, num_left_threads %u expected %zu\n",
    //   sanitizer_gpu_patch_buffer_host->head_index, sanitizer_gpu_patch_buffer_host->tail_index,
    //   sanitizer_gpu_patch_buffer_host->num_threads, num_left_threads);

    if (sanitizer_gpu_analysis_blocks != 0) {
      sanitizer_kernel_analyze(persistent_id, correlation_id, stream_id, cubin_id, mod_id, priority_stream,
                              kernel_stream, false);
    }

    // Wait until the buffer is full or the kernel is finished
    if (!(sanitizer_gpu_patch_buffer_host->num_threads == num_left_threads || sanitizer_gpu_patch_buffer_host->full)) {
      continue;
    }

    // Reserve for debugging correctness
    // PRINT("num_records %zu\n", num_records);

    if (sanitizer_gpu_analysis_blocks == 0 && !(sanitizer_liveness_ongpu || sanitizer_torch_analysis_ongpu)) {
      buffer_analyze(persistent_id, correlation_id, stream_id, cubin_id, mod_id, sanitizer_gpu_patch_type,
        sanitizer_gpu_patch_record_size, sanitizer_gpu_patch_buffer_host,
        sanitizer_gpu_patch_buffer_device, priority_stream);

      PRINT("Sanitizer-> analysis cpu in process\n");
    }

    // Awake background thread
    if (sanitizer_analysis_async) {
      // If multiple application threads are created, it might miss a signal,
      // but we finally still process all the records
      sanitizer_process_signal(); 
    }
    
    // Finish all the threads
    if (sanitizer_gpu_patch_buffer_host->num_threads == num_left_threads) {
      break;
    }
  }

  if (sanitizer_liveness_ongpu || sanitizer_torch_analysis_ongpu) {
    aux_buffer_analyze(persistent_id, correlation_id, stream_id, cubin_id, mod_id, sanitizer_gpu_patch_type,
        sanitizer_gpu_patch_record_size, sanitizer_gpu_patch_buffer_host,
        sanitizer_gpu_patch_buffer_device, priority_stream);
    PRINT("Sanitizer-> analysis aux for liveness\n");
  }

  if (sanitizer_gpu_analysis_blocks != 0) {
    // Kernel is done
    sanitizer_kernel_analyze(persistent_id, correlation_id, stream_id, cubin_id, mod_id,
                            priority_stream, kernel_stream, true);
  }

  // To ensure previous copies are done
  HPCRUN_SANITIZER_CALL(sanitizerStreamSynchronize, (priority_stream));

// sync will comsume buffer here
  if (!sanitizer_analysis_async) {
    // Empty current buffer
    sanitizer_buffer_channel_t *channel = sanitizer_buffer_channel_get(sanitizer_gpu_patch_type);
    sanitizer_buffer_channel_consume(channel);

    if (sanitizer_gpu_analysis_blocks != 0) {
      channel = sanitizer_buffer_channel_get(sanitizer_gpu_analysis_type);
      sanitizer_buffer_channel_consume(channel);
    }
  }
}

//******************************************************************************
// callbacks
//******************************************************************************

// add for sub-allocation callback in memory profile
// @Lin-Mao: never used, can be deleted
void memory_suballoc_callback (void *ptr, size_t size) {
    
    uint64_t correlation_id = gpu_correlation_id();
    cct_node_t *api_node = sanitizer_correlation_callback(correlation_id, 0);
    hpcrun_cct_retain(api_node);

    int32_t persistent_id = hpcrun_cct_persistent_id(api_node);

    // redshow_sub_memory_register(persistent_id, correlation_id, ptr, ptr + size);

    PRINT("Sanitizer-> Sub-allocte memory address %p, size %zu, op %lu, id %d\n",
        ptr, size, correlation_id, persistent_id);
        
}

// cbid == SANITIZER_CBID_LAUNCH_BEGIN
static void
sanitizer_kernel_launch_callback
(
 uint64_t correlation_id,
 CUcontext context,
 Sanitizer_StreamHandle priority_stream,
 CUfunction function,
 dim3 grid_size,
 dim3 block_size,
 bool kernel_sampling
)
{
  int grid_dim = grid_size.x * grid_size.y * grid_size.z;
  int block_dim = block_size.x * block_size.y * block_size.z;
  int block_sampling_frequency = kernel_sampling ? sanitizer_block_sampling_frequency_get() : 0;
  int block_sampling_offset = kernel_sampling ? rand() % grid_dim % block_sampling_frequency : 0;

  PRINT("Sanitizer-> kernel sampling %d\n", kernel_sampling);
  PRINT("Sanitizer-> sampling offset %d\n", block_sampling_offset);
  PRINT("Sanitizer-> sampling frequency %d\n", block_sampling_frequency);

  // Get cached entries, already locked
  sanitizer_buffer_init(context);

  // reset buffer
  sanitizer_gpu_patch_buffer_reset->num_threads = grid_dim * block_dim;
  sanitizer_gpu_patch_buffer_reset->block_sampling_frequency = block_sampling_frequency;
  sanitizer_gpu_patch_buffer_reset->block_sampling_offset = block_sampling_offset;

  HPCRUN_SANITIZER_CALL(sanitizerMemcpyHostToDeviceAsync,
    (sanitizer_gpu_patch_buffer_device, sanitizer_gpu_patch_buffer_reset,
     sizeof(gpu_patch_buffer_t), priority_stream));

  if (sanitizer_gpu_analysis_blocks != 0) {
    HPCRUN_SANITIZER_CALL(sanitizerMemcpyHostToDeviceAsync, (sanitizer_gpu_patch_buffer_addr_read_device,
        sanitizer_gpu_patch_buffer_addr_read_reset, sizeof(gpu_patch_buffer_t), priority_stream));
    HPCRUN_SANITIZER_CALL(sanitizerMemcpyHostToDeviceAsync, (sanitizer_gpu_patch_buffer_addr_write_device,
        sanitizer_gpu_patch_buffer_addr_write_reset, sizeof(gpu_patch_buffer_t), priority_stream));
  }

  if (sanitizer_read_trace_ignore) {
    if (sanitizer_gpu_patch_aux_addr_dict_host == NULL) {
      sanitizer_gpu_patch_aux_addr_dict_host = (gpu_patch_aux_address_dict_t *)
        hpcrun_malloc_safe(sizeof(gpu_patch_aux_address_dict_t));
    }
    memset(sanitizer_gpu_patch_aux_addr_dict_host->hit, 0, sizeof(uint8_t) * GPU_PATCH_ADDRESS_DICT_SIZE);

    // Get memory ranges from redshow
    uint64_t limit = GPU_PATCH_ADDRESS_DICT_SIZE;
    redshow_memory_ranges_get(correlation_id, limit, sanitizer_gpu_patch_aux_addr_dict_host->start_end,
      &sanitizer_gpu_patch_aux_addr_dict_host->size);
    // Copy
    HPCRUN_SANITIZER_CALL(sanitizerMemcpyHostToDeviceAsync,
      (sanitizer_gpu_patch_buffer_reset->aux, sanitizer_gpu_patch_aux_addr_dict_host,
       sizeof(gpu_patch_aux_address_dict_t), priority_stream));
  }

  // initialize the aux
  if (sanitizer_liveness_ongpu) {
    if (sanitizer_gpu_patch_aux_addr_dict_host == NULL) {
      sanitizer_gpu_patch_aux_addr_dict_host = (gpu_patch_aux_address_dict_t *)
        hpcrun_malloc_safe(sizeof(gpu_patch_aux_address_dict_t));
    }
    memset(sanitizer_gpu_patch_aux_addr_dict_host->hit, 0, sizeof(uint8_t) * GPU_PATCH_ADDRESS_DICT_SIZE);
    memset(sanitizer_gpu_patch_aux_addr_dict_host->read, 0, sizeof(uint8_t) * GPU_PATCH_ADDRESS_DICT_SIZE);
    memset(sanitizer_gpu_patch_aux_addr_dict_host->write, 0, sizeof(uint8_t) * GPU_PATCH_ADDRESS_DICT_SIZE);

    // Get memory ranges from redshow
    uint64_t limit = GPU_PATCH_ADDRESS_DICT_SIZE;
    redshow_memory_ranges_get(correlation_id, limit, sanitizer_gpu_patch_aux_addr_dict_host->start_end,
      &sanitizer_gpu_patch_aux_addr_dict_host->size);
    // Copy
    HPCRUN_SANITIZER_CALL(sanitizerMemcpyHostToDeviceAsync,
      (sanitizer_gpu_patch_buffer_reset->aux, sanitizer_gpu_patch_aux_addr_dict_host,
       sizeof(gpu_patch_aux_address_dict_t), priority_stream));
  }

  if (sanitizer_torch_analysis_ongpu) {
    if (sanitizer_gpu_patch_aux_addr_dict_host == NULL) {
      sanitizer_gpu_patch_aux_addr_dict_host = (gpu_patch_aux_address_dict_t *)
        hpcrun_malloc_safe(sizeof(gpu_patch_aux_address_dict_t));
    }
    memset(sanitizer_gpu_patch_aux_addr_dict_host->hit, 0, sizeof(uint8_t) * GPU_PATCH_ADDRESS_DICT_SIZE);

    // Get memory ranges from redshow
    uint64_t limit = GPU_PATCH_ADDRESS_DICT_SIZE;
    redshow_memory_ranges_get(correlation_id, limit, sanitizer_gpu_patch_aux_addr_dict_host->start_end,
      &sanitizer_gpu_patch_aux_addr_dict_host->size);
    // Copy
    HPCRUN_SANITIZER_CALL(sanitizerMemcpyHostToDeviceAsync,
      (sanitizer_gpu_patch_buffer_reset->aux, sanitizer_gpu_patch_aux_addr_dict_host,
       sizeof(gpu_patch_aux_address_dict_t), priority_stream));

    if (sanitizer_gpu_patch_torch_aux_addr_dict_host == NULL) {
      sanitizer_gpu_patch_torch_aux_addr_dict_host = (gpu_patch_aux_address_dict_t *)
        hpcrun_malloc_safe(sizeof(gpu_patch_aux_address_dict_t));
    }
    memset(sanitizer_gpu_patch_torch_aux_addr_dict_host->hit, 0, sizeof(uint8_t) * GPU_PATCH_ADDRESS_DICT_SIZE);
    // Get submemory ranges from redshow
    redshow_submemory_ranges_get(correlation_id, limit, sanitizer_gpu_patch_torch_aux_addr_dict_host->start_end,
      &sanitizer_gpu_patch_torch_aux_addr_dict_host->size);

    // Copy
    HPCRUN_SANITIZER_CALL(sanitizerMemcpyHostToDeviceAsync,
      (sanitizer_gpu_patch_buffer_reset->torch_aux, sanitizer_gpu_patch_torch_aux_addr_dict_host,
       sizeof(gpu_patch_aux_address_dict_t), priority_stream));

  }

  HPCRUN_SANITIZER_CALL(sanitizerSetCallbackData, (function, sanitizer_gpu_patch_buffer_device));

  HPCRUN_SANITIZER_CALL(sanitizerStreamSynchronize, (priority_stream));
}

//-------------------------------------------------------------
// callback controls
//-------------------------------------------------------------

static void
sanitizer_subscribe_callback
(
 void* userdata,
 Sanitizer_CallbackDomain domain,
 Sanitizer_CallbackId cbid,
 const void* cbdata
)
{
  if (cuda_api_internal()) {
    return;
  }

  if (!sanitizer_stop_flag) {
    sanitizer_thread_id_local = atomic_fetch_add(&sanitizer_thread_id, 1);
    sanitizer_stop_flag_set();
  }

  if (domain == SANITIZER_CB_DOMAIN_DRIVER_API) {
    Sanitizer_CallbackData *cb = (Sanitizer_CallbackData *)cbdata;
    if (cb->callbackSite == SANITIZER_API_ENTER) {
      // Reserve for debug
      //PRINT("Sanitizer-> Thread %u enter context %p function %s\n", sanitizer_thread_id_local, cb->context, cb->functionName);
      sanitizer_context_map_context_lock(cb->context, sanitizer_thread_id_local);
      sanitizer_thread_context = cb->context;
    } else {
      // Reserve for debug
      //PRINT("Sanitizer-> Thread %u exit context %p function %s\n", sanitizer_thread_id_local, cb->context, cb->functionName);
      // Caution, do not use cb->context. When cuCtxGetCurrent is used, cb->context != sanitizer_thread_context
      sanitizer_context_map_context_unlock(sanitizer_thread_context, sanitizer_thread_id_local);
      sanitizer_thread_context = NULL;
    }

    return;
  }

  // XXX(keren): assume single thread per stream
  if (domain == SANITIZER_CB_DOMAIN_RESOURCE) {
    switch (cbid) {
      case SANITIZER_CBID_RESOURCE_MODULE_LOADED:
        {
          // single thread
          Sanitizer_ResourceModuleData *md = (Sanitizer_ResourceModuleData *)cbdata;

          sanitizer_load_callback(md->context, md->module, md->pCubin, md->cubinSize);
          break;
        }
      case SANITIZER_CBID_RESOURCE_MODULE_UNLOAD_STARTING:
        {
          // single thread
          Sanitizer_ResourceModuleData *md = (Sanitizer_ResourceModuleData *)cbdata;

          sanitizer_unload_callback(md->module, md->pCubin, md->cubinSize);
          break;
        }
      case SANITIZER_CBID_RESOURCE_STREAM_CREATED:
        {
          // single thread
          PRINT("Sanitizer-> Stream create starting\n");
          break;
        }
      case SANITIZER_CBID_RESOURCE_STREAM_DESTROY_STARTING:
        {
          // single thread
          // TODO
          PRINT("Sanitizer-> Stream destroy starting\n");
          break;
        }
      case SANITIZER_CBID_RESOURCE_CONTEXT_CREATION_STARTING:
        {
          PRINT("Sanitizer-> Context creation starting\n");
	  
	  sanitizer_context_creation_flag = true;
          break;
        }
      case SANITIZER_CBID_RESOURCE_CONTEXT_CREATION_FINISHED:
        {
          PRINT("Sanitizer-> Context creation finished\n");
	  
	  sanitizer_context_creation_flag = false;
	  sanitizer_priority_stream_get(sanitizer_thread_context);
	  sanitizer_kernel_stream_get(sanitizer_thread_context);
	  sanitizer_module_load(sanitizer_thread_context);
// @Lin-Mao: might be the extra 512B memory, comment but may incur problems!!!
          if (sanitizer_memory_register_delegate.flag) {
            // redshow_memory_register(
            //   0,
            //   sanitizer_memory_register_delegate.persistent_id,
            //   sanitizer_memory_register_delegate.correlation_id,
            //   sanitizer_memory_register_delegate.start,
            //   sanitizer_memory_register_delegate.end);
            sanitizer_memory_register_delegate.flag = false;
          }
          break;
        }
      case SANITIZER_CBID_RESOURCE_CONTEXT_DESTROY_STARTING:
        {
          // single thread
          // TODO
          PRINT("Sanitizer-> Context destroy starting\n");
          break;
        }
      case SANITIZER_CBID_RESOURCE_HOST_MEMORY_ALLOC:
        {
          Sanitizer_ResourceMemoryData *md = (Sanitizer_ResourceMemoryData *)cbdata;

          PRINT("Sanitizer-> Allocate memory address %p, size %zu\n", (void *)md->address, md->size);

          break;
        }
      case SANITIZER_CBID_RESOURCE_HOST_MEMORY_FREE:
        {
          Sanitizer_ResourceMemoryData *md = (Sanitizer_ResourceMemoryData *)cbdata;

          PRINT("Sanitizer-> Free memory address %p, size %zu\n", (void *)md->address, md->size);

          break;
        }
      case SANITIZER_CBID_RESOURCE_DEVICE_MEMORY_ALLOC:
        {
          Sanitizer_ResourceMemoryData *md = (Sanitizer_ResourceMemoryData *)cbdata;

          uint64_t correlation_id = gpu_correlation_id();
          cct_node_t *api_node = sanitizer_correlation_callback(correlation_id, 0);
          hpcrun_cct_retain(api_node);

          hpcrun_safe_enter();

          gpu_op_ccts_t gpu_op_ccts;
          gpu_op_placeholder_flags_t gpu_op_placeholder_flags = 0;
          gpu_op_placeholder_flags_set(&gpu_op_placeholder_flags, 
            gpu_placeholder_type_alloc);
          gpu_op_ccts_insert(api_node, &gpu_op_ccts, gpu_op_placeholder_flags);
          api_node = gpu_op_ccts_get(&gpu_op_ccts, gpu_placeholder_type_alloc);
          hpcrun_cct_retain(api_node);

          hpcrun_safe_exit();

          int32_t persistent_id = hpcrun_cct_persistent_id(api_node);

          uint32_t stream_id = sanitizer_stearm_id_query(md->stream);

          if (sanitizer_context_creation_flag) {
            // For some driver versions, the primary context is not fully initialized here.
	    // So we have to delay memory register to the point when context initialization is done.
	    //
	    // CUDA context is often initalized lazily.
	    // The primary context is only initialized when seeing cudaDeviceReset or the first CUDA
	    // runtime API (e.g., cudaMemalloc).
            sanitizer_memory_register_delegate.flag = true;
            sanitizer_memory_register_delegate.persistent_id = persistent_id;
            sanitizer_memory_register_delegate.correlation_id = correlation_id;
            sanitizer_memory_register_delegate.start = md->address;
            sanitizer_memory_register_delegate.end = md->address + md->size;
          } else {
            redshow_memory_register(stream_id, persistent_id, correlation_id, md->address, md->address + md->size);
          }

          PRINT("Sanitizer-> Allocate memory address %p, size %zu, op %lu, id %d\n",
            (void *)md->address, md->size, correlation_id, persistent_id);

          break;
        }
      case SANITIZER_CBID_RESOURCE_DEVICE_MEMORY_FREE:
        {
          Sanitizer_ResourceMemoryData *md = (Sanitizer_ResourceMemoryData *)cbdata;
          
          uint64_t correlation_id = gpu_correlation_id();
          cct_node_t *api_node = sanitizer_correlation_callback(correlation_id, 0);
          hpcrun_cct_retain(api_node);

          hpcrun_safe_enter();

          gpu_op_ccts_t gpu_op_ccts;
          gpu_op_placeholder_flags_t gpu_op_placeholder_flags = 0;
          gpu_op_placeholder_flags_set(&gpu_op_placeholder_flags, gpu_placeholder_type_alloc);
          gpu_op_ccts_insert(api_node, &gpu_op_ccts, gpu_op_placeholder_flags);
          api_node = gpu_op_ccts_get(&gpu_op_ccts, gpu_placeholder_type_alloc);
          hpcrun_cct_retain(api_node);

          hpcrun_safe_exit();

          int32_t persistent_id = hpcrun_cct_persistent_id(api_node);

          uint32_t stream_id = sanitizer_stearm_id_query(md->stream);

          redshow_memory_unregister(stream_id, persistent_id, correlation_id, md->address, md->address + md->size);

          PRINT("Sanitizer-> Free memory address %p, size %zu, op %lu\n",
                (void *)md->address, md->size, correlation_id);

          break;
        }
      case SANITIZER_CBID_RESOURCE_MEMORY_ALLOC_ASYNC:
      {
        Sanitizer_ResourceMemoryData *md = (Sanitizer_ResourceMemoryData *)cbdata;

        uint64_t correlation_id = gpu_correlation_id();
        cct_node_t *api_node = sanitizer_correlation_callback(correlation_id, 0);
        hpcrun_cct_retain(api_node);

        hpcrun_safe_enter();

        gpu_op_ccts_t gpu_op_ccts;
        gpu_op_placeholder_flags_t gpu_op_placeholder_flags = 0;
        gpu_op_placeholder_flags_set(&gpu_op_placeholder_flags, 
          gpu_placeholder_type_alloc);
        gpu_op_ccts_insert(api_node, &gpu_op_ccts, gpu_op_placeholder_flags);
        api_node = gpu_op_ccts_get(&gpu_op_ccts, gpu_placeholder_type_alloc);
        hpcrun_cct_retain(api_node);

        hpcrun_safe_exit();

        int32_t persistent_id = hpcrun_cct_persistent_id(api_node);

        uint32_t stream_id = sanitizer_stearm_id_query(md->stream);

        redshow_memory_register(stream_id, persistent_id, correlation_id, md->address, md->address + md->size);

        PRINT("Sanitizer-> Allocate memory address %p, size %zu, op %lu, id %d\n",
          (void *)md->address, md->size, correlation_id, persistent_id);

        break;
      }
      case SANITIZER_CBID_RESOURCE_MEMORY_FREE_ASYNC:
      {
        Sanitizer_ResourceMemoryData *md = (Sanitizer_ResourceMemoryData *)cbdata;
        
        uint64_t correlation_id = gpu_correlation_id();
        cct_node_t *api_node = sanitizer_correlation_callback(correlation_id, 0);
        hpcrun_cct_retain(api_node);

        hpcrun_safe_enter();

        gpu_op_ccts_t gpu_op_ccts;
        gpu_op_placeholder_flags_t gpu_op_placeholder_flags = 0;
        gpu_op_placeholder_flags_set(&gpu_op_placeholder_flags, gpu_placeholder_type_alloc);
        gpu_op_ccts_insert(api_node, &gpu_op_ccts, gpu_op_placeholder_flags);
        api_node = gpu_op_ccts_get(&gpu_op_ccts, gpu_placeholder_type_alloc);
        hpcrun_cct_retain(api_node);

        hpcrun_safe_exit();

        int32_t persistent_id = hpcrun_cct_persistent_id(api_node);

        uint32_t stream_id = sanitizer_stearm_id_query(md->stream);

        redshow_memory_unregister(stream_id, persistent_id, correlation_id, md->address, md->address + md->size);

        PRINT("Sanitizer-> Free memory address %p, size %zu, op %lu\n",
              (void *)md->address, md->size, correlation_id);
        break;
      }

      default:
        {
          break;
        }
    }
  } else if (domain == SANITIZER_CB_DOMAIN_LAUNCH) {
    Sanitizer_LaunchData *ld = (Sanitizer_LaunchData *)cbdata;

    static __thread dim3 grid_size = { .x = 0, .y = 0, .z = 0};
    static __thread dim3 block_size = { .x = 0, .y = 0, .z = 0};
    static __thread Sanitizer_StreamHandle priority_stream = NULL;
    static __thread Sanitizer_StreamHandle kernel_stream = NULL;
    static __thread bool kernel_sampling = true;
    static __thread uint64_t correlation_id = 0;
    static __thread int32_t persistent_id = 0;

    if (cbid == SANITIZER_CBID_LAUNCH_BEGIN) {
      // Use function list to filter functions
      if (sanitizer_whitelist != NULL) {
        if (sanitizer_function_list_lookup(sanitizer_whitelist, ld->functionName) == NULL) {
          kernel_sampling = false;
        }
      }

      if (sanitizer_blacklist != NULL) {
        if (sanitizer_function_list_lookup(sanitizer_blacklist, ld->functionName) != NULL) {
          kernel_sampling = false;
        }
      }

      // Get a place holder cct node
      correlation_id = gpu_correlation_id();
      // TODO(Keren): why two extra layers?
      cct_node_t *api_node = sanitizer_correlation_callback(correlation_id, 0);
      hpcrun_cct_retain(api_node);

      // Insert a function cct node
      hpcrun_safe_enter();

      gpu_op_ccts_t gpu_op_ccts;
      gpu_op_placeholder_flags_t gpu_op_placeholder_flags = 0;
      gpu_op_placeholder_flags_set(&gpu_op_placeholder_flags, 
        gpu_placeholder_type_kernel);
      gpu_op_ccts_insert(api_node, &gpu_op_ccts, gpu_op_placeholder_flags);
      api_node = gpu_op_ccts_get(&gpu_op_ccts, gpu_placeholder_type_kernel);
      hpcrun_cct_retain(api_node);

      hpcrun_safe_exit();

      // Look up persisitent id
      persistent_id = hpcrun_cct_persistent_id(api_node);
      
      if (kernel_sampling) {
        // Kernel is not ignored
        // By default ignored this kernel
        kernel_sampling = false;
        int kernel_sampling_frequency = sanitizer_kernel_sampling_frequency_get();
        // TODO(Keren): thread safe rand
        kernel_sampling = rand() % kernel_sampling_frequency == 0;

        // First time must be sampled
        if (sanitizer_op_map_lookup(persistent_id) == NULL) {
          kernel_sampling = true;
          sanitizer_op_map_init(persistent_id, api_node);
        }
      }

      grid_size.x = ld->gridDim_x;
      grid_size.y = ld->gridDim_y;
      grid_size.z = ld->gridDim_z;
      block_size.x = ld->blockDim_x;
      block_size.y = ld->blockDim_y;
      block_size.z = ld->blockDim_z;

      PRINT("Sanitizer-> Launch kernel %s <%d, %d, %d>:<%d, %d, %d>, op %lu, id %d, mod_id %u\n", ld->functionName,
        ld->gridDim_x, ld->gridDim_y, ld->gridDim_z, ld->blockDim_x, ld->blockDim_y, ld->blockDim_z,
        correlation_id, persistent_id, ((hpctoolkit_cumod_st_t *)ld->module)->mod_id);

      // thread-safe
      // Create a high priority stream for the context at the first time
      // TODO(Keren): change stream->hstream
      redshow_kernel_begin(sanitizer_thread_id_local, persistent_id, correlation_id);

      priority_stream = sanitizer_priority_stream_get(ld->context);

      sanitizer_kernel_launch_callback(correlation_id, ld->context, priority_stream, ld->function,
        grid_size, block_size, kernel_sampling);
    } else if (cbid == SANITIZER_CBID_LAUNCH_AFTER_SYSCALL_SETUP) {
      if (sanitizer_gpu_analysis_blocks != 0 && kernel_sampling) {
        sanitizer_kernel_launch(ld->context);
      }
    } else if (cbid == SANITIZER_CBID_LAUNCH_END) {
      uint32_t stream_id = sanitizer_stearm_id_query(ld->stream);

      if (kernel_sampling) {
        PRINT("Sanitizer-> Sync kernel %s\n", ld->functionName);

        kernel_stream = sanitizer_kernel_stream_get(ld->context);

        sanitizer_kernel_launch_sync(persistent_id, correlation_id, stream_id,
          ld->context, ld->module, ld->function, priority_stream,
          kernel_stream, grid_size, block_size);
      }

      // NOTICE: Need to synchronize this stream even when this kernel is not sampled.
      // TO prevent data is incorrectly copied in the next round
      HPCRUN_SANITIZER_CALL(sanitizerStreamSynchronize, (ld->hStream));

      redshow_kernel_end(sanitizer_thread_id_local, stream_id, persistent_id, correlation_id);

      kernel_sampling = true;

      PRINT("Sanitizer-> kernel %s done\n", ld->functionName);
    }
  } else if (domain == SANITIZER_CB_DOMAIN_MEMCPY) {
    Sanitizer_MemcpyData *md = (Sanitizer_MemcpyData *)cbdata;

    bool src_host = false;
    bool dst_host = false;

    if (md->direction == SANITIZER_MEMCPY_DIRECTION_HOST_TO_DEVICE) {
      src_host = true;
    } else if (md->direction == SANITIZER_MEMCPY_DIRECTION_HOST_TO_HOST) {
      src_host = true;
      dst_host = true;
    } else if (md->direction == SANITIZER_MEMCPY_DIRECTION_DEVICE_TO_HOST) {
      dst_host = true;
    }

    uint64_t correlation_id = gpu_correlation_id();
    cct_node_t *api_node = sanitizer_correlation_callback(correlation_id, 0);
    hpcrun_cct_retain(api_node);

    int32_t persistent_id = hpcrun_cct_persistent_id(api_node);
    if (sanitizer_op_map_lookup(persistent_id) == NULL) {
      sanitizer_op_map_init(persistent_id, api_node);
    }

    PRINT("Sanitizer-> Memcpy async %d direction %d from %p to %p, op %lu, id %d\n", md->isAsync, md->direction,
      (void *)md->srcAddress, (void *)md->dstAddress, correlation_id, persistent_id);

    uint32_t src_stream_id = sanitizer_stearm_id_query(md->srcStream);
    uint32_t dst_stream_id = sanitizer_stearm_id_query(md->dstStream);

    // Avoid memcpy to symbol without allocation
    // Let redshow update shadow memory
    redshow_memcpy_register(persistent_id, correlation_id, src_host, src_stream_id, md->srcAddress,
      dst_host, dst_stream_id, md->dstAddress, md->size);
  } else if (domain == SANITIZER_CB_DOMAIN_MEMSET) {
    Sanitizer_MemsetData *md = (Sanitizer_MemsetData *)cbdata;

    uint64_t correlation_id = gpu_correlation_id();
    cct_node_t *api_node = sanitizer_correlation_callback(correlation_id, 0);
    hpcrun_cct_retain(api_node);

    // Let redshow update shadow
    int32_t persistent_id = hpcrun_cct_persistent_id(api_node);
    
    uint32_t stream_id = sanitizer_stearm_id_query(md->stream);

    redshow_memset_register(stream_id, persistent_id, correlation_id, md->address, md->value, md->width);
  } else if (domain == SANITIZER_CB_DOMAIN_SYNCHRONIZE) {
    // TODO(Keren): sync data
  }
}

//******************************************************************************
// interfaces
//******************************************************************************

int
sanitizer_bind()
{
#ifndef HPCRUN_STATIC_LINK
  // dynamic libraries only availabile in non-static case
  hpcrun_force_dlopen(true);
  CHK_DLOPEN(sanitizer, sanitizer_path(), RTLD_NOW | RTLD_GLOBAL);
  hpcrun_force_dlopen(false);

#define SANITIZER_BIND(fn) \
  CHK_DLSYM(sanitizer, fn);

  FORALL_SANITIZER_ROUTINES(SANITIZER_BIND)

#undef SANITIZER_BIND

  return 0;
#else
  return -1;
#endif // ! HPCRUN_STATIC_LINK
}


static void
output_dir_config(char *dir_name, char *suffix) {
  size_t used = 0;
  used += sprintf(&dir_name[used], "%s", hpcrun_files_output_directory());
  used += sprintf(&dir_name[used], "%s", suffix);
  mkdir(dir_name, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}


void
sanitizer_redundancy_analysis_enable()
{
  redshow_analysis_enable(REDSHOW_ANALYSIS_SPATIAL_REDUNDANCY);
  redshow_analysis_enable(REDSHOW_ANALYSIS_TEMPORAL_REDUNDANCY);

  char dir_name[PATH_MAX];
  output_dir_config(dir_name, "/redundancy/");

  redshow_output_dir_config(REDSHOW_ANALYSIS_SPATIAL_REDUNDANCY, dir_name);
  redshow_output_dir_config(REDSHOW_ANALYSIS_TEMPORAL_REDUNDANCY, dir_name);

  sanitizer_gpu_patch_record_size = sizeof(gpu_patch_record_t);
}


void
sanitizer_data_flow_analysis_enable()
{
  redshow_analysis_enable(REDSHOW_ANALYSIS_DATA_FLOW);
  // XXX(Keren): value flow analysis must be sync
  sanitizer_analysis_async = false;

  char dir_name[PATH_MAX];
  output_dir_config(dir_name, "/data_flow/");

  redshow_output_dir_config(REDSHOW_ANALYSIS_DATA_FLOW, dir_name);
  redshow_analysis_config(REDSHOW_ANALYSIS_DATA_FLOW, REDSHOW_ANALYSIS_DATA_FLOW_HASH, sanitizer_data_flow_hash);
  redshow_analysis_config(REDSHOW_ANALYSIS_DATA_FLOW, REDSHOW_ANALYSIS_READ_TRACE_IGNORE, sanitizer_read_trace_ignore);

  sanitizer_gpu_patch_type = GPU_PATCH_TYPE_ADDRESS_PATCH;
  sanitizer_gpu_patch_record_size = sizeof(gpu_patch_record_address_t);
  sanitizer_gpu_analysis_type = GPU_PATCH_TYPE_ADDRESS_ANALYSIS;
  sanitizer_gpu_analysis_record_size = sizeof(gpu_patch_analysis_address_t);
}


void
sanitizer_value_pattern_analysis_enable()
{
  redshow_analysis_enable(REDSHOW_ANALYSIS_VALUE_PATTERN);

  char dir_name[PATH_MAX];
  output_dir_config(dir_name, "/value_pattern/");

  redshow_output_dir_config(REDSHOW_ANALYSIS_VALUE_PATTERN, dir_name);

  sanitizer_gpu_patch_record_size = sizeof(gpu_patch_record_t);
}

// @Lin-Mao: New mode enable in sanitizer-api.c.
void
sanitizer_memory_profile_analysis_enable()
{
  redshow_analysis_enable(REDSHOW_ANALYSIS_MEMORY_PROFILE);

  char dir_name[PATH_MAX];
  output_dir_config(dir_name, "/memory_profile/");

  redshow_output_dir_config(REDSHOW_ANALYSIS_MEMORY_PROFILE, dir_name);

  sanitizer_gpu_patch_type = GPU_PATCH_TYPE_ADDRESS_PATCH;
  sanitizer_gpu_patch_record_size = sizeof(gpu_patch_record_address_t);
  sanitizer_gpu_analysis_type = GPU_PATCH_TYPE_ADDRESS_ANALYSIS;
  sanitizer_gpu_analysis_record_size = sizeof(gpu_patch_analysis_address_t);

}

// @Lin-Mao: New mode enable in sanitizer-api.c.
void
sanitizer_memory_heatmap_analysis_enable()
{
  redshow_analysis_enable(REDSHOW_ANALYSIS_MEMORY_HEATMAP);

  char dir_name[PATH_MAX];
  output_dir_config(dir_name, "/memory_heatmap/");

  redshow_output_dir_config(REDSHOW_ANALYSIS_MEMORY_HEATMAP, dir_name);

  sanitizer_gpu_patch_type = GPU_PATCH_TYPE_ADDRESS_PATCH;
  sanitizer_gpu_patch_record_size = sizeof(gpu_patch_record_address_t);
  sanitizer_gpu_analysis_type = GPU_PATCH_TYPE_ADDRESS_ANALYSIS;
  sanitizer_gpu_analysis_record_size = sizeof(gpu_patch_analysis_address_t);

}

void
sanitizer_memory_liveness_analysis_enable()
{
  redshow_analysis_enable(REDSHOW_ANALYSIS_MEMORY_LIVENESS);

  char dir_name[PATH_MAX];
  output_dir_config(dir_name, "/memory_liveness/");

  redshow_output_dir_config(REDSHOW_ANALYSIS_MEMORY_LIVENESS, dir_name);

  sanitizer_gpu_patch_type = GPU_PATCH_TYPE_ADDRESS_PATCH;
  sanitizer_gpu_patch_record_size = sizeof(gpu_patch_record_address_t);

}

void
sanitizer_data_dependency_analysis_enable()
{
  redshow_analysis_enable(REDSHOW_ANALYSIS_DATA_DEPENDENCY);

  sanitizer_analysis_async = false;

  char dir_name[PATH_MAX];
  output_dir_config(dir_name, "/data_dependency/");

  redshow_output_dir_config(REDSHOW_ANALYSIS_DATA_DEPENDENCY, dir_name);
  redshow_analysis_config(REDSHOW_ANALYSIS_DATA_DEPENDENCY, REDSHOW_ANALYSIS_DATA_FLOW_HASH, sanitizer_data_flow_hash);
  redshow_analysis_config(REDSHOW_ANALYSIS_DATA_DEPENDENCY, REDSHOW_ANALYSIS_READ_TRACE_IGNORE, sanitizer_read_trace_ignore);

  sanitizer_gpu_patch_type = GPU_PATCH_TYPE_ADDRESS_PATCH;
  sanitizer_gpu_patch_record_size = sizeof(gpu_patch_record_address_t);
  sanitizer_gpu_analysis_type = GPU_PATCH_TYPE_ADDRESS_ANALYSIS;
  sanitizer_gpu_analysis_record_size = sizeof(gpu_patch_analysis_address_t);
}

void
sanitizer_torch_monitor_analysis_enable()
{
  redshow_analysis_enable(REDSHOW_ANALYSIS_TORCH_MONITOR);
  sanitizer_analysis_async = false;

  char dir_name[PATH_MAX];
  output_dir_config(dir_name, "/torch_monitor/");

  redshow_output_dir_config(REDSHOW_ANALYSIS_TORCH_MONITOR, dir_name);

  sanitizer_gpu_patch_type = GPU_PATCH_TYPE_ADDRESS_PATCH;
  sanitizer_gpu_patch_record_size = sizeof(gpu_patch_record_address_t);
}


void
sanitizer_callbacks_subscribe() 
{
  sanitizer_correlation_callback = gpu_application_thread_correlation_callback;

  redshow_log_data_callback_register(sanitizer_log_data_callback);

  redshow_record_data_callback_register(sanitizer_record_data_callback, sanitizer_pc_views, sanitizer_mem_views);

  if (sanitizer_torch_analysis || sanitizer_torch_analysis_ongpu) {
    redshow_torch_enable();

    redshow_get_op_id_register(gpu_correlation_id);
  }

  redshow_tool_dtoh_register(sanitizer_dtoh);

  HPCRUN_SANITIZER_CALL(sanitizerSubscribe,
    (&sanitizer_subscriber_handle, sanitizer_subscribe_callback, NULL));

  HPCRUN_SANITIZER_CALL(sanitizerEnableDomain,
    (1, sanitizer_subscriber_handle, SANITIZER_CB_DOMAIN_DRIVER_API));

  HPCRUN_SANITIZER_CALL(sanitizerEnableDomain,
    (1, sanitizer_subscriber_handle, SANITIZER_CB_DOMAIN_RESOURCE));

  HPCRUN_SANITIZER_CALL(sanitizerEnableDomain,
    (1, sanitizer_subscriber_handle, SANITIZER_CB_DOMAIN_LAUNCH));

  HPCRUN_SANITIZER_CALL(sanitizerEnableDomain,
    (1, sanitizer_subscriber_handle, SANITIZER_CB_DOMAIN_MEMCPY));

  HPCRUN_SANITIZER_CALL(sanitizerEnableDomain,
    (1, sanitizer_subscriber_handle, SANITIZER_CB_DOMAIN_MEMSET));
}


void
sanitizer_callbacks_unsubscribe() 
{
  sanitizer_correlation_callback = 0;

  HPCRUN_SANITIZER_CALL(sanitizerUnsubscribe, (sanitizer_subscriber_handle));

  HPCRUN_SANITIZER_CALL(sanitizerEnableDomain,
    (0, sanitizer_subscriber_handle, SANITIZER_CB_DOMAIN_DRIVER_API));

  HPCRUN_SANITIZER_CALL(sanitizerEnableDomain,
    (0, sanitizer_subscriber_handle, SANITIZER_CB_DOMAIN_RESOURCE));

  HPCRUN_SANITIZER_CALL(sanitizerEnableDomain,
    (0, sanitizer_subscriber_handle, SANITIZER_CB_DOMAIN_LAUNCH));

  HPCRUN_SANITIZER_CALL(sanitizerEnableDomain,
    (0, sanitizer_subscriber_handle, SANITIZER_CB_DOMAIN_MEMCPY));

  HPCRUN_SANITIZER_CALL(sanitizerEnableDomain,
    (0, sanitizer_subscriber_handle, SANITIZER_CB_DOMAIN_MEMSET));
}


void
sanitizer_async_config
(
 bool async
)
{
  sanitizer_analysis_async = async;
}


static void
function_list_add
(
 sanitizer_function_list_entry_t *head,
 char *file_name
)
{
  FILE *fp = NULL;
  fp = fopen(file_name, "r");
  if (fp != NULL) {
    char function[FUNCTION_NAME_LENGTH];
    while (fgets(function, FUNCTION_NAME_LENGTH, fp) != NULL) {
      char *pos = NULL;
      if ((pos=strchr(function, '\n')) != NULL)
        *pos = '\0';
      PRINT("Sanitizer-> Add function %s from %s\n", function, file_name);
      sanitizer_function_list_register(head, function);
    }
    fclose(fp);
  }
}


void
sanitizer_function_config
(
 char *file_whitelist,
 char *file_blacklist
)
{
  if (file_whitelist != NULL) {
    sanitizer_function_list_init(&sanitizer_whitelist);
    function_list_add(sanitizer_whitelist, file_whitelist);
  }
  if (file_blacklist != NULL) {
    sanitizer_function_list_init(&sanitizer_blacklist);
    function_list_add(sanitizer_blacklist, file_blacklist);
  }
}


void
sanitizer_buffer_config
(
 int gpu_patch_record_num,
 int buffer_pool_size
)
{
  sanitizer_gpu_patch_record_num = gpu_patch_record_num;
  sanitizer_gpu_analysis_record_num = gpu_patch_record_num * GPU_PATCH_WARP_SIZE * 4;
  sanitizer_buffer_pool_size = buffer_pool_size;
}


void
sanitizer_approx_level_config
(
 int approx_level
)
{
  sanitizer_approx_level = approx_level;

  redshow_approx_level_config(sanitizer_approx_level);
}


void
sanitizer_views_config
(
 int pc_views,
 int mem_views
)
{
  sanitizer_pc_views = pc_views;
  sanitizer_mem_views = mem_views;
}


void
sanitizer_data_type_config
(
 char *data_type
)
{
  if (data_type == NULL) {
    sanitizer_data_type = REDSHOW_DATA_UNKNOWN;
  } else if (strcmp(data_type, "float") == 0 || strcmp(data_type, "FLOAT") == 0) {
    sanitizer_data_type = REDSHOW_DATA_FLOAT;
  } else if (strcmp(data_type, "int") == 0 || strcmp(data_type, "INT") == 0) {
    sanitizer_data_type = REDSHOW_DATA_INT;
  } else {
    sanitizer_data_type = REDSHOW_DATA_UNKNOWN;
  }

  PRINT("Sanitizer-> Config data type %u\n", sanitizer_data_type);

  redshow_data_type_config(sanitizer_data_type);
}


size_t
sanitizer_gpu_patch_record_num_get()
{
  return sanitizer_gpu_patch_record_num;
}


size_t
sanitizer_gpu_analysis_record_num_get()
{
  return sanitizer_gpu_analysis_record_num;
}


int
sanitizer_buffer_pool_size_get()
{
  return sanitizer_buffer_pool_size;
}


void
sanitizer_stop_flag_set()
{
  sanitizer_stop_flag = true;
}


void
sanitizer_stop_flag_unset()
{
  sanitizer_stop_flag = false;
}


void
sanitizer_gpu_analysis_config(int gpu_analysis_blocks)
{
  sanitizer_gpu_analysis_blocks = gpu_analysis_blocks;
}


void
sanitizer_read_trace_ignore_config(int read_trace_ignore)
{
  sanitizer_read_trace_ignore = read_trace_ignore == 1 ? true : false;
}


void
sanitizer_data_flow_hash_config(int data_flow_hash)
{
  sanitizer_data_flow_hash = data_flow_hash == 1 ? true : false;
}

void
sanitizer_liveness_ongpu_config(int liveness_ongpu)
{
  sanitizer_liveness_ongpu = liveness_ongpu == 1 ? true : false;
}

void
sanitizer_torch_analysis_config(int torch_analysis)
{
  sanitizer_torch_analysis = torch_analysis == 1 ? true : false;
}

void
sanitizer_torch_analysis_ongpu_config(int torch_ongpu)
{
  sanitizer_torch_analysis_ongpu = torch_ongpu == 1 ? true : false;
}

// cpu thread end
void
sanitizer_device_flush(void *args)
{
  if (sanitizer_stop_flag) {
    sanitizer_stop_flag_unset();

    if (sanitizer_analysis_async) {
      // Spin wait
      sanitizer_buffer_channel_flush(sanitizer_gpu_patch_type);
      if (sanitizer_gpu_analysis_blocks != 0) {
        sanitizer_buffer_channel_flush(sanitizer_gpu_analysis_type);
      }
      sanitizer_process_signal(); 
      while (sanitizer_buffer_channel_finish(sanitizer_gpu_patch_type) == false) {}
      while (sanitizer_buffer_channel_finish(sanitizer_gpu_analysis_type) == false) {}
    }

    // Attribute performance metrics to CCTs
    redshow_flush_thread(sanitizer_thread_id_local);
  }
}

// application end
void
sanitizer_device_shutdown(void *args)
{
  sanitizer_callbacks_unsubscribe();

  if (sanitizer_analysis_async) {
    atomic_store(&sanitizer_process_stop_flag, true);

    // Spin wait
    sanitizer_buffer_channel_flush(sanitizer_gpu_patch_type);
    sanitizer_process_signal(); 
    while (sanitizer_buffer_channel_finish(sanitizer_gpu_analysis_type) == false) {}
  }

  // Attribute performance metrics to CCTs
  redshow_flush();
  while (atomic_load(&sanitizer_process_thread_counter));
}


void
sanitizer_init
(
)
{
  sanitizer_stop_flag = false;
  sanitizer_thread_id_local = 0;
  sanitizer_thread_context = NULL;

  sanitizer_gpu_patch_buffer_device = NULL;
  sanitizer_gpu_patch_buffer_addr_read_device = NULL;
  sanitizer_gpu_patch_buffer_addr_write_device = NULL;

  sanitizer_gpu_patch_buffer_host = NULL;
  sanitizer_gpu_patch_buffer_addr_read_host = NULL;
  sanitizer_gpu_patch_buffer_addr_write_host = NULL;

  atomic_store(&sanitizer_process_awake_flag, false);
  atomic_store(&sanitizer_process_stop_flag, false);
}


void
sanitizer_process_init
(
)
{
  if (sanitizer_analysis_async) {
    pthread_t *thread = &(sanitizer_thread.thread);
    pthread_mutex_t *mutex = &(sanitizer_thread.mutex);
    pthread_cond_t *cond = &(sanitizer_thread.cond);

    // Create a new thread for the context without libmonitor watching
    monitor_disable_new_threads();

    atomic_fetch_add(&sanitizer_process_thread_counter, 1);

    pthread_mutex_init(mutex, NULL);
    pthread_cond_init(cond, NULL);
    pthread_create(thread, NULL, sanitizer_process_thread, NULL);

    monitor_enable_new_threads();
  }
}
