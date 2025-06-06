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
// Copyright ((c)) 2002-2020, Rice University
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
//   cuda-api.h
//
// Purpose:
//   interface definitions for wrapper around NVIDIA CUDA layer
//  
//***************************************************************************

#ifndef HPCTOOLKIT_GPU_NVIDIA_CUDA_API_H
#define HPCTOOLKIT_GPU_NVIDIA_CUDA_API_H

//*****************************************************************************
// nvidia includes
//*****************************************************************************

#include <link.h>
#include <cuda.h>

//*****************************************************************************
// interface operations
//*****************************************************************************

typedef struct cuda_device_property {
  int sm_count;
  int sm_clock_rate;
  int sm_shared_memory;
  int sm_registers;
  int sm_threads;
  int sm_blocks;
  int sm_schedulers;
  int num_threads_per_warp;
} cuda_device_property_t;


typedef struct {
  uint32_t unknown_field1[12];
  uint64_t function_addr;
} hpctoolkit_cufunc_record_st_t;


// DRIVER_UPDATE_CHECK(Keren): cufunc and cumod fields are reverse engineered
typedef struct {
  uint32_t unknown_field1[4]; 
  uint32_t function_index;
  uint32_t unknown_field2[3]; 
  CUmodule cumod;
  uint32_t unknown_field3[24];
  hpctoolkit_cufunc_record_st_t *cufunc_record;
} hpctoolkit_cufunc_st_t;


#if CUDA_VERSION >= 12000
typedef struct {
  uint32_t unknown_field_0[1];
  uint32_t cubin_id;
  uint32_t mod_id;
  uint32_t unknown_field_1[1];
} hpctoolkit_cumod_st_t;
;
#else
typedef struct {
  uint32_t cubin_id;
  uint32_t unknown_field[1];
  uint32_t mod_id;
} hpctoolkit_cumod_st_t;
;
#endif


//*****************************************************************************
// interface operations
//*****************************************************************************

// returns 0 on success
int 
cuda_bind
(
 void
);


void
cuda_kernel_launch
(
 CUfunction f,
 unsigned int gridDimX,
 unsigned int gridDimY,
 unsigned int gridDimZ,
 unsigned int blockDimX,
 unsigned int blockDimY,
 unsigned int blockDimZ,
 unsigned int sharedMemBytes,
 CUstream hStream,
 void **kernelParams
);


void
cuda_shared_mem_size_set
(
 CUfunction function,
 int size
);


void
cuda_module_load
(
 CUmodule *module,
 const char *fname
);


void
cuda_module_function_get
(
 CUfunction *hfunc,
 CUmodule hmod,
 const char *name
);


// returns 0 on success
int
cuda_context
(
 CUcontext *ctx
);


int
cuda_context_set
(
 CUcontext ctx
);


int
cuda_host_alloc
(
 void** pHost,
 size_t size
);

// returns 0 on success
int 
cuda_device_property_query
(
 int device_id, 
 cuda_device_property_t *property
);


CUstream
cuda_priority_stream_create
(
);


CUstream
cuda_stream_create
(
);


void
cuda_stream_synchronize
(
 CUstream stream
);


void
cuda_memcpy_dtoh
(
 void *dst,
 CUdeviceptr src,
 size_t byteCount,
 CUstream stream
);


void
cuda_memcpy_htod
(
 CUdeviceptr dst,
 void *src,
 size_t byteCount,
 CUstream stream
);


void
cuda_load_callback
(
 uint32_t cubin_id, 
 const void *cubin, 
 size_t cubin_size
);


void
cuda_unload_callback
(
 uint32_t cubin_id
);


// returns 0 on success
int
cuda_global_pc_sampling_required
(
  int *required
);


int
cuda_path
(
 char *buffer
);


bool
cuda_api_internal
(
);


#endif
