#ifndef _HPCTOOLKIT_GPU_NVIDIA_SANITIZER_CONTEXT_MAP_H_
#define _HPCTOOLKIT_GPU_NVIDIA_SANITIZER_CONTEXT_MAP_H_

/******************************************************************************
 * system includes
 *****************************************************************************/

#include <stdint.h>

/******************************************************************************
 * local includes
 *****************************************************************************/

#include <cuda.h>
#include <sanitizer.h>

#include <gpu-patch.h>

/******************************************************************************
 * type definitions 
 *****************************************************************************/

typedef struct sanitizer_context_map_entry_s sanitizer_context_map_entry_t;

/******************************************************************************
 * interface operations
 *****************************************************************************/

sanitizer_context_map_entry_t *
sanitizer_context_map_lookup
(
 CUcontext context
);


sanitizer_context_map_entry_t *
sanitizer_context_map_init
(
 CUcontext context
);


sanitizer_context_map_entry_t *
sanitizer_context_map_init_nolock
(
 CUcontext context
);


void
sanitizer_context_map_delete
(
 CUcontext context
);


void
sanitizer_context_map_insert
(
 CUcontext context,
 CUstream stream
);


void
sanitizer_context_map_analysis_function_update
(
 CUcontext context,
 CUfunction function
);


void
sanitizer_context_map_context_lock
(
 CUcontext context,
 int32_t thread_id
);


void
sanitizer_context_map_context_unlock
(
 CUcontext context,
 int32_t thread_id
);


void
sanitizer_context_map_stream_lock
(
 CUcontext context,
 CUstream stream
);


void
sanitizer_context_map_stream_unlock
(
 CUcontext context,
 CUstream stream
);


void
sanitizer_context_map_priority_stream_handle_update
(
 CUcontext context,
 Sanitizer_StreamHandle priority_stream_handle
);


void
sanitizer_context_map_priority_stream_handle_update_nolock
(
 CUcontext context,
 Sanitizer_StreamHandle priority_stream_handle
);


void
sanitizer_context_map_kernel_stream_handle_update
(
 CUcontext context,
 Sanitizer_StreamHandle kernel_stream_handle
);


void
sanitizer_context_map_kernel_stream_handle_update_nolock
(
 CUcontext context,
 Sanitizer_StreamHandle kernel_stream_handle
);


void
sanitizer_context_map_buffer_device_update
(
 CUcontext context,
 gpu_patch_buffer_t *buffer_device
);


void
sanitizer_context_map_buffer_addr_read_device_update
(
 CUcontext context,
 gpu_patch_buffer_t *buffer_addr_read_device
);


void
sanitizer_context_map_buffer_addr_write_device_update
(
 CUcontext context,
 gpu_patch_buffer_t *buffer_addr_write_device
);


void
sanitizer_context_map_aux_addr_dict_device_update
(
 CUcontext context,
 gpu_patch_aux_address_dict_t *aux_addr_dict_device
);


void
sanitizer_context_map_aux_torchview_dict_device_update
(
 CUcontext context,
 gpu_patch_aux_torchview_dict_t *aux_torchview_dict_device
);

void
sanitizer_context_map_aux_addr_dict_start_end_device_update
(
 CUcontext context,
 gpu_patch_analysis_address_t *aux_torchview_start_end_device
);


void
sanitizer_context_map_aux_addr_dict_read_pc_range_bit_map_device_update
(
 CUcontext context,
 uint64_t *aux_torchview_read_pc_range_bit_map_device
);


void
sanitizer_context_map_aux_addr_dict_write_pc_range_bit_map_device_update
(
 CUcontext context,
 uint64_t *aux_torchview_write_pc_range_bit_map_device
);


void
sanitizer_context_map_torch_aux_addr_dict_device_update
(
 CUcontext context,
 gpu_patch_aux_address_dict_t *torch_aux_addr_dict_device
);


void
sanitizer_context_map_buffer_reset_update
(
 CUcontext context,
 gpu_patch_buffer_t *buffer_reset
);


void
sanitizer_context_map_buffer_addr_read_reset_update
(
 CUcontext context,
 gpu_patch_buffer_t *buffer_addr_read_reset
);


void
sanitizer_context_map_buffer_addr_write_reset_update
(
 CUcontext context,
 gpu_patch_buffer_t *buffer_addr_write_reset
);


void
sanitizer_context_map_aux_addr_dict_reset_update
(
 CUcontext context,
 gpu_patch_aux_address_dict_t *aux_addr_dict_reset
);

void
sanitizer_context_map_torch_aux_addr_dict_reset_update
(
 CUcontext context,
 gpu_patch_aux_address_dict_t *torch_aux_addr_dict_reset
);


CUstream
sanitizer_context_map_entry_priority_stream_get
(
 sanitizer_context_map_entry_t *entry
);


CUstream
sanitizer_context_map_entry_kernel_stream_get
(
 sanitizer_context_map_entry_t *entry
);


Sanitizer_StreamHandle
sanitizer_context_map_entry_priority_stream_handle_get
(
 sanitizer_context_map_entry_t *entry
);


Sanitizer_StreamHandle
sanitizer_context_map_entry_kernel_stream_handle_get
(
 sanitizer_context_map_entry_t *entry
);


gpu_patch_buffer_t *
sanitizer_context_map_entry_buffer_device_get
(
 sanitizer_context_map_entry_t *entry
);


gpu_patch_buffer_t *
sanitizer_context_map_entry_buffer_addr_read_device_get
(
 sanitizer_context_map_entry_t *entry
);


gpu_patch_buffer_t *
sanitizer_context_map_entry_buffer_addr_write_device_get
(
 sanitizer_context_map_entry_t *entry
);


gpu_patch_aux_address_dict_t *
sanitizer_context_map_entry_aux_addr_dict_device_get
(
 sanitizer_context_map_entry_t *entry
);

gpu_patch_aux_torchview_dict_t *
sanitizer_context_map_entry_aux_torchview_dict_device_get
(
 sanitizer_context_map_entry_t *entry
);

gpu_patch_analysis_address_t *
sanitizer_context_map_entry_aux_torchview_dict_start_end_device_get
(
 sanitizer_context_map_entry_t *entry
);

uint64_t *
sanitizer_context_map_entry_aux_torchview_dict_read_pc_range_bit_map_device_get
(
 sanitizer_context_map_entry_t *entry
);

uint64_t *
sanitizer_context_map_entry_aux_torchview_dict_write_pc_range_bit_map_device_get
(
 sanitizer_context_map_entry_t *entry
);

gpu_patch_aux_address_dict_t *
sanitizer_context_map_entry_torch_aux_addr_dict_device_get
(
 sanitizer_context_map_entry_t *entry
);

gpu_patch_buffer_t *
sanitizer_context_map_entry_buffer_reset_get
(
 sanitizer_context_map_entry_t *entry
);


gpu_patch_buffer_t *
sanitizer_context_map_entry_buffer_addr_read_reset_get
(
 sanitizer_context_map_entry_t *entry
);


gpu_patch_buffer_t *
sanitizer_context_map_entry_buffer_addr_write_reset_get
(
 sanitizer_context_map_entry_t *entry
);


gpu_patch_aux_address_dict_t *
sanitizer_context_map_entry_aux_addr_dict_reset_get
(
 sanitizer_context_map_entry_t *entry
);

gpu_patch_aux_torchview_dict_t *
sanitizer_context_map_entry_aux_torchview_dict_reset_get
(
 sanitizer_context_map_entry_t *entry
);

gpu_patch_analysis_address_t *
sanitizer_context_map_entry_aux_torchview_dict_start_end_reset_get
(
 sanitizer_context_map_entry_t *entry
);

uint64_t *
sanitizer_context_map_entry_aux_torchview_dict_read_pc_range_bit_map_reset_get
(
 sanitizer_context_map_entry_t *entry
);

uint64_t *
sanitizer_context_map_entry_aux_torchview_dict_write_pc_range_bit_map_reset_get
(
 sanitizer_context_map_entry_t *entry
);

gpu_patch_aux_address_dict_t *
sanitizer_context_map_entry_torch_aux_addr_dict_reset_get
(
 sanitizer_context_map_entry_t *entry
);


CUfunction
sanitizer_context_map_entry_analysis_function_get
(
 sanitizer_context_map_entry_t *entry
);


#endif
