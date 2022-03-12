#ifndef _HPCTOOLKIT_GPU_NVIDIA_SANITIZER_API_H_
#define _HPCTOOLKIT_GPU_NVIDIA_SANITIZER_API_H_

int
sanitizer_bind();

void
sanitizer_redundancy_analysis_enable();

void
sanitizer_data_flow_analysis_enable();

void
sanitizer_value_pattern_analysis_enable();

void
sanitizer_memory_profile_analysis_enable();

void
sanitizer_memory_heatmap_analysis_enable();

void
sanitizer_memory_liveness_analysis_enable();

// for sub-allocation callback
void
memory_sub_alloc_callback (void *ptr, size_t size);

void
sanitizer_callbacks_subscribe();

void
sanitizer_callbacks_unsubscribe();

void
sanitizer_stop_flag_set();

void
sanitizer_stop_flag_unset();

void
sanitizer_device_flush(void *args);

void
sanitizer_device_shutdown(void *args);

void
sanitizer_init();

void
sanitizer_process_init();

void
sanitizer_process_signal();

void
sanitizer_async_config(bool async);

void
sanitizer_function_config(char *whitelist, char *blacklist);

void
sanitizer_buffer_config(int gpu_patch_record_num, int buffer_pool_size);

void
sanitizer_approx_level_config(int approx_level);

void
sanitizer_views_config(int pc_views, int mem_views);

void
sanitizer_data_type_config(char *data_type);

void
sanitizer_read_trace_ignore_config(int read_trace_ignore);

void
sanitizer_data_flow_hash_config(int data_flow_hash);

size_t
sanitizer_gpu_patch_record_num_get();

size_t
sanitizer_gpu_analysis_record_num_get();

int
sanitizer_buffer_pool_size_get();

void
sanitizer_gpu_analysis_config(int gpu_analysis_blocks);

#endif
