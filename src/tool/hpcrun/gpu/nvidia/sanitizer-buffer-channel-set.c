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
// local includes
//******************************************************************************

#include <lib/prof-lean/stacks.h>

#include <hpcrun/memory/hpcrun-malloc.h>

#include "sanitizer-buffer-channel.h"
#include "sanitizer-buffer-channel-set.h"


//******************************************************************************
// macros
//******************************************************************************

#define channel_stack_push  \
  typed_stack_push(sanitizer_buffer_channel_ptr_t, cstack)

#define channel_stack_forall \
  typed_stack_forall(sanitizer_buffer_channel_ptr_t, cstack)

#define channel_stack_elem_t \
  typed_stack_elem(sanitizer_buffer_channel_ptr_t)

#define channel_stack_elem_ptr_set \
  typed_stack_elem_ptr_set(sanitizer_buffer_channel_ptr_t, cstack)


//******************************************************************************
// type declarations
//******************************************************************************

//----------------------------------------------------------
// support for a stack of buffer channels
//----------------------------------------------------------

typedef sanitizer_buffer_channel_t *sanitizer_buffer_channel_ptr_t;

typedef struct {
  s_element_ptr_t next;
  sanitizer_buffer_channel_ptr_t channel;
} typed_stack_elem(sanitizer_buffer_channel_ptr_t);

typed_stack_declare_type(sanitizer_buffer_channel_ptr_t);


//******************************************************************************
// local data
//******************************************************************************

static 
typed_stack_elem_ptr(sanitizer_buffer_channel_ptr_t) 
sanitizer_buffer_channel_stack[GPU_PATCH_TYPE_COUNT];

//******************************************************************************
// private operations
//******************************************************************************

// implement stack of correlation channels
typed_stack_impl(sanitizer_buffer_channel_ptr_t, cstack);

static void
channel_forone
(
 channel_stack_elem_t *se,
 void *arg
)
{
  sanitizer_buffer_channel_t *channel = se->channel;

  sanitizer_buffer_channel_fn_t channel_fn =
    (sanitizer_buffer_channel_fn_t) arg;

  channel_fn(channel);
}


static void
sanitizer_buffer_channel_set_forall
(
 sanitizer_buffer_channel_fn_t channel_fn,
 uint32_t type
)
{
  channel_stack_forall(&sanitizer_buffer_channel_stack[type], channel_forone, 
    channel_fn);
}


//******************************************************************************
// interface operations
//******************************************************************************

void
sanitizer_buffer_channel_set_insert
(
 sanitizer_buffer_channel_t *channel,
 uint32_t type
)
{
  // allocate and initialize new entry for channel stack
  channel_stack_elem_t *e = 
    (channel_stack_elem_t *) hpcrun_malloc_safe(sizeof(channel_stack_elem_t));

  // initialize the new entry
  e->channel = channel;
  channel_stack_elem_ptr_set(e, 0); // clear the entry's next ptr

  // add the entry to the channel stack
  channel_stack_push(&sanitizer_buffer_channel_stack[type], e);
}

void
sanitizer_buffer_channel_set_consume
(
)
{
  for (uint32_t i = 0; i < GPU_PATCH_TYPE_COUNT; ++i) {
    sanitizer_buffer_channel_set_forall(&sanitizer_buffer_channel_consume, i);
  }
}
