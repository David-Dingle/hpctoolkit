// -*-Mode: C++;-*-

// * BeginRiceCopyright *****************************************************
//
// $HeadURL$
// $Id$
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
//   $HeadURL$
//
// Purpose:
//   [The purpose of this file]
//
// Description:
//   [The set of functions, macros, etc. defined in the file]
//
//***************************************************************************

//************************* System Include Files ****************************

#include <iostream>
#include <fstream>
#include <filesystem>

#include <string>
#include <climits>
#include <cstring>

#include <typeinfo>
#include <unordered_map>

#include <sys/stat.h>

//*************************** User Include Files ****************************

#include <include/uint.h>
#include <include/gcc-attr.h>
#include <include/gpu-metric-names.h>

#include "CallPath-TorchView.hpp"

using std::string;

#include <lib/prof/CCT-Tree.hpp>
#include <lib/prof/CallPath-Profile.hpp>
#include <lib/prof/Metric-Mgr.hpp>
#include <lib/prof/Metric-ADesc.hpp>

#include <lib/profxml/XercesUtil.hpp>
#include <lib/profxml/PGMReader.hpp>

#include <lib/prof-lean/hpcrun-metric.h>

#include <lib/binutils/LM.hpp>
#include <lib/binutils/VMAInterval.hpp>

#include <lib/xml/xml.hpp>

#include <lib/support/diagnostics.h>
#include <lib/support/Logic.hpp>
#include <lib/support/IOUtil.hpp>
#include <lib/support/StrUtil.hpp>


#include <vector>
#include <queue>
#include <iostream>

#include "redshow_graphviz.h"
#include <memory>

namespace Analysis {

  namespace CallPath {

    inline bool is_file_exists (const std::string& name) {
      if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
      } else {
        return false;
      }   
    }

    typedef struct TV_CTX_NODE{
      int32_t ctx_id;
      std::string context;
      std::vector<uint64_t> pcs;
      // uint64_t pcs;

      TV_CTX_NODE() = default;

      TV_CTX_NODE(int32_t cid) : ctx_id(cid), context("") {
        this->pcs = std::vector<uint64_t>{};
      }

      TV_CTX_NODE(const TV_CTX_NODE& rhs){
        this->ctx_id = rhs.ctx_id;
        this->context = std::string(rhs.context);
        this->pcs = std::vector<uint64_t>(rhs.pcs.begin(), rhs.pcs.end());
      }

      void operator=(const TV_CTX_NODE& rhs){
        this->ctx_id = rhs.ctx_id;
        this->context = std::string(rhs.context);
        this->pcs = std::vector<uint64_t>(rhs.pcs.begin(), rhs.pcs.end());
      }
    } ctx_node_t;

    typedef struct PythonContex{
      std::string file_name;
      std::string function_name;
      int function_first_lineno;
      int lineno;

      PythonContex() = default;

      PythonContex(const PythonContex& rhs){
        this->file_name = std::string(rhs.file_name);
        this->function_name = std::string(rhs.function_name);
        this->function_first_lineno = rhs.function_first_lineno;
        this->lineno = rhs.lineno;
      }

      void operator=(const PythonContex& rhs){
        this->file_name = std::string(rhs.file_name);
        this->function_name = std::string(rhs.function_name);
        this->function_first_lineno = rhs.function_first_lineno;
        this->lineno = rhs.lineno;
      }
    } python_context_t;

    typedef struct PyState{
      int index;
      int num_states;
      std::vector<python_context_t> python_contexts = std::vector<python_context_t>{};
      std::string hash;

      PyState() = default;

      PyState(const PyState& rhs){
        this->index = rhs.index;
        this->num_states = rhs.num_states;
        this->python_contexts = std::vector<python_context_t>{rhs.python_contexts.begin(), rhs.python_contexts.end()};
      }

      void operator=(const PyState& rhs){
        this->index = rhs.index;
        this->num_states = rhs.num_states;
        this->python_contexts = std::vector<python_context_t>{rhs.python_contexts.begin(), rhs.python_contexts.end()};
      }
    } py_state_t;

    typedef struct TORCH_VIEW_ACCESS{
      uint64_t global_id;  // view_node_id or mem_block_id
      uint8_t access_type;  // view_node or mem block
      uint64_t total_stalls = 0;
      uint64_t map_size = 0;
      std::vector<std::vector<py_state_t>> py_states = std::vector<std::vector<py_state_t>>{};
      std::vector<std::vector<ctx_node_t>> ctx_ids = std::vector<std::vector<ctx_node_t>>{};

      TORCH_VIEW_ACCESS(){};
    } torch_view_access_t;

    typedef std::vector<torch_view_access_t> VIEW_CTX_MAP;



    static void read_memory_node(const std::string &file_name, VIEW_CTX_MAP &view_ctx_map) {
      std::ifstream fileread(file_name);
      std::string word;
      bool is_id = false;
      bool is_index = false;
      bool is_num_states = false;
      bool is_file_name = false;
      bool is_function_name = false;
      bool is_function_first_lineno = false;
      bool is_lineno = false;
      bool is_pytates_hash = false;
      bool is_object_type = false;
      bool is_ctx_id = false;
      bool is_pcs = false;

      std::set<int32_t> _ctx_set = std::set<int32_t>{};

      while (fileread >> word) {
        // std::cout << "word: " << word << std::endl;

        if (word == "id"){
          is_id = true;
          is_index = false;
          is_num_states = false;
          is_file_name = false;
          is_function_name = false;
          is_function_first_lineno = false;
          is_lineno = false;
          is_pytates_hash = false;
          is_object_type = false;
          is_ctx_id = false;
          is_pcs = false;

          if (!view_ctx_map.empty()) { // skip the first loop when the map is empty
            if (view_ctx_map.back().ctx_ids.empty()) { 
              view_ctx_map.pop_back();  // if the block is empty, remove the block
            } else {
              for (int it = (view_ctx_map.back().ctx_ids.size() - 1); it >= 0; --it) {  // check if any pystate got no ctx_id, then remove them both
                if (view_ctx_map.back().ctx_ids.at(it).empty() || (view_ctx_map.back().ctx_ids.at(it).size() == 1 
                                                                    && 
                                                                  view_ctx_map.back().ctx_ids.at(it).at(0).ctx_id == 0)) {
                  // std::cout << "branch is taken" << std::endl;
                  view_ctx_map.back().py_states.erase(view_ctx_map.back().py_states.begin() + it);
                  view_ctx_map.back().ctx_ids.erase(view_ctx_map.back().ctx_ids.begin() + it);
                  view_ctx_map.back().map_size--;
                }
              }

            }
          }

          view_ctx_map.emplace_back();
          continue;
        }

        if (word == "python_state"){
          is_id = false;
          is_index = false;
          is_num_states = false;
          is_file_name = false;
          is_function_name = false;
          is_function_first_lineno = false;
          is_lineno = false;
          is_pytates_hash = false;
          is_object_type = false;
          is_ctx_id = false;
          is_pcs = false;

//          view_ctx_map.back().map_size++;
//          view_ctx_map.back().py_states[view_ctx_map.back().map_size - 1] = std::vector<py_state_t>{};
//          view_ctx_map.back().ctx_ids[view_ctx_map.back().map_size - 1] = std::vector<ctx_node_t>{};
          continue;
        }

        if (word == "index"){
          is_id = false;
          is_index = true;
          is_num_states = false;
          is_file_name = false;
          is_function_name = false;
          is_function_first_lineno = false;
          is_lineno = false;
          is_pytates_hash = false;
          is_object_type = false;
          is_ctx_id = false;
          is_pcs = false;

          view_ctx_map.back().map_size++;
          std::vector<ctx_node_t> v_a = std::vector<ctx_node_t>{};
          std::vector<py_state_t> v_b = std::vector<py_state_t>{};
          view_ctx_map.back().ctx_ids.push_back(v_a); //[view_ctx_map.back().map_size - 1] = std::vector<ctx_node_t>{};
          view_ctx_map.back().py_states.push_back(v_b); //[view_ctx_map.back().map_size - 1] = std::vector<py_state_t>{};

          view_ctx_map.back().py_states.back().emplace_back();
          continue;
        }

        if (word == "num_states"){
          is_id = false;
          is_index = false;
          is_num_states = true;
          is_file_name = false;
          is_function_name = false;
          is_function_first_lineno = false;
          is_lineno = false;
          is_pytates_hash = false;
          is_object_type = false;
          is_ctx_id = false;
          is_pcs = false;

          continue;
        }

        if (word == "py_state"){
          is_id = false;
          is_index = false;
          is_num_states = false;
          is_file_name = false;
          is_function_name = false;
          is_function_first_lineno = false;
          is_lineno = false;
          is_pytates_hash = false;
          is_object_type = false;
          is_ctx_id = false;
          is_pcs = false;

          continue;
        }

        if (word == "file_name"){
          is_id = false;
          is_index = false;
          is_num_states = false;
          is_file_name = true;
          is_function_name = false;
          is_function_first_lineno = false;
          is_lineno = false;
          is_pytates_hash = false;
          is_object_type = false;
          is_ctx_id = false;
          is_pcs = false;

          view_ctx_map.back().py_states.back().back().python_contexts.emplace_back();
          continue;
        }

        if (word == "function_name"){
          is_id = false;
          is_index = false;
          is_num_states = false;
          is_file_name = false;
          is_function_name = true;
          is_function_first_lineno = false;
          is_lineno = false;
          is_pytates_hash = false;
          is_object_type = false;
          is_ctx_id = false;
          is_pcs = false;

          continue;
        }

        if (word == "function_first_lineno"){
          is_id = false;
          is_index = false;
          is_num_states = false;
          is_file_name = false;
          is_function_name = false;
          is_function_first_lineno = true;
          is_lineno = false;
          is_pytates_hash = false;
          is_object_type = false;
          is_ctx_id = false;
          is_pcs = false;

          continue;
        }

        if (word == "lineno"){
          is_id = false;
          is_index = false;
          is_num_states = false;
          is_file_name = false;
          is_function_name = false;
          is_function_first_lineno = false;
          is_lineno = true;
          is_pytates_hash = false;
          is_object_type = false;
          is_ctx_id = false;
          is_pcs = false;

          continue;
        }

        if (word == "pytates_hash"){
          is_id = false;
          is_index = false;
          is_num_states = false;
          is_file_name = false;
          is_function_name = false;
          is_function_first_lineno = false;
          is_lineno = false;
          is_pytates_hash = true;
          is_object_type = false;
          is_ctx_id = false;
          is_pcs = false;

          continue;
        }

        if (word == "object_type"){
          is_id = false;
          is_index = false;
          is_num_states = false;
          is_file_name = false;
          is_function_name = false;
          is_function_first_lineno = false;
          is_lineno = false;
          is_pytates_hash = false;
          is_object_type = true;
          is_ctx_id = false;
          is_pcs = false;

          continue;
        }

        if (word == "ctx_id"){
          is_id = false;
          is_index = false;
          is_num_states = false;
          is_file_name = false;
          is_function_name = false;
          is_function_first_lineno = false;
          is_lineno = false;
          is_pytates_hash = false;
          is_object_type = false;
          is_ctx_id = true;
          is_pcs = false;

          _ctx_set.clear();
          continue;
        }

        if (word == "pc"){
          is_id = false;
          is_index = false;
          is_num_states = false;
          is_file_name = false;
          is_function_name = false;
          is_function_first_lineno = false;
          is_lineno = false;
          is_pytates_hash = false;
          is_object_type = false;
          is_ctx_id = false;
          is_pcs = true;

          continue;
        }

        if(is_id) {
          // std::size_t pos{};
          uint64_t id = (uint64_t)std::stoul(word);
          view_ctx_map.back().global_id = id;

          continue;
        }
        if(is_index){
          view_ctx_map.back().py_states.back().back().index = std::stoi(word);

          continue;
        }
        if(is_num_states){
          view_ctx_map.back().py_states.back().back().num_states = std::stoi(word);

          continue;
        }
        if(is_file_name) {
          view_ctx_map.back().py_states.back().back().python_contexts.back().file_name = word;

          continue;
        }
        if(is_function_name) {
          view_ctx_map.back().py_states.back().back().python_contexts.back().function_name = word;

          continue;
        }
        if(is_function_first_lineno) {
          view_ctx_map.back().py_states.back().back().python_contexts.back().function_first_lineno = std::stoi(word);

          continue;
        }
        if(is_lineno) {
          view_ctx_map.back().py_states.back().back().python_contexts.back().lineno = std::stoi(word);

          continue;
        }

        if(is_pytates_hash) {
          view_ctx_map.back().py_states.back().back().hash = word;

          continue;
        }

        if(is_object_type) {
          uint8_t type = std::stoul(word);
          view_ctx_map.back().access_type = type;

          continue;
        }

        if(is_ctx_id) {
          int32_t ctx = (int32_t)std::stol(word);
          if(_ctx_set.find(ctx) == _ctx_set.end()) {
            view_ctx_map.back().ctx_ids.back().emplace_back(ctx);
            _ctx_set.insert(ctx);
            // std::cout << ctx << " : ";
          }

          continue;
        }

        if(is_pcs) {
          uint64_t pc = (uint64_t)std::stoul(word);
          view_ctx_map.back().ctx_ids.back().back().pcs.push_back(pc);
          // std::cout << view_ctx_map.back().ctx_ids.back().back().pcs.back() << " size: "<< view_ctx_map.back().ctx_ids.back().back().pcs.size() << std::endl;

          continue;
        }

      }
      fileread.close();
    }


#define MAX_STR_LEN 128

    static std::string
    trunc(const std::string &raw_str) {
      std::string str = raw_str;
      if (str.size() > MAX_STR_LEN) {
        str.erase(str.begin() + MAX_STR_LEN, str.end());
      }
      return str;
    }

    static std::vector<std::string>
    getInlineStack(Prof::Struct::ACodeNode *stmt) {
      std::vector<std::string> st;
      Prof::Struct::Alien *alien = stmt->ancestorAlien();
      if (alien) {
        auto func_name = trunc(alien->name());
        auto *stmt = alien->parent();
        if (stmt) {
          if (alien->name() == "<inline>") {
            // Inline macro
          } else if (stmt->type() == Prof::Struct::ANode::TyAlien) {
            // inline function
            alien = dynamic_cast<Prof::Struct::Alien *>(stmt);
          } else {
            return st;
          }
          auto file_name = alien->fileName();
          auto line = std::to_string(alien->begLine());
          auto name = file_name + ":" + line + "\t" + func_name;
          st.push_back(name);

          while (true) {
            stmt = alien->parent();
            if (stmt) {
              alien = stmt->ancestorAlien();
              if (alien) {
                func_name = trunc(alien->name());
                stmt = alien->parent();
                if (stmt) {
                  if (alien->name() == "<inline>") {
                    // Inline macro
                  } else if (stmt->type() == Prof::Struct::ANode::TyAlien) {
                    // inline function
                    alien = dynamic_cast<Prof::Struct::Alien *>(stmt);
                  } else {
                    break;
                  }
                  file_name = alien->fileName();
                  line = std::to_string(alien->begLine());
                  name = file_name + ":" + line + "\t" + func_name;
                  st.push_back(name);
                } else {
                  break;
                }
              } else {
                break;
              }
            } else {
              break;
            }
          }
        }
      }

      std::reverse(st.begin(), st.end());
      return st;
    }

#define MAX_FRAMES 20

    static void matchCCTNode(Prof::CallPath::CCTIdToCCTNodeMap &cctNodeMap, VIEW_CTX_MAP &ctx_node_map) {
      // match nodes
      for (auto &iter : ctx_node_map) {
        for (auto &citer: iter.ctx_ids) {
          for (auto &node: citer) {
            Prof::CCT::ANode *cct = NULL;

            if (cctNodeMap.find(node.ctx_id) != cctNodeMap.end()) {
              cct = cctNodeMap.at(node.ctx_id);
            } else {
              auto node_id = (uint32_t)(-node.ctx_id);
              if (cctNodeMap.find(node_id) != cctNodeMap.end()) {
                cct = cctNodeMap.at(node_id);
              }
            }

            if (cct) {
              std::stack < Prof::CCT::ProcFrm * > st;
              Prof::CCT::ProcFrm *proc_frm = NULL;
              std::string cct_context;

              if (cct->type() != Prof::CCT::ANode::TyProcFrm &&
                  cct->type() != Prof::CCT::ANode::TyRoot) {
                proc_frm = cct->ancestorProcFrm();

                if (proc_frm != NULL) {
                  auto *strct = cct->structure();
                  if (strct->ancestorAlien()) {
                    auto alien_st = getInlineStack(strct);
                    for (auto &name: alien_st) {
                      // Get inline call stack
                      cct_context.append(name);
                      cct_context.append("#\n");
                    }
                  }
                  auto *file_struct = strct->ancestorFile();
                  auto file_name = file_struct->name();
                  auto line = std::to_string(strct->begLine());
                  auto name = file_name + ":" + line + "\t <op>";
                  cct_context.append(name);
                  cct_context.append("#\n");
                }
              } else {
                proc_frm = dynamic_cast<Prof::CCT::ProcFrm *>(cct);
              }

              while (proc_frm) {
                if (st.size() > MAX_FRAMES) {
                  break;
                }
                st.push(proc_frm);
                auto *stmt = proc_frm->parent();
                if (stmt) {
                  proc_frm = stmt->ancestorProcFrm();
                } else {
                  break;
                }
              };

              while (!st.empty()) {
                proc_frm = st.top();
                st.pop();
                if (proc_frm->structure()) {
                  if (proc_frm->ancestorCall()) {
                    auto func_name = trunc(proc_frm->structure()->name());
                    auto *call = proc_frm->ancestorCall();
                    auto *call_strct = call->structure();
                    auto line = std::to_string(call_strct->begLine());
                    std::string file_name = "Unknown";
                    if (call_strct->ancestorAlien()) {
                      auto alien_st = getInlineStack(call_strct);
                      for (auto &name: alien_st) {
                        // Get inline call stack
                        node.context.append(name);
                        node.context.append("#\n");
                      }

                      auto fname = call_strct->ancestorAlien()->fileName();
                      if (fname.find("<unknown file>") == std::string::npos) {
                        file_name = fname;
                      }
                      auto name = file_name + ":" + line + "\t" + func_name;
                      node.context.append(name);
                      node.context.append("#\n");
                    } else if (call_strct->ancestorFile()) {
                      auto fname = call_strct->ancestorFile()->name();
                      if (fname.find("<unknown file>") == std::string::npos) {
                        file_name = fname;
                      }
                      auto name = file_name + ":" + line + "\t" + func_name;
                      node.context.append(name);
                      node.context.append("#\n");
                    }
                  }
                }
              }

              if (cct_context.size() != 0) {
                node.context.append(cct_context);
              }
            }
          }
        }
      }
    }

    /** Work with GPA torch_view blame shiftting
     * IFF output_dir/../../gpa-measurments/torch_view/torch_view_report.csv.context_v2 exists
    */
    typedef uint64_t pystates_hash_t;
    typedef uint64_t pc_t;
    typedef uint64_t pc_count_t;
    typedef std::map<pc_t, pc_count_t> pc_count_map_t;
    typedef std::map<pystates_hash_t, pc_count_map_t> STALL_WEIGHTS_MAP; //stall_weights_map_t;
    // stall_weights_map_t stall_weights = stall_weights_map_t{};

    void read_stall_weights_file(const std::string& file_name, uint64_t& gpa_total_stall, STALL_WEIGHTS_MAP& stall_weights) {
      std::ifstream fileread(file_name);
      std::string word;

      bool is_total_stalls = false;
      bool is_gpa_id = false;
      bool is_pystates_hash = false;
      bool is_leaf_lm_id = false;
      bool is_lm_ip = false;
      bool is_pcs = false;
      bool is_count = false;

      pystates_hash_t current_hash = 0;
      pc_t current_pc = 0;

      // std::cout << "Start reading from " << file_name << std::endl;

      while (fileread >> word) {

        if (word == "gpa_id"){
          is_total_stalls = false;
          is_gpa_id = true;
          is_pystates_hash = false;
          is_leaf_lm_id = false;
          is_lm_ip = false;
          is_pcs = false;
          is_count = false;

          continue;
        }
      
        if (word == "pystates_hash"){
          is_total_stalls = false;
          is_gpa_id = false;
          is_pystates_hash = true;
          is_leaf_lm_id = false;
          is_lm_ip = false;
          is_pcs = false;
          is_count = false;

          continue;
        }
      
        if (word == "leaf_lm_id"){
          is_total_stalls = false;
          is_gpa_id = false;
          is_pystates_hash = false;
          is_leaf_lm_id = true;
          is_lm_ip = false;
          is_pcs = false;
          is_count = false;

          continue;
        }

        if (word == "lm_ip"){
          is_total_stalls = false;
          is_gpa_id = false;
          is_pystates_hash = false;
          is_leaf_lm_id = false;
          is_lm_ip = true;
          is_pcs = false;
          is_count = false;

         continue;
        }

        if (word == "pc"){
          is_total_stalls = false;
          is_gpa_id = false;
          is_pystates_hash = false;
          is_leaf_lm_id = false;
          is_lm_ip = false;
          is_pcs = true;
          is_count = false;

          continue;
        }

        if (word == "count"){
          is_total_stalls = false;
          is_gpa_id = false;
          is_pystates_hash = false;
          is_leaf_lm_id = false;
          is_lm_ip = false;
          is_pcs = false;
          is_count = true;

          continue;
        }

        if (word == "total_stalls"){
          is_total_stalls = true;
          is_gpa_id = false;
          is_pystates_hash = false;
          is_leaf_lm_id = false;
          is_lm_ip = false;
          is_pcs = false;
          is_count = false;
        
          continue;
        }

        if (is_gpa_id || is_leaf_lm_id || is_lm_ip) {
          continue;
        } // do nothing

        if (is_total_stalls){
          gpa_total_stall = (uint64_t)std::stoul(word);
          continue;
        }

        if (is_pystates_hash){
          current_hash = (uint64_t)std::stoul(word);
          if (stall_weights.find(current_hash) == stall_weights.end()) {
            stall_weights[current_hash] = pc_count_map_t();
          }
          // stall_weights[current_hash] = pc_count_map_t();
          continue;
        }

        if (is_pcs){
          current_pc = (uint64_t)std::stoul(word);
          if (stall_weights[current_hash].find(current_pc) == stall_weights[current_hash].end()) {
            stall_weights[current_hash][current_pc] = (uint64_t)0;
          }
          continue;
        }

        if (is_count){
          stall_weights[current_hash][current_pc] += (uint64_t)std::stoul(word);
          continue;
        }
      }
      return;
    }

    int get_pc_stall_weights(const std::string &file_path, uint64_t& gpa_total_stall, STALL_WEIGHTS_MAP& stall_weights_map) {
      // get file directory (parent folder)
      std::string directoryPath;
      const size_t last_slash_idx = file_path.rfind('/');
      if (std::string::npos != last_slash_idx) {
        directoryPath = file_path.substr(0, last_slash_idx);
      }

      std::string gpa_file = directoryPath.append("/../../gpa-measurements/torch_view/torch_view_report.csv.context_v2");
      if(!is_file_exists(gpa_file)) {
        std::cout << "Stall weights not found. " << gpa_file << std::endl;
        return 0; // file doesn't exist
      } else {
        read_stall_weights_file(gpa_file, gpa_total_stall, stall_weights_map);
        std::cout << "Read stall weights from " << gpa_file << std::endl;
        return 1; // read data 
      }
    }

    std::uint64_t get_pystate_total_stalls(STALL_WEIGHTS_MAP& stall_weights_map, const pystates_hash_t& pystates_hash, const torch_view_access_t& ctx_node, const int& index) {
      std::uint64_t pystate_stalls = 0;
      std::set<uint64_t> pc_set = std::set<uint64_t>{};
      for (auto& ctx : ctx_node.ctx_ids.at(index)) {
        for (auto& pc : ctx.pcs) {
          pc_set.insert(pc);
        }
      }
      // std::cout << "Global_id: " << ctx_node.global_id << std::endl;
      // for (auto& pc : pc_set) {
      //   std::cout << "  " << pc << std::endl;
      // }
      // std::cout << std::endl;
      for (auto& [pc, count] : stall_weights_map[pystates_hash]){
        if (pc_set.find(pc) != pc_set.end()) {
          pystate_stalls += count;
        }
      }
      // std::cout << "total node-state stalls: " << pystate_stalls << std::endl;
      return pystate_stalls;
    }

    /**
     * Traverse each node in "stall_weights_map";
     * Use earch "_ctx_node.py_states.hash" is an index and search in stall_weights_map.first
     * 
    */
    void handle_aggregation(VIEW_CTX_MAP& ctx_node_map, STALL_WEIGHTS_MAP& stall_weights_map) {
      for (auto& ctx_node : ctx_node_map) {
        int map_size = ctx_node.map_size;
        for (int index = 0; index < map_size; index++) {
          for (auto &_states : ctx_node.py_states.at(index)) {
            pystates_hash_t _hash = (pystates_hash_t)std::stoul(_states.hash);
            ctx_node.total_stalls += get_pystate_total_stalls(stall_weights_map, _hash, ctx_node, index);
            
          }
        }
        // std::cout << ctx_node.total_stalls << std::endl;
        // std::cout << "/////" << std::endl;
      }
    }

    void aggregate_stall_weights_by_view_node(VIEW_CTX_MAP& ctx_node_map, STALL_WEIGHTS_MAP& stall_weights_map, const std::string &file_path, uint64_t& gpa_total_stall) {
      // Step 1: read gpa file
      int res = get_pc_stall_weights(file_path, gpa_total_stall, stall_weights_map);
      // Step 2: insert data into "ctx_node_map"
      handle_aggregation(ctx_node_map, stall_weights_map);
      // // step 3: recalculate gpa_total_stall
      // gpa_total_stall = 0;
      // for (auto& niter : ctx_node_map) {
      //   gpa_total_stall += niter.total_stalls;
      // }

      // std::cout << "//**Verify STALL_WEIGHTS_MAP**//" << std::endl;
      // for (auto & [_hash, pc_count] : stall_weights_map) {
      //   std::cout << _hash << std::endl;
      //   for (auto& [_pc, _count] : pc_count) {
      //     std::cout << "  " << std::hex << _pc << " "<< std::dec << _count << std::endl;
      //   }
      //   std::cout << std::endl;
      // }
    }

    static void outputContext(const std::string &file_name, const VIEW_CTX_MAP& ctx_node_map, STALL_WEIGHTS_MAP& stall_weights_map, uint64_t& gpa_total_stall) {
      std::ofstream out(file_name + ".context");
      for (auto& iter : ctx_node_map) {
        if (iter.access_type != 0) {
          continue;
        }
        out << iter.global_id << " / " << (iter.access_type == 0 ? "View Node" : "Memory Block") << std::endl;
        int map_size = iter.map_size;
        if(gpa_total_stall != 0) { 
          out << "life time stalls: " << iter.total_stalls << " (" << (double)(100 * iter.total_stalls/gpa_total_stall) << "%)"<< std::endl;
        }
        for (int i = 0; i < map_size; i++) {
          for (auto &_states : iter.py_states.at(i)){
            out << "arg index: " << _states.index << " num_states: " << _states.num_states << std::endl;
            out << " pystates_hash: " << _states.hash;
            if(gpa_total_stall != 0) {
              uint64_t pystate_stalls = get_pystate_total_stalls(stall_weights_map, (pystates_hash_t)std::stoul(_states.hash), iter, i);
              out << " stalls: " << pystate_stalls << " (" << (double)(100 * pystate_stalls/gpa_total_stall) << "%)";
            }
            out << std::endl;
            for (auto & _state : _states.python_contexts){
              out << "  " << _state.file_name << ":" << _state.function_name << ":" << _state.function_first_lineno << ":" << _state.lineno << std::endl;
            }
          }
          for (auto &_ctxs : iter.ctx_ids.at(i)){
            out << "ctx_id: " << _ctxs.ctx_id << std::endl;
            out << _ctxs.context; // << std::endl;
            // out << "pc:" << _ctxs.pcs;
            out << "pcs_size: " << _ctxs.pcs.size() << std::endl;
            out << "pc:";
            for (auto & pc : _ctxs.pcs){
              out << " " << std::hex << pc << std::dec;
            }
            out << std::endl;
          }
          out << std::endl;
        }
        out << "---------------------------------------------" << std::endl << std::endl;
      }
      out.close();
    }

    static void finish(VIEW_CTX_MAP& ctx_node_map) {
      for(auto& iter : ctx_node_map) {
        for (auto &piter: iter.py_states) {
          piter.clear();
        }
        for (auto &piter: iter.ctx_ids) {
          piter.clear();
        }
      }
    }


    void analyzeTorchViewMain(Prof::CallPath::Profile &prof, const std::vector<std::string> &torchViewFiles) {
      Prof::CallPath::CCTIdToCCTNodeMap cctNodeMap;

      Prof::CCT::ANodeIterator prof_it(prof.cct()->root(), NULL/*filter*/, false/*leavesOnly*/,
                                       IteratorStack::PreOrder);
      for (Prof::CCT::ANode *n = NULL; (n = prof_it.current()); ++prof_it) {
        Prof::CCT::ADynNode* n_dyn = dynamic_cast<Prof::CCT::ADynNode*>(n);
        if (n_dyn) {
          cctNodeMap.insert(std::make_pair(n_dyn->cpId(), n));
        }
      }
// start Debugging logs. remove later
      std:: cout << "SIZE: " << cctNodeMap.size() << std::endl;
      for (auto &iter : cctNodeMap){
        std:: cout << "cct_id: " << iter.first << std::endl;
      }
// end 

      for (auto &file : torchViewFiles) {
        // if (file.compare("torch_view_report.csv") != 0) {
        //   continue;
        // }

        VIEW_CTX_MAP view_ctx_map;
        STALL_WEIGHTS_MAP stall_weights_map; 
        uint64_t gpa_total_stall = 0;

        read_memory_node(file, view_ctx_map);

        matchCCTNode(cctNodeMap, view_ctx_map);

        aggregate_stall_weights_by_view_node(view_ctx_map, stall_weights_map, file, gpa_total_stall);

        outputContext(file, view_ctx_map, stall_weights_map, gpa_total_stall);

        finish(view_ctx_map);

      }
    }
  } // namespace CallPath
} // namespace Analysis
