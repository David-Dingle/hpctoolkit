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

    typedef struct TV_CTX_NODE{
      int32_t ctx_id;
      std::string context;

      TV_CTX_NODE() = default;

      TV_CTX_NODE(int32_t cid) : ctx_id(cid), context("") {}

      TV_CTX_NODE(const TV_CTX_NODE& rhs){
        this->ctx_id = rhs.ctx_id;
        this->context = std::string(rhs.context);
      }

      void operator=(const TV_CTX_NODE& rhs){
        this->ctx_id = rhs.ctx_id;
        this->context = std::string(rhs.context);
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

          if (!view_ctx_map.empty()) { // skip the first loop when the map is empty
            if (view_ctx_map.back().ctx_ids.empty()) { 
              view_ctx_map.pop_back();  // if the block is empty, remove the block
            } else {
              for (int it = (view_ctx_map.back().ctx_ids.size() - 1); it >= 0; --it) {  // check if any pystate got no ctx_id, then remove them both
                if (view_ctx_map.back().ctx_ids.at(it).empty() || (view_ctx_map.back().ctx_ids.at(it).size() == 1 
                                                                    && 
                                                                  view_ctx_map.back().ctx_ids.at(it).at(0).ctx_id == 0)) {
                  std::cout << "branch is taken" << std::endl;
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

          _ctx_set.clear();
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
          }
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


    static void outputContext(const std::string &file_name, const VIEW_CTX_MAP& ctx_node_map) {
      std::ofstream out(file_name + ".context");
      for (auto& iter : ctx_node_map) {
        out << iter.global_id << " / " << (iter.access_type == 0 ? "View Node" : "Memory Block") << std::endl;
        int map_size = iter.map_size;
        for (int i = 0; i < map_size; i++) {
          for (auto &_states : iter.py_states.at(i)){
            out << "arg index: " << _states.index << " num_states: " << _states.num_states << std::endl;
            out << " pystates_hash: " << _states.hash << std::endl;
            for (auto & _state : _states.python_contexts){
              out << "  " << _state.file_name << ":" << _state.function_name << ":" << _state.function_first_lineno << ":" << _state.lineno << std::endl;
            }
          }
          for (auto &_ctxs : iter.ctx_ids.at(i)){
            out << _ctxs.context << std::endl;
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

// VIEW_CTX_MAP view_ctx_map;

    void analyzeTorchViewMain(Prof::CallPath::Profile &prof, const std::vector<std::string> &torchViewFiles) {
      Prof::CallPath::CCTIdToCCTNodeMap cctNodeMap;

      Prof::CCT::ANodeIterator prof_it(prof.cct()->root(), NULL/*filter*/, false/*leavesOnly*/,
                                       IteratorStack::PreOrder);
      for (Prof::CCT::ANode *n = NULL; (n = prof_it.current()); ++prof_it) {
// start
       //n = n->ancestor(Prof::CCT::ANode::TyProcFrm); 
// end
        Prof::CCT::ADynNode* n_dyn = dynamic_cast<Prof::CCT::ADynNode*>(n);
        if (n_dyn) {
          cctNodeMap.insert(std::make_pair(n_dyn->cpId(), n));
        }
      }
// start
      std:: cout << "SIZE: " << cctNodeMap.size() << std::endl;
      for (auto &iter : cctNodeMap){
        std:: cout << "cct_id: " << iter.first << std::endl;
      }
// end 

      for (auto &file : torchViewFiles) {

        VIEW_CTX_MAP view_ctx_map;

        read_memory_node(file, view_ctx_map);

        matchCCTNode(cctNodeMap, view_ctx_map);

        outputContext(file, view_ctx_map);

        finish(view_ctx_map);

      }
    }
  } // namespace CallPath
} // namespace Analysis
