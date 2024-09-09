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

#include "CallPath-TorchMonitor.hpp"

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

namespace Analysis {

namespace CallPath {

struct CTX_NODE{
  int32_t ctx_id;
  std::string context;
  std::string type;
  uint64_t count; // invocation counts // for memory is access kernel counts

  CTX_NODE() = default;

  CTX_NODE(int32_t cid) : ctx_id(cid), context(""), type(""), count(0) {}
};

typedef std::map<int32_t, CTX_NODE> CTX_NODE_MAP;

static bool proces_memory_node(CTX_NODE_MAP &ctx_node_map, int32_t cid, std::string ctx_type) {
  auto memory_node = ctx_node_map.find(cid);

  if (memory_node == ctx_node_map.end()) {
    CTX_NODE node(cid);
    node.type = ctx_type;
    node.count++;
    ctx_node_map.emplace(cid, node);
  } else {
    memory_node->second.count++;
  }

  return false;
}


static void read_memory_node(const std::string &file_name, CTX_NODE_MAP &ctx_node_map) {
  std::ifstream file(file_name);
  std::string word;
  bool memory_flag = false;
  bool kernel_flag = false;
  bool free_flag = false;
  bool set_flag = false;
  bool copyt_flag = false;
  bool copyf_flag = false;

  while (file >> word) {
    // std::cout << "word: " << word << std::endl;
    
    if (memory_flag) {
      int32_t cid = std::stoi(word);
      memory_flag = proces_memory_node(ctx_node_map, cid, "Malloc");
    }
    if (word == "ALLOC") { // this if-clause should after previous if-clause
      memory_flag = true;
    }

    if (kernel_flag) {
      int32_t cid = std::stoi(word);
      kernel_flag = proces_memory_node(ctx_node_map, cid, "Kernel"); 
    }
    if (word == "ACCESS") {
      kernel_flag = true;
    }

    if (free_flag) {
      int32_t cid = std::stoi(word);
      free_flag = proces_memory_node(ctx_node_map, cid, "Free");  
    }
    if (word == "FREE") {
      free_flag = true;
    }

    if (set_flag) {
      int32_t cid = std::stoi(word);
      set_flag = proces_memory_node(ctx_node_map, cid, "Set");
    }
    if (word == "SET") {
      set_flag = true;
    }

    if (copyt_flag) {
      int32_t cid = std::stoi(word);
      copyt_flag = proces_memory_node(ctx_node_map, cid, "CopyToDM");
    }
    if (word == "COPYT") {
      copyt_flag = true;
    }

    if (copyf_flag) {
      int32_t cid = std::stoi(word);
      copyf_flag = proces_memory_node(ctx_node_map, cid, "CopyFromDM");
    }
    if (word == "COPYF") {
      copyf_flag = true;
    }

  }

  file.close();

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

static void matchCCTNode(Prof::CallPath::CCTIdToCCTNodeMap &cctNodeMap, CTX_NODE_MAP &ctx_node_map) { 
  // match nodes
  for (auto &iter : ctx_node_map) {
    auto &node = iter.second;
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
      std::stack<Prof::CCT::ProcFrm *> st;
      Prof::CCT::ProcFrm *proc_frm = NULL;
      std::string cct_context;

      if (cct->type() != Prof::CCT::ANode::TyProcFrm &&
        cct->type() != Prof::CCT::ANode::TyRoot) {
        proc_frm = cct->ancestorProcFrm(); 

        if (proc_frm != NULL) {
          auto *strct = cct->structure();
          if (strct->ancestorAlien()) {
            auto alien_st = getInlineStack(strct);
            for (auto &name : alien_st) {
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

      while (st.empty() == false) {
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
              for (auto &name : alien_st) {
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


static void outputContext(const std::string &file_name, const CTX_NODE_MAP &ctx_node_map) {
  std::ofstream out(file_name + ".context");
  for (auto iter : ctx_node_map) {
    out << iter.first << "  " << iter.second.type << "  " \
        << iter.second.count << " count(s)"<< std::endl;
    // out << iter.first << "  " << iter.second.type << std::endl;
    out << iter.second.context << std::endl;
  }

  out.close();
}



void analyzeTorchMonitorMain(Prof::CallPath::Profile &prof, const std::vector<std::string> &torch_monitor_files) {
  Prof::CallPath::CCTIdToCCTNodeMap cctNodeMap;

  Prof::CCT::ANodeIterator prof_it(prof.cct()->root(), NULL/*filter*/, false/*leavesOnly*/,
    IteratorStack::PreOrder);
  for (Prof::CCT::ANode *n = NULL; (n = prof_it.current()); ++prof_it) {
// start setting 2
        // Invoke [overlayStaticStructure] (load per module if needed); get cubin node n's ancestors recursively 
    // n = n->ancestor(Prof::CCT::ANode::TyProcFrm); 
// end setting 2
    Prof::CCT::ADynNode* n_dyn = dynamic_cast<Prof::CCT::ADynNode*>(n);
    if (n_dyn) {
      cctNodeMap.insert(std::make_pair(n_dyn->cpId(), n));
    }
  }
  // start
      std::cout << "SIZE: " << cctNodeMap.size() << std::endl;
      for (auto &iter : cctNodeMap){
        std::cout << "cct_id: " << iter.first << std::endl;
      }
// end 

  for (auto &file : torch_monitor_files) {
    
    CTX_NODE_MAP ctx_node_map;

    read_memory_node(file, ctx_node_map);

    matchCCTNode(cctNodeMap, ctx_node_map);

    outputContext(file, ctx_node_map);
    
  }
}

} // namespace CallPath

} // namespace Analysis