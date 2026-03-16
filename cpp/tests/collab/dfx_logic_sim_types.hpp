/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "prims/kv_store.cuh"
#include "prims/vertex_frontier.cuh"

#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct DfxLogicSim_Usecase {
  std::string circuit_file_name{};            // (src, dst, order) triplets
  std::string cell_output_table_file_name{};  // cell (gate) output table file name
  std::string node2cell_map_file_name{};      // node Idx => cell (gate) Idx file name
  std::string
    seq_cell_flag_file_name{};         // cell (gate) Idx => true (1) if sequential else false (0)
  std::string latch_flag_file_name{};  // cell (gate) Idx => true (1) if latch else false (0)
  std::string zstate_nodes_flag_file_name{}; // cell (gate) Idx => true (1) if node can produce z values as outputs else false (0)
  std::string loop_id_node_id_pairs_file_name{};  // loopId, nodeId pairs file name

  std::string input_node_value_file_name{};  // (cycle, pattern, PI_idx, state) for all input nodes
  std::optional<std::string>
    output_node_value_file_name{};  // (cycle, pattern, PO_idx, state) for all output nodes
  std::optional<std::string>
    used_output_indices_file_name{};  // num_ouput_nodes can be larger than number of
                                      // output nodes used for comparison; this file contains the
                                      // indices of the output nodes actually used for output
                                      // comparison
                                      // FIXME: This is the same as PO_idx in the output file, we
                                      // have a separate file as IO logic comes after pre-processing
                                      // This is used to filter the output_node_bucket during
                                      // pre-processing

  std::optional<std::string>
    setup_node_states_file_name{};  // node_id, state pairs; stores states of all nodes in the graph
                                    // after running test setup for ATPG simulations
  std::optional<std::string>
    setup_sequential_table_idx_file_name{};  // node_id, table_idx; stores prev state for all
                                             // sequential nodes in the circuit from test setup for
                                             // ATPG simulations
};

namespace dfx_logic_sim {

// Graph configuration constants
constexpr bool store_transposed         = true;
constexpr bool multi_gpu                = false;
constexpr bool renumber                 = true;
constexpr bool sorted_unique_key_bucket = true;

// State packing constants
constexpr size_t num_valid_values_per_state = 3;
constexpr size_t num_bits_per_state         = 2;
constexpr size_t num_states_per_word = (sizeof(uint32_t) * 8) / num_bits_per_state;
constexpr uint32_t state_mask        = (uint32_t{1} << num_bits_per_state) - uint32_t{1};

// Special cell index constants (using int16_t as cell_idx_t is always int16_t)
constexpr int16_t invalid_cell_idx = -1;
constexpr int16_t input_cell_idx   = -2;
constexpr int16_t output_cell_idx  = -3;
constexpr int16_t ground_cell_idx  = -4;
constexpr int16_t pwr_cell_idx     = -5;
constexpr int16_t tie_x_cell_idx   = -6;
constexpr int16_t z_cell_idx       = -7;

template <typename vertex_t, typename edge_t, typename cell_idx_t>
struct CoarsenGraphResult {
  cugraph::graph_t<vertex_t, edge_t, !store_transposed, multi_gpu>
    coarsen_graph;  // Coarsen graph in which all combinational loops are
                    // condensed to super nodes; used for levelization
  rmm::device_uvector<vertex_t>
    coarsen_renumber_map;  // Renumber map for the coarsen graph;
                           // coarsen graph ids => graph vertex ids map
  rmm::device_uvector<cell_idx_t>
    coarsen_node_cell_indices;  // Cell indices for the nodes in the coarsen graph

  rmm::device_uvector<int32_t> unique_loop_ids;  // Unique loop ids in the loop file
  rmm::device_uvector<vertex_t>
    loop_labels;  // Label for each unique loop id; label is the min vertex for all nodes within a
                  // loop id id for all nodes with this loop id

  CoarsenGraphResult(raft::handle_t& handle)
    : coarsen_graph(handle),
      coarsen_renumber_map(0, handle.get_stream()),
      coarsen_node_cell_indices(0, handle.get_stream()),
      unique_loop_ids(0, handle.get_stream()),
      loop_labels(0, handle.get_stream())
  {
  }
};

// Struct to hold simulation input data read from files
template <typename vertex_t>
struct SimulationInputData {
  rmm::device_uvector<uint32_t> input_node_states;  // Packed input node states per cycle/pattern
  std::optional<rmm::device_uvector<uint32_t>>
    expected_output_node_states;  // Packed expected output states (optional)
  // Sequential node output table indices
  rmm::device_uvector<size_t> old_seq_node_output_table_indices;
  // Latch node output table indices
  rmm::device_uvector<size_t> old_latch_node_output_table_indices;
  size_t num_cycles;  // Number of clock cycles

  SimulationInputData(raft::handle_t& handle)
    : input_node_states(0, handle.get_stream()),
      expected_output_node_states(std::nullopt),
      old_seq_node_output_table_indices(0, handle.get_stream()),
      old_latch_node_output_table_indices(0, handle.get_stream()),
      num_cycles(0)
  {
  }
};

template <typename vertex_t, typename order_t>
struct LoopDataResult {
  rmm::device_uvector<size_t> loop_id_nodes_offsets;
  rmm::device_uvector<vertex_t> loop_edge_srcs;
  rmm::device_uvector<vertex_t> loop_edge_dsts;
  rmm::device_uvector<order_t> loop_edge_orders;
  rmm::device_uvector<size_t> loop_id_edge_offsets;
  std::vector<rmm::device_uvector<int32_t>> levelized_loop_ids;

  LoopDataResult(raft::handle_t& handle)
    : loop_id_nodes_offsets(0, handle.get_stream()),
      loop_edge_srcs(0, handle.get_stream()),
      loop_edge_dsts(0, handle.get_stream()),
      loop_edge_orders(0, handle.get_stream()),
      loop_id_edge_offsets(0, handle.get_stream()),
      levelized_loop_ids()
  {
  }
};

template <typename vertex_t, typename edge_t>
struct CircuitGraphResult {
  cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu> graph;
  cugraph::edge_property_t<edge_t, edge_t> edge_orders;  // order_t == edge_t
  rmm::device_uvector<vertex_t> renumber_map;

  CircuitGraphResult(raft::handle_t& handle)
    : graph(handle), edge_orders(handle), renumber_map(0, handle.get_stream())
  {
  }
};

template <typename vertex_t, typename pattern_idx_t>
struct NodeBuckets {
  cugraph::key_bucket_t<vertex_t, void, multi_gpu, sorted_unique_key_bucket> input_node_bucket;
  cugraph::key_bucket_t<vertex_t, void, multi_gpu, sorted_unique_key_bucket> ground_node_bucket;
  cugraph::key_bucket_t<vertex_t, void, multi_gpu, sorted_unique_key_bucket> pwr_node_bucket;
  cugraph::key_bucket_t<vertex_t, void, multi_gpu, sorted_unique_key_bucket> tie_x_node_bucket;
  cugraph::key_bucket_t<vertex_t, void, multi_gpu, sorted_unique_key_bucket> z_node_bucket;
  cugraph::key_bucket_t<vertex_t, void, multi_gpu, sorted_unique_key_bucket> output_node_bucket;
  cugraph::key_bucket_t<vertex_t, pattern_idx_t, multi_gpu, false> seq_node_bucket;
  cugraph::kv_store_t<vertex_t, size_t, true> renumbered_input_nodes_sorted_bucket_indices_map;
  cugraph::kv_store_t<vertex_t, size_t, true> renumbered_output_nodes_sorted_bucket_indices_map;
  rmm::device_uvector<vertex_t> output_node_inputs;
  size_t num_input_nodes;
  size_t num_output_nodes;
  size_t num_output_nodes_used;

  NodeBuckets(raft::handle_t& handle)
    : input_node_bucket(handle),
      ground_node_bucket(handle),
      pwr_node_bucket(handle),
      tie_x_node_bucket(handle),
      z_node_bucket(handle),
      output_node_bucket(handle),
      seq_node_bucket(handle),
      renumbered_input_nodes_sorted_bucket_indices_map(handle.get_stream()),
      renumbered_output_nodes_sorted_bucket_indices_map(handle.get_stream()),
      output_node_inputs(0, handle.get_stream()),
      num_input_nodes(0),
      num_output_nodes(0),
      num_output_nodes_used(0)
  {
  }
};

template <typename vertex_t, typename pattern_idx_t>
struct LevelizedNodeBuckets {
  std::vector<
    cugraph::key_bucket_t<vertex_t, pattern_idx_t, multi_gpu, sorted_unique_key_bucket>>
    levelized_comb_node_buckets;
  std::vector<cugraph::key_bucket_t<int32_t, pattern_idx_t, multi_gpu, false>>
    levelized_comb_node_loop_ids_buckets;
  std::vector<
    cugraph::key_bucket_t<vertex_t, pattern_idx_t, multi_gpu, sorted_unique_key_bucket>>
    levelized_latch_node_buckets;
  rmm::device_uvector<vertex_t> all_latch_nodes;
  rmm::device_uvector<size_t> latch_node_to_offset_map;
  size_t num_comb_nodes;
  size_t num_all_latch_nodes;
  size_t num_latch_nodes;

  LevelizedNodeBuckets(raft::handle_t& handle)
    : levelized_comb_node_buckets(),
      levelized_comb_node_loop_ids_buckets(),
      levelized_latch_node_buckets(),
      all_latch_nodes(0, handle.get_stream()),
      latch_node_to_offset_map(0, handle.get_stream()),
      num_comb_nodes(0),
      num_all_latch_nodes(0),
      num_latch_nodes(0)
  {
  }
};

template <typename state_t>
struct cell_output_table_info {
  rmm::device_uvector<state_t> tables;
  rmm::device_uvector<size_t> offsets;
  size_t num_cell_types;
  size_t max_table_size;

  cell_output_table_info(raft::handle_t& handle)
    : tables(0, handle.get_stream()),
      offsets(0, handle.get_stream()),
      num_cell_types(0),
      max_table_size(0)
  {
  }
};

template <typename vertex_t, typename edge_t, typename cell_idx_t>
struct node_cell_map_result {
  rmm::device_uvector<cell_idx_t> node_cell_indices;
  rmm::device_uvector<edge_t> cell_input_degrees;

  node_cell_map_result(raft::handle_t& handle)
    : node_cell_indices(0, handle.get_stream()),
      cell_input_degrees(0, handle.get_stream())
  {
  }
};

struct CellTypeBooleanFlags {
  rmm::device_uvector<bool> seq_cell_flags;
  rmm::device_uvector<bool> zstate_nodes_flags;
  rmm::device_uvector<bool> latch_flags;

  CellTypeBooleanFlags(raft::handle_t& handle)
    : seq_cell_flags(0, handle.get_stream()),
      zstate_nodes_flags(0, handle.get_stream()),
      latch_flags(0, handle.get_stream())
  {
  }
};

}  // namespace dfx_logic_sim
