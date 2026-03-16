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

#include "dfx_logic_sim_types.hpp"

#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/device_span.hpp>

#include <rmm/device_uvector.hpp>

#include <optional>
#include <string>
#include <tuple>

namespace dfx_logic_sim {

// Reads circuit edge list from a CSV file, creates a graph, and adjusts edge orders.
// The CSV file is expected to have (src, dst, order) columns, all INT32.
template <typename vertex_t, typename edge_t, typename weight_t>
CircuitGraphResult<vertex_t, edge_t> read_circuit_graph_from_csv_file(
  raft::handle_t& handle, std::string const& file_name);

// Reads loop_id, node_id pairs from a CSV file, renumbers node ids, and filters out
// tie, gnd, pwr, and z nodes. Returns {filtered_loop_ids, filtered_loop_nodes}.
template <typename vertex_t, typename cell_idx_t>
std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<vertex_t>>
read_and_filter_loop_id_node_id_pairs(raft::handle_t& handle,
                           raft::device_span<cell_idx_t const> node_cell_indices,
                           raft::device_span<vertex_t const> renumber_map,
                           std::string const& file_name,
                           vertex_t num_vertices);

// Reads the cell output table CSV file and returns the packed output tables and their offsets.
// The CSV is expected to have (cell_idx, output_state) columns sorted by cell_idx.
template <typename cell_idx_t, typename state_t>
cell_output_table_info<state_t>
read_cell_output_tables(raft::handle_t& handle, std::string const& file_name);

// Reads the node-to-cell index map CSV file and computes cell input degrees.
// The CSV is expected to have (node_idx, cell_idx) columns.
template <typename vertex_t, typename edge_t, typename cell_idx_t>
node_cell_map_result<vertex_t, edge_t, cell_idx_t>
read_node_cell_map(
  raft::handle_t& handle,
  std::string const& file_name,
  cugraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  vertex_t const* renumber_map_data,
  size_t num_cell_types);

// Reads a boolean flag CSV file with (cell_idx, flag) columns and returns a device vector
// of flags indexed by cell_idx. Entries not present in the file default to false.
template <typename cell_idx_t>
rmm::device_uvector<bool>
read_boolean_flags_from_csv_file(
  raft::handle_t& handle, std::string const& file_name, size_t num_entries);

// Reads all simulation input data from files (input patterns, output patterns, node states,
// sequential table indices)
template <typename vertex_t,
          typename edge_t,
          typename state_t,
          typename order_t,
          typename cell_idx_t,
          typename pattern_idx_t>
SimulationInputData<vertex_t> read_simulation_input_data(
  raft::handle_t& handle,
  HighResTimer& hr_timer,
  cugraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  DfxLogicSim_Usecase const& usecase,
  size_t num_patterns,
  NodeBuckets<vertex_t, pattern_idx_t>& node_buckets,
  node_cell_map_result<vertex_t, edge_t, cell_idx_t> const& node_cell_map,
  LevelizedNodeBuckets<vertex_t, pattern_idx_t> const& levelized_buckets,
  raft::device_span<vertex_t const> renumber_map,
  raft::device_span<uint32_t> node_states,
  raft::device_span<size_t const> idx_multipliers);

}  // namespace dfx_logic_sim
