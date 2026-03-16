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

// Suppress false positive warnings from GCC's static analysis struggling with
// complex template instantiation in STL containers used by cuGraph primitives
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"

#include "dfx_logic_sim_io.hpp"
#include "prims/transform_e.cuh"
#include "utilities/base_fixture.hpp"
#include "utilities/test_graphs.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/graph_functions.hpp>

#include <cudf/io/csv.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuda/functional>
#include <cuda/std/functional>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>

#include <iostream>

namespace dfx_logic_sim {

template <typename vertex_t, typename edge_t, typename weight_t>
CircuitGraphResult<vertex_t, edge_t> read_circuit_graph_from_csv_file(raft::handle_t& handle,
                                                                      std::string const& file_name)
{
  using order_t     = edge_t;
  using edge_type_t = int32_t;
  using edge_time_t = int32_t;

  CircuitGraphResult<vertex_t, edge_t> result(handle);

  // Read edge list from CSV

  std::vector<cudf::data_type> const circuit_data_types = {
    cudf::data_type(cudf::type_id::INT32),
    cudf::data_type(cudf::type_id::INT32),
    cudf::data_type(cudf::type_id::INT32)};  // (src, dst, order)

  rmm::device_uvector<vertex_t> edgelist_srcs(0, handle.get_stream());
  rmm::device_uvector<vertex_t> edgelist_dsts(0, handle.get_stream());
  rmm::device_uvector<order_t> edgelist_orders(0, handle.get_stream());
  rmm::device_uvector<vertex_t> vertices(0, handle.get_stream());
  {
    cudf::io::csv_reader_options edge_opts =
      cudf::io::csv_reader_options::builder(cudf::io::source_info(file_name));
    edge_opts.set_dtypes(circuit_data_types);

    auto edge_data     = cudf::io::read_csv(edge_opts);
    auto edge_tbl_view = edge_data.tbl->view();

    auto num_edges = edge_tbl_view.num_rows();

    edgelist_srcs.resize(num_edges, handle.get_stream());
    edgelist_dsts.resize(num_edges, handle.get_stream());
    edgelist_orders.resize(num_edges, handle.get_stream());
    auto src_first   = edge_tbl_view.column(size_t{0}).data<vertex_t>();
    auto dst_first   = edge_tbl_view.column(size_t{1}).data<vertex_t>();
    auto order_first = edge_tbl_view.column(size_t{2}).data<order_t>();
    auto input_first = thrust::make_zip_iterator(src_first, dst_first, order_first);
    thrust::copy(handle.get_thrust_policy(),
                 input_first,
                 input_first + num_edges,
                 thrust::make_zip_iterator(
                   edgelist_srcs.begin(), edgelist_dsts.begin(), edgelist_orders.begin()));

    auto larger_v_first = cuda::make_transform_iterator(
      thrust::make_zip_iterator(src_first, dst_first),
      cuda::proclaim_return_type<vertex_t>([] __device__(auto pair) {
        return cuda::std::max(cuda::std::get<0>(pair), cuda::std::get<1>(pair));
      }));
    auto max_v =
      thrust::reduce(handle.get_thrust_policy(),
                     larger_v_first,
                     larger_v_first + num_edges,
                     vertex_t{0},
                     cuda::proclaim_return_type<vertex_t>(
                       [] __device__(auto lhs, auto rhs) { return cuda::std::max(lhs, rhs); }));
    vertices.resize(max_v + 1, handle.get_stream());
    thrust::sequence(handle.get_thrust_policy(), vertices.begin(), vertices.end(), vertex_t{0});
  }

  // Create graph from edge list

  {
    std::vector<cugraph::arithmetic_device_uvector_t> tmp_edge_property_vectors{};
    std::vector<cugraph::edge_arithmetic_property_t<edge_t>> tmp_edge_properties{};
    tmp_edge_property_vectors.push_back(std::move(edgelist_orders));
    std::optional<rmm::device_uvector<vertex_t>> tmp_renumber_map{};
    std::tie(result.graph, tmp_edge_properties, tmp_renumber_map) =
      cugraph::create_graph_from_edgelist<vertex_t, edge_t, store_transposed, multi_gpu>(
        handle,
        std::make_optional(std::move(vertices)),
        std::move(edgelist_srcs),
        std::move(edgelist_dsts),
        std::move(tmp_edge_property_vectors),
        cugraph::graph_properties_t{false /* symmetric */, true /* multi-graph */},
        renumber);
    result.edge_orders =
      std::move(std::get<cugraph::edge_property_t<edge_t, edge_t>>(tmp_edge_properties[0]));
    result.renumber_map = std::move(*tmp_renumber_map);
  }

  // Adjust edge order values (idx multipliers for each input pin is inverted => pin order 0 is the
  // MSB in the truth table inputs, its multiplier is in-degree - 1 - order)
  auto graph_view = result.graph.view();
  auto in_degrees = graph_view.compute_in_degrees(handle);
  {
    cugraph::edge_property_t<edge_t, order_t> tmp_edge_orders(handle, graph_view);
    cugraph::transform_e(
      handle,
      graph_view,
      cugraph::edge_src_dummy_property_t{}.view(),
      cugraph::edge_dst_dummy_property_t{}.view(),
      result.edge_orders.view(),
      cuda::proclaim_return_type<order_t>(
        [in_degrees   = raft::device_span<edge_t const>(in_degrees.data(), in_degrees.size()),
         renumber_map = raft::device_span<vertex_t const>(
           result.renumber_map.data(),
           result.renumber_map.size())] __device__(auto, auto dst, auto, auto, auto order) {
          auto max_order = static_cast<order_t>(in_degrees[dst] - 1);
          auto ret       = max_order - order;
          if (ret < 0) {
            printf(
              "dst: %d, renumber_map[dst]: %d, max_order: %d, order: %d, ret: %d, "
              "in_degrees[dst]: %d\n",
              dst,
              renumber_map[dst],
              max_order,
              order,
              ret,
              in_degrees[dst]);
          }
          return max_order - order;
        }),
      tmp_edge_orders.mutable_view());
    result.edge_orders = std::move(tmp_edge_orders);
  }

  return result;
}

template <typename vertex_t, typename cell_idx_t>
std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<vertex_t>>
read_and_filter_loop_id_node_id_pairs(raft::handle_t& handle,
                                      raft::device_span<cell_idx_t const> node_cell_indices,
                                      raft::device_span<vertex_t const> renumber_map,
                                      std::string const& file_name,
                                      vertex_t num_vertices)
{
  std::vector<cudf::data_type> const data_types = {
    cudf::data_type(cudf::type_id::INT32),
    cudf::data_type(cudf::type_id::INT32)};  // (loop_id, node_idx)

  cudf::io::csv_reader_options opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info(file_name));
  opts.set_dtypes(data_types);

  auto csv_data = cudf::io::read_csv(opts);
  auto tbl_view = csv_data.tbl->view();

  auto num_rows      = static_cast<size_t>(tbl_view.num_rows());
  auto loop_id_first = tbl_view.column(size_t{0}).data<int32_t>();
  auto node_id_first = tbl_view.column(size_t{1}).data<vertex_t>();

  rmm::device_uvector<int32_t> loop_ids(num_rows, handle.get_stream());
  thrust::copy(
    handle.get_thrust_policy(), loop_id_first, loop_id_first + num_rows, loop_ids.begin());

  rmm::device_uvector<vertex_t> renumbered_nodes(num_rows, handle.get_stream());
  thrust::copy(
    handle.get_thrust_policy(), node_id_first, node_id_first + num_rows, renumbered_nodes.begin());

  cugraph::renumber_ext_vertices<vertex_t, multi_gpu>(handle,
                                                      renumbered_nodes.data(),
                                                      renumbered_nodes.size(),
                                                      renumber_map.data(),
                                                      vertex_t{0},
                                                      num_vertices);

  // Filter tie, gnd, pwr or z nodes
  // node_ids are already renumbered above
  rmm::device_uvector<int32_t> filtered_loop_ids(num_rows, handle.get_stream());
  rmm::device_uvector<vertex_t> filtered_loop_nodes(num_rows, handle.get_stream());

  auto input_pair_first = thrust::make_zip_iterator(loop_ids.begin(), renumbered_nodes.begin());
  // This code assumes that loop_ids are sorted
  auto filtered_pair_first =
    thrust::make_zip_iterator(filtered_loop_ids.begin(), filtered_loop_nodes.begin());

  auto filtered_pair_end =
    thrust::copy_if(handle.get_thrust_policy(),
                    input_pair_first,
                    input_pair_first + num_rows,
                    filtered_pair_first,
                    [node_cell_indices,
                     tie_x_cell_idx_v  = tie_x_cell_idx,
                     pwr_cell_idx_v    = pwr_cell_idx,
                     ground_cell_idx_v = ground_cell_idx,
                     z_cell_idx_v      = z_cell_idx] __device__(auto pair) {
                      auto vertex_id = cuda::std::get<1>(pair);
                      auto cell_idx  = node_cell_indices[vertex_id];
                      return (cell_idx != tie_x_cell_idx_v) && (cell_idx != pwr_cell_idx_v) &&
                             (cell_idx != ground_cell_idx_v) && (cell_idx != z_cell_idx_v);
                    });

  auto filtered_count = cuda::std::distance(filtered_pair_first, filtered_pair_end);
  filtered_loop_nodes.resize(filtered_count, handle.get_stream());
  filtered_loop_nodes.shrink_to_fit(handle.get_stream());
  filtered_loop_ids.resize(filtered_count, handle.get_stream());
  filtered_loop_ids.shrink_to_fit(handle.get_stream());

  return std::make_tuple(std::move(filtered_loop_ids), std::move(filtered_loop_nodes));
}

template <typename cell_idx_t, typename state_t>
cell_output_table_info<state_t> read_cell_output_tables(raft::handle_t& handle,
                                                        std::string const& file_name)
{
  std::vector<cudf::data_type> const cell_output_table_data_types = {
    cudf::data_type(cudf::type_id::INT16),
    cudf::data_type(cudf::type_id::UINT8)};  // (cell_idx, output_state)

  cudf::io::csv_reader_options cell_output_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info(file_name));
  cell_output_opts.set_dtypes(cell_output_table_data_types);

  auto cell_output_data     = cudf::io::read_csv(cell_output_opts);
  auto cell_output_tbl_view = cell_output_data.tbl->view();

  auto num_rows           = static_cast<size_t>(cell_output_tbl_view.num_rows());
  auto cell_idx_first     = cell_output_tbl_view.column(size_t{0}).data<cell_idx_t>();
  auto output_state_first = cell_output_tbl_view.column(size_t{1}).data<state_t>();

  auto num_unique_cells =
    thrust::unique_count(handle.get_thrust_policy(), cell_idx_first, cell_idx_first + num_rows);
  rmm::device_uvector<cell_idx_t> unique_cell_indices(num_unique_cells, handle.get_stream());
  rmm::device_uvector<size_t> unique_cell_table_sizes(num_unique_cells, handle.get_stream());
  thrust::reduce_by_key(handle.get_thrust_policy(),
                        cell_idx_first,
                        cell_idx_first + num_rows,
                        cuda::make_constant_iterator(size_t{1}),
                        unique_cell_indices.begin(),
                        unique_cell_table_sizes.begin());
  size_t num_cell_types =
    thrust::reduce(handle.get_thrust_policy(),
                   unique_cell_indices.begin(),
                   unique_cell_indices.end(),
                   cell_idx_t{0},
                   cuda::proclaim_return_type<cell_idx_t>(
                     [] __device__(auto lhs, auto rhs) { return cuda::std::max(lhs, rhs); })) +
    1;
  size_t max_table_size =
    thrust::reduce(handle.get_thrust_policy(),
                   unique_cell_table_sizes.begin(),
                   unique_cell_table_sizes.end(),
                   size_t{0},
                   cuda::proclaim_return_type<size_t>(
                     [] __device__(auto lhs, auto rhs) { return cuda::std::max(lhs, rhs); }));

  std::cout << "Max table size: " << max_table_size << std::endl;

  rmm::device_uvector<size_t> cell_output_table_sizes(num_cell_types, handle.get_stream());
  thrust::fill(handle.get_thrust_policy(),
               cell_output_table_sizes.begin(),
               cell_output_table_sizes.end(),
               size_t{0});
  thrust::scatter(handle.get_thrust_policy(),
                  unique_cell_table_sizes.begin(),
                  unique_cell_table_sizes.end(),
                  unique_cell_indices.begin(),
                  cell_output_table_sizes.begin());
  rmm::device_uvector<size_t> cell_output_table_offsets(cell_output_table_sizes.size() + 1,
                                                        handle.get_stream());
  cell_output_table_offsets.set_element_to_zero_async(0, handle.get_stream());
  thrust::inclusive_scan(handle.get_thrust_policy(),
                         cell_output_table_sizes.begin(),
                         cell_output_table_sizes.end(),
                         cell_output_table_offsets.begin() + 1);

  rmm::device_uvector<state_t> cell_output_tables(
    cell_output_table_offsets.back_element(handle.get_stream()), handle.get_stream());
  thrust::copy(handle.get_thrust_policy(),
               output_state_first,
               output_state_first + num_rows,
               cell_output_tables.begin());

  cell_output_table_info<state_t> result(handle);
  result.tables         = std::move(cell_output_tables);
  result.offsets        = std::move(cell_output_table_offsets);
  result.num_cell_types = num_cell_types;
  result.max_table_size = max_table_size;
  return result;
}

template <typename vertex_t, typename edge_t, typename cell_idx_t>
node_cell_map_result<vertex_t, edge_t, cell_idx_t> read_node_cell_map(
  raft::handle_t& handle,
  std::string const& file_name,
  cugraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  vertex_t const* renumber_map_data,
  size_t num_cell_types)
{
  std::vector<cudf::data_type> const node2cell_map_data_types = {
    cudf::data_type(cudf::type_id::INT32),
    cudf::data_type(cudf::type_id::INT16)};  // (node_idx, cell_idx)

  cudf::io::csv_reader_options node2cell_map_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info(file_name));
  node2cell_map_opts.set_dtypes(node2cell_map_data_types);

  auto node2cell_map_data     = cudf::io::read_csv(node2cell_map_opts);
  auto node2cell_map_tbl_view = node2cell_map_data.tbl->view();

  auto num_rows = static_cast<size_t>(node2cell_map_tbl_view.num_rows());
  CUGRAPH_EXPECTS(num_rows == static_cast<size_t>(graph_view.number_of_vertices()),
                  "# nodes in the nodeId to cellIdx map file doesn't match the number of nodes "
                  "in the circuit.");
  auto node_first     = node2cell_map_tbl_view.column(size_t{0}).data<vertex_t>();
  auto cell_idx_first = node2cell_map_tbl_view.column(size_t{1}).data<cell_idx_t>();

  rmm::device_uvector<vertex_t> renumbered_nodes(num_rows, handle.get_stream());
  thrust::copy(
    handle.get_thrust_policy(), node_first, node_first + num_rows, renumbered_nodes.begin());
  cugraph::renumber_ext_vertices<vertex_t, multi_gpu>(handle,
                                                      renumbered_nodes.data(),
                                                      renumbered_nodes.size(),
                                                      renumber_map_data,
                                                      vertex_t{0},
                                                      graph_view.number_of_vertices());

  rmm::device_uvector<cell_idx_t> node_cell_indices(graph_view.number_of_vertices(),
                                                    handle.get_stream());
  thrust::fill(handle.get_thrust_policy(),
               node_cell_indices.begin(),
               node_cell_indices.end(),
               invalid_cell_idx);
  thrust::scatter(handle.get_thrust_policy(),
                  cell_idx_first,
                  cell_idx_first + num_rows,
                  renumbered_nodes.begin(),
                  node_cell_indices.begin());

  rmm::device_uvector<edge_t> cell_input_degrees(num_cell_types, handle.get_stream());
  {
    thrust::fill(
      handle.get_thrust_policy(), cell_input_degrees.begin(), cell_input_degrees.end(), edge_t{0});
    auto in_degrees = graph_view.compute_in_degrees(handle);
    rmm::device_uvector<cell_idx_t> tmp_cell_indices(graph_view.number_of_vertices(),
                                                     handle.get_stream());
    rmm::device_uvector<edge_t> tmp_in_degrees(tmp_cell_indices.size(), handle.get_stream());
    auto tmp_pair_first =
      thrust::make_zip_iterator(tmp_cell_indices.begin(), tmp_in_degrees.begin());
    thrust::tabulate(handle.get_thrust_policy(),
                     tmp_pair_first,
                     tmp_pair_first + tmp_cell_indices.size(),
                     cuda::proclaim_return_type<cuda::std::tuple<cell_idx_t, edge_t>>(
                       [node_cell_indices = raft::device_span<cell_idx_t const>(
                          node_cell_indices.data(), node_cell_indices.size()),
                        in_degrees = raft::device_span<edge_t const>(
                          in_degrees.data(), in_degrees.size())] __device__(size_t v) {
                         return cuda::std::make_tuple(node_cell_indices[v], in_degrees[v]);
                       }));
    tmp_cell_indices.resize(
      cuda::std::distance(
        tmp_pair_first,
        thrust::remove_if(
          handle.get_thrust_policy(),
          tmp_pair_first,
          tmp_pair_first + tmp_cell_indices.size(),
          cuda::proclaim_return_type<bool>([] __device__(auto pair) {
            return cuda::std::get<0>(pair) <
                   cell_idx_t{0};  // negative cell IDs for speical nodes (input/output/ground)
          }))),
      handle.get_stream());
    tmp_in_degrees.resize(tmp_cell_indices.size(), handle.get_stream());
    thrust::sort_by_key(handle.get_thrust_policy(),
                        tmp_cell_indices.begin(),
                        tmp_cell_indices.end(),
                        tmp_in_degrees.begin());
    auto unique_pair_last = thrust::unique_by_key(handle.get_thrust_policy(),
                                                  tmp_cell_indices.begin(),
                                                  tmp_cell_indices.end(),
                                                  tmp_in_degrees.begin());
    auto num_unique_pairs =
      cuda::std::distance(tmp_cell_indices.begin(), cuda::std::get<0>(unique_pair_last));
    thrust::scatter(handle.get_thrust_policy(),
                    tmp_in_degrees.begin(),
                    tmp_in_degrees.begin() + num_unique_pairs,
                    tmp_cell_indices.begin(),
                    cell_input_degrees.begin());
  }

  node_cell_map_result<vertex_t, edge_t, cell_idx_t> result(handle);
  result.node_cell_indices  = std::move(node_cell_indices);
  result.cell_input_degrees = std::move(cell_input_degrees);
  return result;
}

template <typename cell_idx_t>
rmm::device_uvector<bool> read_boolean_flags_from_csv_file(raft::handle_t& handle,
                                                           std::string const& file_name,
                                                           size_t num_entries)
{
  std::vector<cudf::data_type> const flag_data_types = {
    cudf::data_type(cudf::type_id::INT16),
    cudf::data_type(cudf::type_id::UINT8)};  // (cell_idx, boolean flag)

  cudf::io::csv_reader_options flag_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info(file_name));
  flag_opts.set_dtypes(flag_data_types);

  auto flag_data     = cudf::io::read_csv(flag_opts);
  auto flag_tbl_view = flag_data.tbl->view();

  auto num_rows       = static_cast<size_t>(flag_tbl_view.num_rows());
  auto cell_idx_first = flag_tbl_view.column(size_t{0}).data<cell_idx_t>();
  auto flag_first     = flag_tbl_view.column(size_t{1}).data<uint8_t>();

  rmm::device_uvector<bool> flags(num_entries, handle.get_stream());
  thrust::fill(handle.get_thrust_policy(), flags.begin(), flags.end(), false);
  thrust::scatter(
    handle.get_thrust_policy(), flag_first, flag_first + num_rows, cell_idx_first, flags.begin());
  return flags;
}

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
  raft::device_span<size_t const> idx_multipliers)
{
  // Extract struct fields as local aliases to keep the rest of the body unchanged
  auto const& input_node_value_file_name           = usecase.input_node_value_file_name;
  auto const& output_node_value_file_name          = usecase.output_node_value_file_name;
  auto const& setup_node_states_file_name          = usecase.setup_node_states_file_name;
  auto const& setup_sequential_table_idx_file_name = usecase.setup_sequential_table_idx_file_name;
  auto num_input_nodes                             = node_buckets.num_input_nodes;
  auto num_output_nodes                            = node_buckets.num_output_nodes;
  auto num_output_nodes_used                       = node_buckets.num_output_nodes_used;
  auto& seq_node_bucket                            = node_buckets.seq_node_bucket;
  auto const& renumbered_input_nodes_sorted_bucket_indices_map =
    node_buckets.renumbered_input_nodes_sorted_bucket_indices_map;
  auto const& renumbered_output_nodes_sorted_bucket_indices_map =
    node_buckets.renumbered_output_nodes_sorted_bucket_indices_map;
  auto node_cell_indices = raft::device_span<cell_idx_t const>(
    node_cell_map.node_cell_indices.data(), node_cell_map.node_cell_indices.size());
  auto cell_input_degrees = raft::device_span<edge_t const>(
    node_cell_map.cell_input_degrees.data(), node_cell_map.cell_input_degrees.size());
  auto all_latch_nodes = raft::device_span<vertex_t const>(
    levelized_buckets.all_latch_nodes.data(), levelized_buckets.all_latch_nodes.size());

  SimulationInputData<vertex_t> result(handle);

  std::vector<cudf::data_type> const input_nodes_table_data_types = {
    cudf::data_type(cudf::type_id::INT32),
    cudf::data_type(cudf::type_id::INT32),
    cudf::data_type(cudf::type_id::INT32),
    cudf::data_type(cudf::type_id::UINT8)};  // (cycle_idx, pattern_idx, PI_idx, state)

  std::vector<cudf::data_type> const output_nodes_table_data_types = {
    cudf::data_type(cudf::type_id::INT32),
    cudf::data_type(cudf::type_id::INT32),
    cudf::data_type(cudf::type_id::INT32),
    cudf::data_type(cudf::type_id::UINT8)};  // (cycle_idx, pattern_idx, PO_idx, state)

  std::vector<cudf::data_type> const node_states_data_types = {
    cudf::data_type(cudf::type_id::INT32),
    cudf::data_type(cudf::type_id::UINT8)};  // (node_id, state)

  std::vector<cudf::data_type> const old_seq_node_output_table_indices_data_types = {
    cudf::data_type(cudf::type_id::INT32),
    cudf::data_type(cudf::type_id::INT64)};  // (node_id, table_idx)

  /*1. read input and output files*/
  if (cugraph::test::g_perf) {
    RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
    hr_timer.start("Read input/(expected-)output node value file");
  }

  auto num_input_node_state_words_per_cycle_per_pattern =
    (num_input_nodes + num_states_per_word - 1) / num_states_per_word;
  auto num_output_node_state_words_per_cycle_per_pattern =
    (num_output_nodes_used + num_states_per_word - 1) / num_states_per_word;

  // expected_output_node_states will be populated only if output file is provided

  {  // assumes one row per clock cycle
    {
      std::string file_name =
        cugraph::test::get_rapids_dataset_root_dir() + "/" + input_node_value_file_name;
      cudf::io::csv_reader_options input_node_value_opts =
        cudf::io::csv_reader_options::builder(cudf::io::source_info(file_name));
      input_node_value_opts.set_dtypes(input_nodes_table_data_types);

      auto input_node_value_data     = cudf::io::read_csv(input_node_value_opts);
      auto input_node_value_tbl_view = input_node_value_data.tbl->view();

      auto cycle_idx_first   = input_node_value_tbl_view.column(size_t{0}).data<int32_t>();
      auto pattern_idx_first = input_node_value_tbl_view.column(size_t{1}).data<int32_t>();
      auto PI_idx_first      = input_node_value_tbl_view.column(size_t{2}).data<int32_t>();
      auto state_first       = input_node_value_tbl_view.column(size_t{3}).data<state_t>();

      auto num_rows = static_cast<size_t>(input_node_value_tbl_view.num_rows());

      rmm::device_uvector<int32_t> PI_input_bucket_indices(num_rows, handle.get_stream());
      auto input_node_kv_store = renumbered_input_nodes_sorted_bucket_indices_map.view();
      input_node_kv_store.find(PI_idx_first,
                               PI_idx_first + num_rows,
                               PI_input_bucket_indices.begin(),
                               handle.get_stream());

      result.num_cycles = thrust::reduce(handle.get_thrust_policy(),
                                         cycle_idx_first,
                                         cycle_idx_first + num_rows,
                                         int32_t{0},
                                         cuda::maximum<int32_t>()) +
                          1;
      auto num_patterns_file = thrust::reduce(handle.get_thrust_policy(),
                                              pattern_idx_first,
                                              pattern_idx_first + num_rows,
                                              int32_t{0},
                                              cuda::maximum<int32_t>()) +
                               1;

      std::cout << "num_patterns_file: " << num_patterns_file << std::endl;

      result.input_node_states.resize(
        result.num_cycles * num_patterns * num_input_node_state_words_per_cycle_per_pattern,
        handle.get_stream());

      uint32_t all_x_state{0};
      for (size_t i = 0; i < num_states_per_word; i++) {
        all_x_state |= state_t{2} << (num_bits_per_state * i);
      }

      thrust::fill(handle.get_thrust_policy(),
                   result.input_node_states.begin(),
                   result.input_node_states.end(),
                   all_x_state);

      auto input_data_firsts =
        thrust::make_zip_iterator(cycle_idx_first, PI_input_bucket_indices.begin(), state_first);

      auto num_cycles = result.num_cycles;
      thrust::for_each(
        handle.get_thrust_policy(),
        input_data_firsts,
        input_data_firsts + num_rows,
        [input_node_states = raft::device_span<uint32_t>(result.input_node_states.data(),
                                                         result.input_node_states.size()),
         num_cycles,
         num_patterns,
         num_input_node_state_words_per_cycle_per_pattern,
         num_states_per_word_v = num_states_per_word,
         num_bits_per_state_v  = num_bits_per_state] __device__(auto data) {
          auto cycle                 = cuda::std::get<0>(data);
          auto PI_input_bucket_index = cuda::std::get<1>(data);
          auto state                 = cuda::std::get<2>(data);

          for (auto pattern = 0; pattern < num_patterns; pattern++) {
            auto word_idx =
              cycle * num_patterns * num_input_node_state_words_per_cycle_per_pattern +
              pattern * num_input_node_state_words_per_cycle_per_pattern +
              PI_input_bucket_index / num_states_per_word_v;
            auto j = PI_input_bucket_index % num_states_per_word_v;

            // Clear the 2 bits first, then set the new state
            auto clear_mask = ~(uint32_t{0x3} << (num_bits_per_state_v * j));
            auto set_mask   = state << (num_bits_per_state_v * j);

            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> word(input_node_states[word_idx]);
            word.fetch_and(clear_mask, cuda::std::memory_order_relaxed);  // Clear the 2 bits
            word.fetch_or(set_mask, cuda::std::memory_order_relaxed);     // Set the new state
          }
        });
    }

    if (output_node_value_file_name.has_value()) {
      // output nodes
      std::string file_name =
        cugraph::test::get_rapids_dataset_root_dir() + "/" + output_node_value_file_name.value();
      cudf::io::csv_reader_options output_node_value_opts =
        cudf::io::csv_reader_options::builder(cudf::io::source_info(file_name));
      output_node_value_opts.set_dtypes(output_nodes_table_data_types);

      auto output_node_value_data     = cudf::io::read_csv(output_node_value_opts);
      auto output_node_value_tbl_view = output_node_value_data.tbl->view();

      auto cycle_idx_first   = output_node_value_tbl_view.column(size_t{0}).data<int32_t>();
      auto pattern_idx_first = output_node_value_tbl_view.column(size_t{1}).data<int32_t>();
      auto PO_idx_first      = output_node_value_tbl_view.column(size_t{2}).data<int32_t>();
      auto state_first       = output_node_value_tbl_view.column(size_t{3}).data<state_t>();

      auto num_rows = static_cast<size_t>(output_node_value_tbl_view.num_rows());

      std::cout << "num_rows: " << num_rows << std::endl;
      std::cout << "num_cycles: " << result.num_cycles << std::endl;
      std::cout << "num_patterns: " << num_patterns << std::endl;
      std::cout << "num_output_nodes: " << num_output_nodes << std::endl;

      rmm::device_uvector<uint32_t> tmp_expected_output_node_states(
        result.num_cycles * num_patterns * num_output_node_state_words_per_cycle_per_pattern,
        handle.get_stream());
      thrust::fill(handle.get_thrust_policy(),
                   tmp_expected_output_node_states.begin(),
                   tmp_expected_output_node_states.end(),
                   uint32_t{0});

      auto ouput_node_IDs_first = cuda::make_transform_iterator(
        PO_idx_first,
        cugraph::detail::shift_right_t<vertex_t>{graph_view.number_of_vertices() -
                                                 static_cast<vertex_t>(num_output_nodes)});

      rmm::device_uvector<int32_t> PO_output_bucket_indices(num_rows, handle.get_stream());
      auto output_kv_store = renumbered_output_nodes_sorted_bucket_indices_map.view();
      output_kv_store.find(ouput_node_IDs_first,
                           ouput_node_IDs_first + num_rows,
                           PO_output_bucket_indices.begin(),
                           handle.get_stream());

      auto output_data_firsts =
        thrust::make_zip_iterator(cycle_idx_first, PO_output_bucket_indices.begin(), state_first);

      auto num_cycles = result.num_cycles;
      thrust::for_each(
        handle.get_thrust_policy(),
        output_data_firsts,
        output_data_firsts + num_rows,
        [expected_output_node_states = raft::device_span<uint32_t>(
           tmp_expected_output_node_states.data(), tmp_expected_output_node_states.size()),
         num_cycles,
         num_patterns,
         num_output_node_state_words_per_cycle_per_pattern,
         num_states_per_word_v = num_states_per_word,
         num_bits_per_state_v  = num_bits_per_state] __device__(auto data) {
          auto cycle                  = cuda::std::get<0>(data);
          auto PO_output_bucket_index = cuda::std::get<1>(data);
          auto state                  = cuda::std::get<2>(data);

          for (auto pattern = 0; pattern < num_patterns; pattern++) {
            auto word_idx =
              cycle * num_patterns * num_output_node_state_words_per_cycle_per_pattern +
              pattern * num_output_node_state_words_per_cycle_per_pattern +
              PO_output_bucket_index / num_states_per_word_v;
            auto j    = PO_output_bucket_index % num_states_per_word_v;
            auto mask = state << (num_bits_per_state_v * j);

            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> word(
              expected_output_node_states[word_idx]);
            word.fetch_or(mask, cuda::std::memory_order_relaxed);
          }
        });

      result.expected_output_node_states = std::move(tmp_expected_output_node_states);
    }
  }

  if (cugraph::test::g_perf) {
    RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
    hr_timer.stop();
    hr_timer.display_and_clear(std::cout);
  }

  /* 3. Initialize node states - either from file or with default X states */
  auto num_words_per_vertex = (num_patterns + num_states_per_word - 1) / num_states_per_word;

  // First, fill with all X states
  uint32_t all_x_state{0};
  for (size_t i = 0; i < num_states_per_word; i++) {
    all_x_state |= state_t{2} << (num_bits_per_state * i);
  }
  thrust::fill(handle.get_thrust_policy(), node_states.begin(), node_states.end(), all_x_state);

  // If setup_node_states_file_name is provided, load and apply the initial states
  if (setup_node_states_file_name.has_value()) {
    std::string node_states_file =
      cugraph::test::get_rapids_dataset_root_dir() + "/" + setup_node_states_file_name.value();
    cudf::io::csv_reader_options node_states_opts =
      cudf::io::csv_reader_options::builder(cudf::io::source_info(node_states_file));
    node_states_opts.set_dtypes(node_states_data_types);

    auto node_states_data_csv = cudf::io::read_csv(node_states_opts);
    auto node_states_tbl_view = node_states_data_csv.tbl->view();

    auto node_id_first = node_states_tbl_view.column(size_t{0}).data<vertex_t>();
    auto state_first   = node_states_tbl_view.column(size_t{1}).data<state_t>();
    auto num_rows      = static_cast<size_t>(node_states_tbl_view.num_rows());

    std::cout << "num_rows (node states): " << num_rows << std::endl;

    // Copy to device vectors for renumbering
    rmm::device_uvector<vertex_t> file_node_ids(num_rows, handle.get_stream());
    rmm::device_uvector<state_t> file_states(num_rows, handle.get_stream());

    thrust::copy(
      handle.get_thrust_policy(), node_id_first, node_id_first + num_rows, file_node_ids.begin());
    thrust::copy(
      handle.get_thrust_policy(), state_first, state_first + num_rows, file_states.begin());

    // Renumber the node IDs
    cugraph::renumber_ext_vertices<vertex_t, multi_gpu>(
      handle,
      file_node_ids.data(),
      file_node_ids.size(),
      renumber_map.data(),
      graph_view.local_vertex_partition_range_first(),
      graph_view.local_vertex_partition_range_last());

    // Apply the states to node_states
    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(num_rows * num_patterns),
      [node_states,
       file_node_ids_span =
         raft::device_span<vertex_t const>(file_node_ids.data(), file_node_ids.size()),
       file_states_span = raft::device_span<state_t const>(file_states.data(), file_states.size()),
       num_rows,
       num_vertices = graph_view.number_of_vertices(),
       num_patterns,
       num_words_per_vertex,
       num_states_per_word_v = num_states_per_word,
       num_bits_per_state_v  = num_bits_per_state,
       state_mask_v          = state_mask] __device__(size_t i) {
        auto row_idx     = i % num_rows;
        auto pattern_idx = i / num_rows;
        auto vertex_id   = file_node_ids_span[row_idx];
        auto state       = file_states_span[row_idx];

        auto word_idx =
          static_cast<size_t>(vertex_id) * static_cast<size_t>(num_words_per_vertex) +
          static_cast<size_t>(pattern_idx) / static_cast<size_t>(num_states_per_word_v);
        auto intra_pattern_node_idx =
          static_cast<size_t>(pattern_idx) % static_cast<size_t>(num_states_per_word_v);
        auto clear_mask = ~(state_mask_v << (num_bits_per_state_v * intra_pattern_node_idx));
        auto set_mask   = state << (num_bits_per_state_v * intra_pattern_node_idx);

        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> word(node_states[word_idx]);

        word.fetch_and(clear_mask, cuda::std::memory_order_relaxed);
        word.fetch_or(set_mask, cuda::std::memory_order_relaxed);
      });

    std::cout << "Loaded " << num_rows << " node states from file and replicated across "
              << num_patterns << " patterns" << std::endl;
  }

  /* 4. Initialize sequential and latch node output table indices - either from file or with
   * default X states */
  result.old_seq_node_output_table_indices.resize(seq_node_bucket.size(), handle.get_stream());
  result.old_latch_node_output_table_indices.resize(all_latch_nodes.size() * num_patterns,
                                                    handle.get_stream());

  if (setup_sequential_table_idx_file_name.has_value()) {
    std::cout << "Loading initial sequential node output table indices from file..." << std::endl;

    std::string old_seq_indices_file = cugraph::test::get_rapids_dataset_root_dir() + "/" +
                                       setup_sequential_table_idx_file_name.value();
    cudf::io::csv_reader_options old_seq_indices_opts =
      cudf::io::csv_reader_options::builder(cudf::io::source_info(old_seq_indices_file));
    old_seq_indices_opts.set_dtypes(old_seq_node_output_table_indices_data_types);

    auto old_seq_indices_data     = cudf::io::read_csv(old_seq_indices_opts);
    auto old_seq_indices_tbl_view = old_seq_indices_data.tbl->view();

    auto node_id_first   = old_seq_indices_tbl_view.column(size_t{0}).data<vertex_t>();
    auto table_idx_first = old_seq_indices_tbl_view.column(size_t{1}).data<int64_t>();
    auto num_rows        = static_cast<size_t>(old_seq_indices_tbl_view.num_rows());

    std::cout << "Read " << num_rows << " sequential node entries from file" << std::endl;

    // Copy to device vectors for processing
    rmm::device_uvector<vertex_t> file_seq_node_ids(num_rows, handle.get_stream());
    rmm::device_uvector<size_t> file_seq_table_indices(num_rows, handle.get_stream());

    thrust::copy(handle.get_thrust_policy(),
                 node_id_first,
                 node_id_first + num_rows,
                 file_seq_node_ids.begin());
    thrust::transform(handle.get_thrust_policy(),
                      table_idx_first,
                      table_idx_first + num_rows,
                      file_seq_table_indices.begin(),
                      cuda::proclaim_return_type<size_t>(
                        [] __device__(int64_t idx) { return static_cast<size_t>(idx); }));

    // Renumber the node IDs
    cugraph::renumber_ext_vertices<vertex_t, multi_gpu>(
      handle,
      file_seq_node_ids.data(),
      file_seq_node_ids.size(),
      renumber_map.data(),
      graph_view.local_vertex_partition_range_first(),
      graph_view.local_vertex_partition_range_last());

    // Sort by vertex IDs for kv_store
    rmm::device_uvector<vertex_t> sorted_seq_vertex_ids(num_rows, handle.get_stream());
    rmm::device_uvector<size_t> sorted_seq_table_indices(num_rows, handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 file_seq_node_ids.begin(),
                 file_seq_node_ids.end(),
                 sorted_seq_vertex_ids.begin());
    thrust::copy(handle.get_thrust_policy(),
                 file_seq_table_indices.begin(),
                 file_seq_table_indices.end(),
                 sorted_seq_table_indices.begin());
    thrust::sort_by_key(handle.get_thrust_policy(),
                        sorted_seq_vertex_ids.begin(),
                        sorted_seq_vertex_ids.end(),
                        sorted_seq_table_indices.begin());

    // Create kv_store for lookup
    cugraph::kv_store_t<vertex_t, size_t, true> seq_node_table_idx_map(
      sorted_seq_vertex_ids.begin(),
      sorted_seq_vertex_ids.end(),
      sorted_seq_table_indices.begin(),
      std::numeric_limits<size_t>::max(),
      false,
      handle.get_stream());

    auto seq_node_table_idx_map_view = seq_node_table_idx_map.view();
    auto device_view =
      cugraph::detail::kv_binary_search_store_device_view_t(seq_node_table_idx_map_view);

    // Apply the table indices
    thrust::transform(
      handle.get_thrust_policy(),
      seq_node_bucket.begin(),
      seq_node_bucket.end(),
      result.old_seq_node_output_table_indices.begin(),
      cuda::proclaim_return_type<size_t>(
        [device_view,
         node_cell_indices,
         cell_input_degrees,
         idx_multipliers,
         invalid_value = std::numeric_limits<size_t>::max()] __device__(auto tagged_v) {
          auto v = cuda::std::get<0>(tagged_v);
          auto p = cuda::std::get<1>(tagged_v);

          auto table_idx = device_view.find(v);

          if (table_idx != invalid_value) {
            return table_idx;
          } else {
            auto cell_idx = node_cell_indices[v];
            auto order    = cell_input_degrees[cell_idx] + 1;
            return idx_multipliers[order] - size_t{1};
          }
        }));

    std::cout << "Loaded sequential node table indices and applied to all " << num_patterns
              << " patterns" << std::endl;

    // Initialize latch node output table indices from the same file
    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(all_latch_nodes.size() * num_patterns),
      [all_latch_nodes,
       old_latch_indices =
         raft::device_span<size_t>(result.old_latch_node_output_table_indices.data(),
                                   result.old_latch_node_output_table_indices.size()),
       device_view,
       node_cell_indices,
       cell_input_degrees,
       idx_multipliers,
       num_patterns,
       invalid_value = std::numeric_limits<size_t>::max()] __device__(size_t flat_idx) {
        auto pos       = flat_idx / num_patterns;
        auto v         = all_latch_nodes[pos];
        auto table_idx = device_view.find(v);
        if (table_idx != invalid_value) {
          old_latch_indices[flat_idx] = table_idx;
        } else {
          auto cell_idx               = node_cell_indices[v];
          auto order                  = cell_input_degrees[cell_idx] + 1;
          old_latch_indices[flat_idx] = idx_multipliers[order] - size_t{1};
        }
      });

    std::cout << "Loaded latch node table indices for " << all_latch_nodes.size() << " latch nodes"
              << std::endl;

  } else {
    // Default behavior: Initialize to all X states
    thrust::transform(handle.get_thrust_policy(),
                      seq_node_bucket.begin(),
                      seq_node_bucket.end(),
                      result.old_seq_node_output_table_indices.begin(),
                      cuda::proclaim_return_type<size_t>(
                        [node_cell_indices,
                         cell_input_degrees,
                         idx_multipliers,
                         num_vertices = graph_view.number_of_vertices()] __device__(auto tagged_v) {
                          auto v        = cuda::std::get<0>(tagged_v);
                          auto p        = cuda::std::get<1>(tagged_v);
                          auto cell_idx = node_cell_indices[v];
                          auto order    = cell_input_degrees[cell_idx] + 1 /* output */;
                          return idx_multipliers[order] - size_t{1};  // all states in X (2)
                        }));

    // Initialize latch node output table indices to all X states
    thrust::for_each(handle.get_thrust_policy(),
                     thrust::make_counting_iterator(size_t{0}),
                     thrust::make_counting_iterator(all_latch_nodes.size() * num_patterns),
                     [all_latch_nodes,
                      old_latch_indices = raft::device_span<size_t>(
                        result.old_latch_node_output_table_indices.data(),
                        result.old_latch_node_output_table_indices.size()),
                      node_cell_indices,
                      cell_input_degrees,
                      idx_multipliers,
                      num_patterns] __device__(size_t flat_idx) {
                       auto pos                    = flat_idx / num_patterns;
                       auto v                      = all_latch_nodes[pos];
                       auto cell_idx               = node_cell_indices[v];
                       auto order                  = cell_input_degrees[cell_idx] + 1;
                       old_latch_indices[flat_idx] = idx_multipliers[order] - size_t{1};
                     });
  }

  return result;
}

// Explicit template instantiations

template CircuitGraphResult<int32_t, int32_t>
read_circuit_graph_from_csv_file<int32_t, int32_t, float>(raft::handle_t&, std::string const&);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
read_and_filter_loop_id_node_id_pairs<int32_t, int16_t>(raft::handle_t&,
                                                        raft::device_span<int16_t const>,
                                                        raft::device_span<int32_t const>,
                                                        std::string const&,
                                                        int32_t);

template cell_output_table_info<uint8_t> read_cell_output_tables<int16_t, uint8_t>(
  raft::handle_t&, std::string const&);

template node_cell_map_result<int32_t, int32_t, int16_t>
read_node_cell_map<int32_t, int32_t, int16_t>(
  raft::handle_t&,
  std::string const&,
  cugraph::graph_view_t<int32_t, int32_t, true, false> const&,
  int32_t const*,
  size_t);

template rmm::device_uvector<bool> read_boolean_flags_from_csv_file<int16_t>(raft::handle_t&,
                                                                             std::string const&,
                                                                             size_t);

template SimulationInputData<int32_t>
read_simulation_input_data<int32_t, int32_t, uint8_t, int32_t, int16_t, int32_t>(
  raft::handle_t&,
  HighResTimer&,
  cugraph::graph_view_t<int32_t, int32_t, true, false> const&,
  DfxLogicSim_Usecase const&,
  size_t,
  NodeBuckets<int32_t, int32_t>&,
  node_cell_map_result<int32_t, int32_t, int16_t> const&,
  LevelizedNodeBuckets<int32_t, int32_t> const&,
  raft::device_span<int32_t const>,
  raft::device_span<uint32_t>,
  raft::device_span<size_t const>);

}  // namespace dfx_logic_sim

#pragma GCC diagnostic pop
