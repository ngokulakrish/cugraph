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
 * See the License for the specific language governin_from_mtxg permissions and
 * limitations under the License.
 */

// Suppress false positive warnings from GCC's static analysis struggling with
// complex template instantiation in STL containers used by cuGraph primitives
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"

#include "dfx_logic_sim_io.hpp"
#include "dfx_logic_sim_types.hpp"
#include "prims/extract_transform_if_e.cuh"
#include "prims/extract_transform_if_v_frontier_outgoing_e.cuh"
#include "prims/fill_edge_property.cuh"
#include "prims/kv_store.cuh"
#include "prims/per_v_transform_reduce_incoming_outgoing_e.cuh"
#include "prims/reduce_op.cuh"
#include "prims/transform_e.cuh"
#include "prims/vertex_frontier.cuh"
#include "utilities/base_fixture.hpp"
#include "utilities/test_graphs.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

#include <cuda/std/functional>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

class Tests_DfxLogicSim : public ::testing::TestWithParam<DfxLogicSim_Usecase> {
 public:
  Tests_DfxLogicSim() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

 private:
  // Aliases to namespace constants (preserves trailing-underscore names used by simulation code)
  static constexpr auto store_transposed_         = dfx_logic_sim::store_transposed;
  static constexpr auto multi_gpu_                = dfx_logic_sim::multi_gpu;
  static constexpr auto renumber_                 = dfx_logic_sim::renumber;
  static constexpr auto sorted_unique_key_bucket_ = dfx_logic_sim::sorted_unique_key_bucket;

  static constexpr auto num_valid_values_per_state_ = dfx_logic_sim::num_valid_values_per_state;
  static constexpr auto num_bits_per_state_         = dfx_logic_sim::num_bits_per_state;
  static constexpr auto num_states_per_word_        = dfx_logic_sim::num_states_per_word;
  static constexpr auto state_mask_                 = dfx_logic_sim::state_mask;

  static constexpr auto invalid_cell_idx_ = dfx_logic_sim::invalid_cell_idx;
  static constexpr auto input_cell_idx_   = dfx_logic_sim::input_cell_idx;
  static constexpr auto output_cell_idx_  = dfx_logic_sim::output_cell_idx;
  static constexpr auto ground_cell_idx_  = dfx_logic_sim::ground_cell_idx;
  static constexpr auto pwr_cell_idx_     = dfx_logic_sim::pwr_cell_idx;
  static constexpr auto tie_x_cell_idx_   = dfx_logic_sim::tie_x_cell_idx;
  static constexpr auto z_cell_idx_       = dfx_logic_sim::z_cell_idx;

 public:
  template <typename vertex_t, typename edge_t, typename cell_idx_t>
  using CoarsenGraphResult = dfx_logic_sim::CoarsenGraphResult<vertex_t, edge_t, cell_idx_t>;
  template <typename vertex_t>
  using SimulationInputData = dfx_logic_sim::SimulationInputData<vertex_t>;
  template <typename vertex_t, typename order_t>
  using LoopDataResult = dfx_logic_sim::LoopDataResult<vertex_t, order_t>;
  template <typename vertex_t, typename edge_t>
  using CircuitGraphResult = dfx_logic_sim::CircuitGraphResult<vertex_t, edge_t>;
  template <typename vertex_t, typename pattern_idx_t>
  using NodeBuckets = dfx_logic_sim::NodeBuckets<vertex_t, pattern_idx_t>;
  template <typename vertex_t, typename pattern_idx_t>
  using LevelizedNodeBuckets = dfx_logic_sim::LevelizedNodeBuckets<vertex_t, pattern_idx_t>;
  template <typename state_t>
  using cell_output_table_info = dfx_logic_sim::cell_output_table_info<state_t>;
  template <typename vertex_t, typename edge_t, typename cell_idx_t>
  using node_cell_map_result = dfx_logic_sim::node_cell_map_result<vertex_t, edge_t, cell_idx_t>;
  using CellTypeBooleanFlags = dfx_logic_sim::CellTypeBooleanFlags;

  // Creates a coarsened graph from loop information and transposes it for levelization
  template <typename vertex_t, typename edge_t, typename weight_t, typename cell_idx_t>
  CoarsenGraphResult<vertex_t, edge_t, cell_idx_t> create_coarsen_graph(
    raft::handle_t& handle,
    cugraph::graph_view_t<vertex_t, edge_t, true, false> const graph_view,
    raft::device_span<int32_t const> filtered_loop_ids,
    raft::device_span<vertex_t const> filtered_loop_nodes,
    raft::device_span<cell_idx_t const> node_cell_indices)
  {
    CoarsenGraphResult<vertex_t, edge_t, cell_idx_t> result(handle);

    auto filtered_count = filtered_loop_ids.size();

    result.unique_loop_ids.resize(filtered_count, handle.get_stream());
    result.loop_labels.resize(filtered_count, handle.get_stream());

    // 1.1 Find the min vertexId for each loopId to use as label for coarsen graph
    auto ends = thrust::reduce_by_key(handle.get_thrust_policy(),
                                      filtered_loop_ids.data(),
                                      filtered_loop_ids.data() + filtered_count,
                                      filtered_loop_nodes.data(),
                                      result.unique_loop_ids.begin(),
                                      result.loop_labels.begin(),
                                      cuda::std::equal_to<int32_t>{},
                                      thrust::minimum<vertex_t>());

    result.unique_loop_ids.resize(cuda::std::distance(result.unique_loop_ids.begin(), ends.first),
                                  handle.get_stream());
    result.unique_loop_ids.shrink_to_fit(handle.get_stream());
    result.loop_labels.resize(cuda::std::distance(result.loop_labels.begin(), ends.second),
                              handle.get_stream());
    result.loop_labels.shrink_to_fit(handle.get_stream());

    std::cout << "Number of combinational loops in the graph: " << result.unique_loop_ids.size()
              << std::endl;

    rmm::device_uvector<int32_t> coarsen_graph_labels(graph_view.number_of_vertices(),
                                                      handle.get_stream());
    thrust::sequence(handle.get_thrust_policy(),
                     coarsen_graph_labels.begin(),
                     coarsen_graph_labels.end(),
                     int32_t{0});

    // 1.2 Assign each row (loop_id, node_id) pair the correct label
    auto label_first = cuda::make_transform_iterator(
      filtered_loop_ids.data(),
      cuda::proclaim_return_type<int32_t>(
        [unique_loop_ids    = raft::device_span<int32_t const>(result.unique_loop_ids.data(),
                                                            result.unique_loop_ids.size()),
         unique_loop_labels = raft::device_span<int32_t const>(
           result.loop_labels.data(), result.loop_labels.size())] __device__(auto loop_id) {
          auto idx = cuda::std::distance(
            unique_loop_ids.begin(),
            thrust::lower_bound(
              thrust::seq, unique_loop_ids.begin(), unique_loop_ids.end(), loop_id));
          return unique_loop_labels[idx];
        }));
    thrust::scatter(handle.get_thrust_policy(),
                    label_first,
                    label_first + filtered_count,
                    filtered_loop_nodes.data(),
                    coarsen_graph_labels.begin());

    /*2. Create a Coarsen Graph using the loop labels*/

    auto [coarse_graph, ignored_edge_weights, coarsen_renumber_map] =
      cugraph::coarsen_graph<vertex_t, edge_t, float, store_transposed_, multi_gpu_>(
        handle, graph_view, std::nullopt, coarsen_graph_labels.data(), true);

    auto coarse_graph_view = coarse_graph.view();

    std::cout << "Number of vertices in the coarsen graph: "
              << coarse_graph_view.number_of_vertices() << std::endl;

    /* 2.1 Get node_cell_indices for the new coarsen graph */
    rmm::device_uvector<cell_idx_t> coarsen_node_cell_indices(
      coarse_graph_view.number_of_vertices(), handle.get_stream());
    thrust::gather(handle.get_thrust_policy(),
                   coarsen_renumber_map->begin(),
                   coarsen_renumber_map->end(),
                   node_cell_indices.data(),
                   coarsen_node_cell_indices.begin());

    /* 2.2 Transpose storage for coarsen graph to run levelization*/
    std::tie(result.coarsen_graph, std::ignore, coarsen_renumber_map) =
      cugraph::transpose_graph_storage<vertex_t, edge_t, weight_t, store_transposed_, multi_gpu_>(
        handle, std::move(coarse_graph), std::nullopt, std::move(coarsen_renumber_map));

    result.coarsen_renumber_map = std::move(*coarsen_renumber_map);

    auto coarsen_graph_view = result.coarsen_graph.view();

    /* 2.3 Get node_cell_indices for the coarsen graph after transposing storage*/
    result.coarsen_node_cell_indices.resize(coarsen_graph_view.number_of_vertices(),
                                            handle.get_stream());

    thrust::gather(handle.get_thrust_policy(),
                   result.coarsen_renumber_map.begin(),
                   result.coarsen_renumber_map.end(),
                   node_cell_indices.data(),
                   result.coarsen_node_cell_indices.begin());

    return result;
  }

  // Perform Levelization
  // Returns {last_levels, num_levels}
  template <typename vertex_t, typename edge_t, typename cell_idx_t>
  std::tuple<rmm::device_uvector<size_t>, size_t> perform_levelization(
    raft::handle_t& handle,
    CoarsenGraphResult<vertex_t, edge_t, cell_idx_t> const& coarsen_result,
    raft::device_span<bool const> seq_cell_flags,
    size_t num_original_vertices)
  {
    rmm::device_uvector<size_t> last_levels(num_original_vertices, handle.get_stream());

    auto coarsen_graph_view = coarsen_result.coarsen_graph.view();

    /* 1. Seed nodes for levelized implementation are sequential nodes and pseudo input nodes */
    thrust::fill(handle.get_thrust_policy(),
                 last_levels.begin(),
                 last_levels.end(),
                 std::numeric_limits<size_t>::max());

    rmm::device_uvector<vertex_t> active_nodes(coarsen_graph_view.number_of_vertices(),
                                               handle.get_stream());

    auto num_active_nodes = cuda::std::distance(
      active_nodes.begin(),
      thrust::copy_if(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(vertex_t{0}),
        thrust::make_counting_iterator(coarsen_graph_view.number_of_vertices()),
        coarsen_result.coarsen_node_cell_indices.begin(),
        active_nodes.begin(),
        [seq_cell_flags, input_cell_idx = input_cell_idx_] __device__(auto idx) {
          return (idx == input_cell_idx) ||
                 ((idx >= 0) &&
                  seq_cell_flags[idx]);  // Excludes special node types like pwr/gnd/tiex/z
        }));

    active_nodes.resize(num_active_nodes, handle.get_stream());
    active_nodes.shrink_to_fit(handle.get_stream());

    /* 2. From each level, find the outgoing vertices, and process them if they are combinational
     * nodes */
    auto level = 0;
    while (true) {
      cugraph::vertex_frontier_t<vertex_t, void, multi_gpu_, sorted_unique_key_bucket_>
      node_frontier(handle, size_t{1});
      node_frontier.bucket(0) =
        cugraph::key_bucket_t<vertex_t, void, multi_gpu_, sorted_unique_key_bucket_>(
          handle, raft::device_span<vertex_t const>(active_nodes.data(), active_nodes.size()));

      auto nexts = cugraph::extract_transform_if_v_frontier_outgoing_e(
        handle,
        coarsen_graph_view,
        node_frontier.bucket(0),
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        cugraph::edge_dummy_property_t{}.view(),
        cuda::proclaim_return_type<vertex_t>(
          [] __device__(auto, auto dst, auto, auto, auto) { return dst; }),
        cuda::proclaim_return_type<bool>(
          [coarsen_node_cell_indices =
             raft::device_span<cell_idx_t const>(coarsen_result.coarsen_node_cell_indices.data(),
                                                 coarsen_result.coarsen_node_cell_indices.size()),
           seq_cell_flags,
           coarsen_renumber_map = raft::device_span<vertex_t const>(
             coarsen_result.coarsen_renumber_map.data(),
             coarsen_result.coarsen_renumber_map
               .size())] __device__(auto src, auto dst, auto, auto, auto) {
            return (src != dst) && (coarsen_node_cell_indices[dst] >= 0) &&
                   !seq_cell_flags[coarsen_node_cell_indices[dst]];
          }));

      thrust::sort(handle.get_thrust_policy(), nexts.begin(), nexts.end());
      nexts.resize(
        cuda::std::distance(nexts.begin(),
                            thrust::unique(handle.get_thrust_policy(), nexts.begin(), nexts.end())),
        handle.get_stream());

      thrust::for_each(
        handle.get_thrust_policy(),
        nexts.begin(),
        nexts.end(),
        [last_levels_span     = raft::device_span<size_t>(last_levels.data(), last_levels.size()),
         coarsen_renumber_map = raft::device_span<vertex_t const>(
           coarsen_result.coarsen_renumber_map.data(), coarsen_result.coarsen_renumber_map.size()),
         level = level] __device__(auto node) {
          last_levels_span[coarsen_renumber_map[node]] = level;
        });

      if (nexts.size() == 0) { break; }

      ++level;
      active_nodes = std::move(nexts);
    }

    std::cout << "Number of levels in the graph: " << level << std::endl;

    return std::make_tuple(std::move(last_levels), static_cast<size_t>(level));
  }

  // Build Loop Data Structures, create loop node offsets, edge lists, and levelized loop IDs
  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            typename order_t,
            typename edge_type_t,
            typename cell_idx_t>
  LoopDataResult<vertex_t, order_t> build_loop_data_structures(
    raft::handle_t& handle,
    cugraph::graph_view_t<vertex_t, edge_t, true, false> const graph_view,
    cugraph::edge_property_view_t<edge_t, order_t const*> const& edge_order_view,
    CoarsenGraphResult<vertex_t, edge_t, cell_idx_t> const& coarsen_result,
    raft::device_span<int32_t const> filtered_loop_ids,
    raft::device_span<vertex_t const> filtered_loop_nodes,
    raft::device_span<size_t> last_levels,
    size_t num_levels,
    raft::device_span<cell_idx_t const> node_cell_indices,
    raft::device_span<bool const> seq_cell_flags)
  {
    LoopDataResult<vertex_t, order_t> result(handle);

    auto num_rows      = filtered_loop_ids.size();
    auto loop_id_first = filtered_loop_ids.data();

    /*1. Create loop nodes offsets and loop srcs/dsts offsets*/
    rmm::device_uvector<int32_t> loop_ids(num_rows, handle.get_stream());
    rmm::device_uvector<size_t> loop_node_counts(num_rows, handle.get_stream());

    auto count_end = thrust::reduce_by_key(handle.get_thrust_policy(),
                                           loop_id_first,
                                           loop_id_first + num_rows,
                                           cuda::make_constant_iterator(size_t{1}),
                                           loop_ids.begin(),
                                           loop_node_counts.begin());

    loop_ids.resize(cuda::std::distance(loop_ids.begin(), count_end.first), handle.get_stream());
    loop_node_counts.resize(cuda::std::distance(loop_node_counts.begin(), count_end.second),
                            handle.get_stream());

    // FIXME: This code assumes loopIDs are contiguous from 0 to n-1. If loopIDs are not
    // contiguous or start from a value other than 0, the offset indexing will be incorrect.
    rmm::device_uvector<size_t> loop_id_nodes_offsets(loop_ids.size() + 1, handle.get_stream());
    thrust::fill(handle.get_thrust_policy(),
                 loop_id_nodes_offsets.begin(),
                 loop_id_nodes_offsets.begin() + 1,
                 size_t{0});
    thrust::inclusive_scan(handle.get_thrust_policy(),
                           loop_node_counts.begin(),
                           loop_node_counts.end(),
                           loop_id_nodes_offsets.begin() + 1);

    /*1.1 Extract edges where dst is a loop node*/
    rmm::device_uvector<vertex_t> sorted_loop_nodes(filtered_loop_nodes.size(),
                                                    handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 filtered_loop_nodes.data(),
                 filtered_loop_nodes.data() + filtered_loop_nodes.size(),
                 sorted_loop_nodes.begin());
    thrust::sort(handle.get_thrust_policy(), sorted_loop_nodes.begin(), sorted_loop_nodes.end());

    auto [loop_edge_srcs, loop_edge_dsts, edge_orders_list] = extract_transform_if_e(
      handle,
      graph_view,
      cugraph::edge_src_dummy_property_t{}.view(),
      cugraph::edge_dst_dummy_property_t{}.view(),
      edge_order_view,
      cuda::proclaim_return_type<cuda::std::tuple<vertex_t, vertex_t, order_t>>(
        [] __device__(auto src, auto dst, auto, auto, auto order) {
          return cuda::std::make_tuple(src, dst, order);
        }),
      cuda::proclaim_return_type<bool>(
        [node_cell_indices =
           raft::device_span<cell_idx_t const>(node_cell_indices.data(), node_cell_indices.size()),
         seq_cell_flags =
           raft::device_span<bool const>(seq_cell_flags.data(), seq_cell_flags.size()),
         sorted_loop_nodes = raft::device_span<vertex_t const>(
           sorted_loop_nodes.data(),
           sorted_loop_nodes.size())] __device__(auto src, auto dst, auto, auto, auto) {
          auto dst_comb = (node_cell_indices[dst] >= 0) &&
                          !seq_cell_flags[node_cell_indices[dst]];  // Excludes special node types
                                                                    // like pwr/gnd/tiex/z/inp/out

          if (!dst_comb) { return false; }

          auto dst_loop = thrust::binary_search(
            thrust::seq, sorted_loop_nodes.begin(), sorted_loop_nodes.end(), dst);
          if (!dst_loop) { return false; }
          return true;
        }));

    /*1.2 Create a vertex Id -> loopId map for all the vertices in loop nodes*/

    rmm::device_uvector<vertex_t> kv_loop_nodes(filtered_loop_nodes.size(), handle.get_stream());
    rmm::device_uvector<int32_t> kv_loop_Ids(filtered_loop_nodes.size(), handle.get_stream());

    thrust::copy(handle.get_thrust_policy(),
                 filtered_loop_nodes.data(),
                 filtered_loop_nodes.data() + filtered_loop_nodes.size(),
                 kv_loop_nodes.begin());
    thrust::copy(
      handle.get_thrust_policy(), loop_id_first, loop_id_first + num_rows, kv_loop_Ids.begin());

    thrust::sort_by_key(
      handle.get_thrust_policy(), kv_loop_nodes.begin(), kv_loop_nodes.end(), kv_loop_Ids.begin());

    cugraph::kv_store_t<vertex_t, int32_t, true> vertex_loop_id_map(
      kv_loop_nodes.begin(),
      kv_loop_nodes.end(),
      kv_loop_Ids.begin(),
      std::numeric_limits<int32_t>::max(),
      false,
      handle.get_stream());

    /*1.3 For each edge, find the correponding loopId of dst using the kv store*/

    rmm::device_uvector<int32_t> edge_loop_ids(loop_edge_dsts.size(), handle.get_stream());
    auto vertex_loop_id_map_view = vertex_loop_id_map.view();

    vertex_loop_id_map_view.find(
      loop_edge_dsts.begin(), loop_edge_dsts.end(), edge_loop_ids.begin(), handle.get_stream());

    /*1.4 Sort edges based on loopIds and compute offsets*/

    auto key_first = thrust::make_zip_iterator(edge_loop_ids.begin(), loop_edge_dsts.begin());
    auto edge_pair_first =
      thrust::make_zip_iterator(loop_edge_srcs.begin(), edge_orders_list.begin());
    thrust::sort_by_key(
      handle.get_thrust_policy(), key_first, key_first + loop_edge_dsts.size(), edge_pair_first);

    rmm::device_uvector<int32_t> edge_count_keys(loop_edge_srcs.size(), handle.get_stream());
    rmm::device_uvector<size_t> edge_counts(loop_edge_srcs.size(), handle.get_stream());

    auto edge_count_end = thrust::reduce_by_key(handle.get_thrust_policy(),
                                                edge_loop_ids.begin(),
                                                edge_loop_ids.end(),
                                                cuda::make_constant_iterator(size_t{1}),
                                                edge_count_keys.begin(),
                                                edge_counts.begin());

    size_t num_loops_with_edges =
      cuda::std::distance(edge_count_keys.begin(), edge_count_end.first);
    edge_counts.resize(num_loops_with_edges, handle.get_stream());
    edge_counts.shrink_to_fit(handle.get_stream());

    // FIXME: This code assumes loopIDs are contiguous from 0 to n-1. If loopIDs are not
    // contiguous or start from a value other than 0, the offset indexing will be incorrect.
    rmm::device_uvector<size_t> loop_id_edge_offsets(num_loops_with_edges + 1, handle.get_stream());
    thrust::fill(handle.get_thrust_policy(),
                 loop_id_edge_offsets.begin(),
                 loop_id_edge_offsets.begin() + 1,
                 size_t{0});
    thrust::inclusive_scan(handle.get_thrust_policy(),
                           edge_counts.begin(),
                           edge_counts.end(),
                           loop_id_edge_offsets.begin() + 1);

    result.loop_id_nodes_offsets = std::move(loop_id_nodes_offsets);
    result.loop_edge_srcs        = std::move(loop_edge_srcs);
    result.loop_edge_dsts        = std::move(loop_edge_dsts);
    result.loop_edge_orders      = std::move(edge_orders_list);
    result.loop_id_edge_offsets  = std::move(loop_id_edge_offsets);

    /*1.5 Build levelized loop ids*/
    for (size_t lvl = 0; lvl <= num_levels; ++lvl) {
      rmm::device_uvector<int32_t> loop_ids_at_level(filtered_loop_nodes.size(),
                                                     handle.get_stream());

      auto ids_end = thrust::copy_if(handle.get_thrust_policy(),
                                     coarsen_result.unique_loop_ids.begin(),
                                     coarsen_result.unique_loop_ids.end(),
                                     coarsen_result.loop_labels.begin(),
                                     loop_ids_at_level.begin(),
                                     [last_levels, lvl] __device__(int32_t loop_label) {
                                       return last_levels[loop_label] == lvl;
                                     });

      size_t num_nodes = cuda::std::distance(loop_ids_at_level.begin(), ids_end);
      loop_ids_at_level.resize(num_nodes, handle.get_stream());

      thrust::sort(handle.get_thrust_policy(), loop_ids_at_level.begin(), loop_ids_at_level.end());

      auto unique_end = thrust::unique(
        handle.get_thrust_policy(), loop_ids_at_level.begin(), loop_ids_at_level.end());

      size_t num_unique = cuda::std::distance(loop_ids_at_level.begin(), unique_end);
      loop_ids_at_level.resize(num_unique, handle.get_stream());
      loop_ids_at_level.shrink_to_fit(handle.get_stream());
      result.levelized_loop_ids.push_back(std::move(loop_ids_at_level));
    }

    /*1.6 Clear last levels for all the loop label vertices*/
    thrust::for_each(handle.get_thrust_policy(),
                     coarsen_result.loop_labels.begin(),
                     coarsen_result.loop_labels.end(),
                     [last_levels] __device__(vertex_t loop_label) {
                       last_levels[loop_label] = std::numeric_limits<size_t>::max();
                     });

    return result;
  }

  template <typename vertex_t, typename state_t>
  void update_input_node_states(
    raft::handle_t const& handle,
    cugraph::key_bucket_t<vertex_t, void, false, true> const& input_node_bucket,
    raft::device_span<uint32_t const> input_node_states,
    raft::device_span<uint32_t> node_states,
    size_t cycle,
    size_t num_patterns,
    size_t num_input_node_state_words_per_cycle_per_pattern,
    vertex_t num_vertices,
    size_t num_words_per_vertex)
  {
    thrust::for_each(handle.get_thrust_policy(),
                     thrust::make_counting_iterator(size_t{0}),
                     thrust::make_counting_iterator(num_patterns * input_node_bucket.size()),
                     [node_states,
                      input_node_bucket_first      = input_node_bucket.begin(),
                      this_cycle_input_node_states = raft::device_span<uint32_t const>(
                        input_node_states.data() +
                          cycle * num_patterns * num_input_node_state_words_per_cycle_per_pattern,
                        num_patterns * num_input_node_state_words_per_cycle_per_pattern),
                      num_vertices,
                      num_words_per_vertex,
                      num_input_nodes = input_node_bucket.size(),
                      num_input_node_state_words_per_cycle_per_pattern,
                      num_bits_per_state  = num_bits_per_state_,
                      num_states_per_word = num_states_per_word_,
                      state_mask          = state_mask_] __device__(auto i) {
                       auto p = i / num_input_nodes;
                       auto v = *(input_node_bucket_first + (i % num_input_nodes));

                       auto intra_pattern_node_idx = i % num_input_nodes;
                       auto intra_pattern_word_idx = intra_pattern_node_idx / num_states_per_word;

                       auto global_word_idx = p * num_input_node_state_words_per_cycle_per_pattern +
                                              intra_pattern_word_idx;

                       auto state = static_cast<state_t>(
                         (this_cycle_input_node_states[global_word_idx] >>
                          ((intra_pattern_node_idx % num_states_per_word) * num_bits_per_state)) &
                         state_mask);

                       auto node_states_word_idx =
                         static_cast<size_t>(v) * static_cast<size_t>(num_words_per_vertex) +
                         static_cast<size_t>(p) / static_cast<size_t>(num_states_per_word);
                       auto intra_word_idx =
                         static_cast<size_t>(p) % static_cast<size_t>(num_states_per_word);
                       auto clear_mask = ~(state_mask << (num_bits_per_state * intra_word_idx));
                       auto set_mask   = state << (num_bits_per_state * intra_word_idx);

                       cuda::atomic_ref<uint32_t, cuda::thread_scope_device> word(
                         node_states[node_states_word_idx]);
                       word.fetch_and(clear_mask, cuda::std::memory_order_relaxed);
                       word.fetch_or(set_mask, cuda::std::memory_order_relaxed);
                     });
  }

  template <typename vertex_t,
            typename edge_t,
            typename state_t,
            typename order_t,
            typename cell_idx_t,
            typename pattern_idx_t>
  void update_combinational_nodes_with_convergence(
    raft::handle_t const& handle,
    cugraph::graph_view_t<vertex_t, edge_t, true, false> const& graph_view,
    cugraph::edge_property_view_t<edge_t, order_t const*> edge_order_view,
    std::vector<cugraph::key_bucket_t<vertex_t, pattern_idx_t, false, true>> const&
      levelized_comb_node_buckets,
    std::vector<cugraph::key_bucket_t<int32_t, pattern_idx_t, false, false>> const&
      levelized_comb_node_loop_ids_buckets,
    std::vector<cugraph::key_bucket_t<vertex_t, pattern_idx_t, false, true>> const&
      levelized_latch_node_buckets,
    raft::device_span<bool const> zstate_nodes_flags,
    raft::device_span<size_t const> idx_multipliers,
    raft::device_span<cell_idx_t const> node_cell_indices,
    raft::device_span<edge_t const> cell_input_degrees,
    raft::device_span<state_t const> cell_output_tables,
    raft::device_span<size_t const> cell_output_table_offsets,
    raft::device_span<bool const> latch_flags,
    raft::device_span<size_t> old_latch_node_output_table_indices,
    raft::device_span<size_t const> latch_node_to_offset_map,
    raft::device_span<vertex_t const> filtered_loop_nodes,
    raft::device_span<size_t const> loop_id_nodes_offsets,
    raft::device_span<vertex_t const> loop_edge_srcs,
    raft::device_span<vertex_t const> loop_edge_dsts,
    raft::device_span<order_t const> loop_edge_orders,
    raft::device_span<size_t const> loop_id_edge_offsets,
    raft::device_span<uint32_t> node_states,
    size_t num_levels,
    size_t num_words_per_vertex,
    raft::device_span<vertex_t const> renumber_map)
  {
    size_t constexpr max_iterations{
      2000};  // max iterations to process combinational loops in the circuit

    for (size_t lvl = 0; lvl <= num_levels; lvl++) {
      auto const& comb_node_bucket          = levelized_comb_node_buckets[lvl];
      auto const& latch_node_bucket         = levelized_latch_node_buckets[lvl];
      auto const& comb_node_loop_ids_bucket = levelized_comb_node_loop_ids_buckets[lvl];

      rmm::device_uvector<size_t> new_comb_node_output_table_indices(comb_node_bucket.size(),
                                                                     handle.get_stream());

      cugraph::per_v_transform_reduce_incoming_e(
        handle,
        graph_view,
        comb_node_bucket,
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        edge_order_view,
        cuda::proclaim_return_type<size_t>(
          [node_states,
           zstate_nodes_flags,
           node_cell_indices,
           node_states_size = node_states.size(),
           idx_multipliers,
           num_vertices = graph_view.number_of_vertices(),
           num_words_per_vertex,
           num_bits_per_state  = num_bits_per_state_,
           num_states_per_word = num_states_per_word_,
           state_mask          = state_mask_,
           renumber_map] __device__(auto src, auto dst, auto, auto, auto order) {
            // this code is valid only in SG
            auto v = cuda::std::get<0>(dst);
            auto p = cuda::std::get<1>(dst);

            auto word_idx = static_cast<size_t>(src) * static_cast<size_t>(num_words_per_vertex) +
                            static_cast<size_t>(p) / static_cast<size_t>(num_states_per_word);
            auto intra_word_idx = static_cast<size_t>(p) % static_cast<size_t>(num_states_per_word);

            auto word = static_cast<uint32_t>(node_states[word_idx]);

            auto src_state = (word >> (intra_word_idx * num_bits_per_state)) & state_mask;

            auto output_state =
              zstate_nodes_flags[node_cell_indices[v]] * (src_state * (1 << (order * 2))) +
              (1 - zstate_nodes_flags[node_cell_indices[v]]) *
                (static_cast<size_t>((thrust::min(src_state, uint32_t{2})) *
                                     idx_multipliers[order]));

            // printf(
            //   "v: %d, renumber_map[v]: %d, src: %d, renumber_map[src]: %d, src_state: %d, "
            //   "output_state: %d\n",
            //   v,
            //   renumber_map[v],
            //   src,
            //   renumber_map[src],
            //   static_cast<int32_t>(src_state),
            //   static_cast<int32_t>(output_state));
            return output_state;
          }),
        size_t{0},
        cugraph::reduce_op::plus<size_t>{},
        new_comb_node_output_table_indices.begin());

      auto updated_pair_first = thrust::make_zip_iterator(
        comb_node_bucket.begin(), new_comb_node_output_table_indices.begin());

      thrust::for_each(
        handle.get_thrust_policy(),
        updated_pair_first,
        updated_pair_first + comb_node_bucket.size(),
        [node_states,
         node_cell_indices,
         idx_multipliers,
         cell_output_tables,
         cell_output_table_offsets,
         num_vertices = graph_view.number_of_vertices(),
         num_words_per_vertex,
         num_states_per_word = num_states_per_word_,
         num_bits_per_state  = num_bits_per_state_,
         state_mask          = state_mask_,
         renumber_map] __device__(auto pair) {
          auto tagged_v  = cuda::std::get<0>(pair);
          auto v         = cuda::std::get<0>(tagged_v);
          auto p         = cuda::std::get<1>(tagged_v);
          auto table_idx = cuda::std::get<1>(pair);
          auto cell_idx  = node_cell_indices[v];
          auto new_state = cell_output_tables[cell_output_table_offsets[cell_idx] + table_idx];
          auto word_idx  = static_cast<size_t>(v) * static_cast<size_t>(num_words_per_vertex) +
                          static_cast<size_t>(p) / static_cast<size_t>(num_states_per_word);
          auto intra_word_idx = static_cast<size_t>(p) % static_cast<size_t>(num_states_per_word);
          auto clear_mask     = ~(state_mask << (num_bits_per_state * intra_word_idx));
          auto set_mask       = new_state << (num_bits_per_state * intra_word_idx);

          cuda::atomic_ref<uint32_t, cuda::thread_scope_device> word(node_states[word_idx]);
          word.fetch_and(clear_mask, cuda::std::memory_order_relaxed);
          word.fetch_or(set_mask, cuda::std::memory_order_relaxed);
          // printf("v: %d, renumber_map[v]: %d, new_state: %d\n",
          //        v,
          //        renumber_map[v],
          //        static_cast<int32_t>(new_state));
        });

      /*Process latch nodes here*/
      if (latch_node_bucket.size() > 0) {
        update_latch_node_states_at_level<vertex_t,
                                          edge_t,
                                          state_t,
                                          order_t,
                                          cell_idx_t,
                                          pattern_idx_t>(
          handle,
          graph_view,
          edge_order_view,
          latch_node_bucket,
          idx_multipliers,
          node_cell_indices,
          cell_input_degrees,
          cell_output_tables,
          cell_output_table_offsets,
          latch_node_to_offset_map,
          node_states,
          raft::device_span<size_t>(old_latch_node_output_table_indices.data(),
                                    old_latch_node_output_table_indices.size()),
          num_words_per_vertex,
          renumber_map);
      }

      /*Process loop nodes here*/
      if (comb_node_loop_ids_bucket.size() > 0) {
        thrust::for_each(
          handle.get_thrust_policy(),
          comb_node_loop_ids_bucket.begin(),
          comb_node_loop_ids_bucket.end(),
          [node_states,
           zstate_nodes_flags,
           num_vertices = graph_view.number_of_vertices(),
           filtered_loop_nodes,
           loop_id_nodes_offsets,
           loop_edge_srcs,
           loop_edge_dsts,
           loop_edge_orders,
           loop_id_edge_offsets,
           node_cell_indices,
           cell_input_degrees,
           idx_multipliers,
           cell_output_tables,
           cell_output_table_offsets,
           latch_flags,
           old_latch_node_output_table_indices,
           latch_node_to_offset_map,
           max_iterations,
           num_words_per_vertex,
           num_states_per_word = num_states_per_word_,
           num_bits_per_state  = num_bits_per_state_,
           state_mask          = state_mask_] __device__(auto tagged_id) {
            auto loopId = cuda::std::get<0>(tagged_id);
            auto p      = cuda::std::get<1>(tagged_id);

            auto loop_nodes_length =
              loop_id_nodes_offsets[loopId + 1] - loop_id_nodes_offsets[loopId];
            auto loop_nodes_start = loop_id_nodes_offsets[loopId];
            auto loop_nodes_end   = loop_nodes_start + loop_nodes_length;

            auto loop_edges_length =
              loop_id_edge_offsets[loopId + 1] - loop_id_edge_offsets[loopId];
            auto loop_edges_start = loop_id_edge_offsets[loopId];
            auto loop_edges_end   = loop_edges_start + loop_edges_length;

            size_t iter = 0;

            while (iter < max_iterations) {
              bool state_changed = false;

              for (auto i = loop_nodes_start; i < loop_nodes_end; i++) {
                auto dst = filtered_loop_nodes[i];
                auto word_idx =
                  static_cast<size_t>(dst) * static_cast<size_t>(num_words_per_vertex) +
                  static_cast<size_t>(p) / static_cast<size_t>(num_states_per_word);
                auto intra_word_idx =
                  static_cast<size_t>(p) % static_cast<size_t>(num_states_per_word);
                auto dst_state =
                  (node_states[word_idx] >> (intra_word_idx * num_bits_per_state)) & state_mask;
                auto cell_idx = node_cell_indices[dst];

                auto edge_start_dist_for_dst = cuda::std::distance(
                  loop_edge_dsts.begin() + loop_edges_start,
                  thrust::lower_bound(thrust::seq,
                                      loop_edge_dsts.begin() + loop_edges_start,
                                      loop_edge_dsts.begin() + loop_edges_start + loop_edges_length,
                                      dst));
                auto edge_end_dist_for_dst = cuda::std::distance(
                  loop_edge_dsts.begin() + loop_edges_start,
                  thrust::upper_bound(thrust::seq,
                                      loop_edge_dsts.begin() + loop_edges_start,
                                      loop_edge_dsts.begin() + loop_edges_start + loop_edges_length,
                                      dst));

                // Accumulate input contributions from edges
                size_t new_idx = 0;
                for (auto j = edge_start_dist_for_dst; j < edge_end_dist_for_dst; j++) {
                  auto src         = loop_edge_srcs[loop_edges_start + j];
                  auto zstate_flag = zstate_nodes_flags[node_cell_indices[dst]];
                  auto word_idx =
                    static_cast<size_t>(src) * static_cast<size_t>(num_words_per_vertex) +
                    static_cast<size_t>(p) / static_cast<size_t>(num_states_per_word);
                  auto intra_word_idx =
                    static_cast<size_t>(p) % static_cast<size_t>(num_states_per_word);
                  auto src_state =
                    (node_states[word_idx] >> (intra_word_idx * num_bits_per_state)) & state_mask;
                  auto order = loop_edge_orders[loop_edges_start + j];
                  new_idx += static_cast<size_t>(
                    zstate_flag * (src_state * (1 << (order * 2))) +
                    (1 - zstate_flag) * (static_cast<size_t>((thrust::min(src_state, uint32_t{2})) *
                                                             idx_multipliers[order])));
                }

                // For latch nodes: dual-index lookup (old_state + new_input)
                // For combinational nodes: single-index lookup (new_input only)
                size_t table_idx = new_idx;
                if (latch_flags[cell_idx]) {
                  auto old_idx =
                    old_latch_node_output_table_indices[latch_node_to_offset_map[dst] + p];
                  auto order = cell_input_degrees[cell_idx] + 1;
                  table_idx  = old_idx + new_idx * idx_multipliers[order];
                }

                auto new_state =
                  cell_output_tables[cell_output_table_offsets[cell_idx] + table_idx];
                if (new_state != dst_state) {
                  state_changed   = true;
                  auto clear_mask = ~(state_mask << (num_bits_per_state * intra_word_idx));
                  auto set_mask   = new_state << (num_bits_per_state * intra_word_idx);
                  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> word(node_states[word_idx]);
                  word.fetch_and(clear_mask, cuda::std::memory_order_relaxed);
                  word.fetch_or(set_mask, cuda::std::memory_order_relaxed);
                }

                // Update old_latch_node_output_table_indices each time a latch is evaluated
                if (latch_flags[cell_idx]) {
                  old_latch_node_output_table_indices[latch_node_to_offset_map[dst] + p] =
                    static_cast<size_t>(new_state & state_mask) + new_idx * idx_multipliers[1];
                }
              }

              if (!state_changed) { break; }

              iter++;
              if (iter >= max_iterations) {
                printf("Max iterations reached for loop %d pattern %d\n", loopId, p);
                break;
              }
            }
          });
      }
    }
  }

  template <typename vertex_t,
            typename edge_t,
            typename state_t,
            typename order_t,
            typename cell_idx_t,
            typename pattern_idx_t>
  void update_sequential_node_states(
    raft::handle_t const& handle,
    cugraph::graph_view_t<vertex_t, edge_t, true, false> const& graph_view,
    cugraph::edge_property_view_t<edge_t, order_t const*> edge_order_view,
    cugraph::key_bucket_t<vertex_t, pattern_idx_t, false, false>& seq_node_bucket,
    raft::device_span<size_t const> idx_multipliers,
    raft::device_span<cell_idx_t const> node_cell_indices,
    raft::device_span<edge_t const> cell_input_degrees,
    raft::device_span<state_t const> cell_output_tables,
    raft::device_span<size_t const> cell_output_table_offsets,
    raft::device_span<size_t> old_seq_node_output_table_indices,
    raft::device_span<uint32_t> node_states,
    size_t num_words_per_vertex,
    raft::device_span<vertex_t const> renumber_map)
  {
    rmm::device_uvector<size_t> cur_seq_node_output_table_indices(seq_node_bucket.size(),
                                                                  handle.get_stream());
    {
      static_assert(!multi_gpu_);  // assumes that it is possible to directly access
                                   // node_states for every node
      cugraph::per_v_transform_reduce_incoming_e(
        handle,
        graph_view,
        seq_node_bucket,
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        edge_order_view,
        cuda::proclaim_return_type<size_t>(
          [node_states,
           idx_multipliers,
           num_vertices = graph_view.number_of_vertices(),
           num_words_per_vertex,
           num_bits_per_state  = num_bits_per_state_,
           num_states_per_word = num_states_per_word_,
           state_mask          = state_mask_,
           renumber_map] __device__(auto src, auto dst, auto, auto, auto order) {
            // this code is valid only in SG
            auto v = cuda::std::get<0>(dst);
            auto p = cuda::std::get<1>(dst);

            auto word_idx = static_cast<size_t>(src) * static_cast<size_t>(num_words_per_vertex) +
                            static_cast<size_t>(p) / static_cast<size_t>(num_states_per_word);
            auto intra_word_idx = static_cast<size_t>(p) % static_cast<size_t>(num_states_per_word);
            auto src_state =
              (node_states[word_idx] >> (intra_word_idx * num_bits_per_state)) & state_mask;

            auto output_state =
              static_cast<size_t>(thrust::min(src_state, uint32_t{2}) * idx_multipliers[order]);

            // printf(
            //   "v: %d, renumber_map[v]: %d, src: %d, renumber_map[src]: %d, src_state: %d, "
            //   "output_state: %d\n",
            //   v,
            //   renumber_map[v],
            //   src,
            //   renumber_map[src],
            //   static_cast<int32_t>(src_state),
            //   static_cast<int32_t>(output_state));
            return output_state;
          }),
        size_t{0},
        cugraph::reduce_op::plus<size_t>{},
        cur_seq_node_output_table_indices.begin());
    }

    auto triplet_first = thrust::make_zip_iterator(seq_node_bucket.begin(),
                                                   old_seq_node_output_table_indices.begin(),
                                                   cur_seq_node_output_table_indices.begin());

    thrust::for_each(
      handle.get_thrust_policy(),
      triplet_first,
      triplet_first + seq_node_bucket.size(),
      [node_cell_indices,
       cell_input_degrees,
       idx_multipliers,
       cell_output_tables,
       cell_output_table_offsets,
       node_states,
       num_vertices = graph_view.number_of_vertices(),
       num_words_per_vertex,
       num_states_per_word = num_states_per_word_,
       num_bits_per_state  = num_bits_per_state_,
       state_mask          = state_mask_,
       renumber_map] __device__(auto triplet) {
        auto tagged_v = cuda::std::get<0>(triplet);
        auto v        = cuda::std::get<0>(tagged_v);
        auto p        = cuda::std::get<1>(tagged_v);
        auto old_idx  = cuda::std::get<1>(triplet);
        auto new_idx  = cuda::std::get<2>(triplet);
        auto cell_idx = node_cell_indices[v];

        auto order     = cell_input_degrees[cell_idx] + 1 /* output */;
        auto table_idx = old_idx + new_idx * idx_multipliers[order];
        auto new_state = cell_output_tables[cell_output_table_offsets[cell_idx] + table_idx];
        auto word_idx  = static_cast<size_t>(v) * static_cast<size_t>(num_words_per_vertex) +
                        static_cast<size_t>(p) / static_cast<size_t>(num_states_per_word);
        auto intra_word_idx = static_cast<size_t>(p) % static_cast<size_t>(num_states_per_word);
        auto clear_mask     = ~(state_mask << (num_bits_per_state * intra_word_idx));
        auto set_mask       = new_state << (num_bits_per_state * intra_word_idx);

        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> word(node_states[word_idx]);
        word.fetch_and(clear_mask, cuda::std::memory_order_relaxed);
        word.fetch_or(set_mask, cuda::std::memory_order_relaxed);
        // printf("v: %d, renumber_map[v]: %d, new_state: %d\n",
        //        v,
        //        renumber_map[v],
        //        static_cast<int32_t>(new_state));
      });

    // old_seq_node_output_table_indices = std::move(raft::device_span<size_t>(
    //   cur_seq_node_output_table_indices.data(), cur_seq_node_output_table_indices.size()));
    auto pair_first =
      thrust::make_zip_iterator(seq_node_bucket.begin(), cur_seq_node_output_table_indices.begin());

    thrust::transform(
      handle.get_thrust_policy(),
      pair_first,
      pair_first + seq_node_bucket.size(),
      old_seq_node_output_table_indices.begin(),
      cuda::proclaim_return_type<size_t>([node_states,
                                          node_cell_indices,
                                          idx_multipliers = raft::device_span<size_t const>(
                                            idx_multipliers.data(), idx_multipliers.size()),
                                          num_vertices = graph_view.number_of_vertices(),
                                          num_words_per_vertex,
                                          num_states_per_word = num_states_per_word_,
                                          num_bits_per_state  = num_bits_per_state_,
                                          state_mask          = state_mask_] __device__(auto pair) {
        auto tagged_v  = cuda::std::get<0>(pair);
        auto v         = cuda::std::get<0>(tagged_v);
        auto p         = cuda::std::get<1>(tagged_v);
        auto table_idx = cuda::std::get<1>(pair);
        auto word_idx  = static_cast<size_t>(v) * static_cast<size_t>(num_words_per_vertex) +
                        static_cast<size_t>(p) / static_cast<size_t>(num_states_per_word);
        auto intra_word_idx = static_cast<size_t>(p) % static_cast<size_t>(num_states_per_word);
        auto output_state =
          (node_states[word_idx] >> (intra_word_idx * num_bits_per_state)) & state_mask;
        auto cell_idx = node_cell_indices[v];
        return static_cast<size_t>(output_state) + table_idx * idx_multipliers[1];
      }));
  }

  // Updates latch node states at a single levelization level using dual-index table lookup.
  // Uses latch_node_to_offset_map for random access into the global
  // old_latch_node_output_table_indices array, which covers ALL latch nodes (both loop and
  // non-loop).
  template <typename vertex_t,
            typename edge_t,
            typename state_t,
            typename order_t,
            typename cell_idx_t,
            typename pattern_idx_t>
  void update_latch_node_states_at_level(
    raft::handle_t const& handle,
    cugraph::graph_view_t<vertex_t, edge_t, true, false> const& graph_view,
    cugraph::edge_property_view_t<edge_t, order_t const*> edge_order_view,
    cugraph::key_bucket_t<vertex_t, pattern_idx_t, false, sorted_unique_key_bucket_> const&
      latch_node_bucket,
    raft::device_span<size_t const> idx_multipliers,
    raft::device_span<cell_idx_t const> node_cell_indices,
    raft::device_span<edge_t const> cell_input_degrees,
    raft::device_span<state_t const> cell_output_tables,
    raft::device_span<size_t const> cell_output_table_offsets,
    raft::device_span<size_t const> latch_node_to_offset_map,
    raft::device_span<uint32_t> node_states,
    raft::device_span<size_t> old_latch_node_output_table_indices,
    size_t num_words_per_vertex,
    raft::device_span<vertex_t const> renumber_map)
  {
    // Step 1: Compute new input table indices from current edge states
    rmm::device_uvector<size_t> cur_latch_node_output_table_indices(latch_node_bucket.size(),
                                                                    handle.get_stream());
    {
      static_assert(!multi_gpu_);
      cugraph::per_v_transform_reduce_incoming_e(
        handle,
        graph_view,
        latch_node_bucket,
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        edge_order_view,
        cuda::proclaim_return_type<size_t>(
          [node_states,
           idx_multipliers,
           num_vertices = graph_view.number_of_vertices(),
           num_words_per_vertex,
           num_bits_per_state  = num_bits_per_state_,
           num_states_per_word = num_states_per_word_,
           state_mask          = state_mask_,
           renumber_map] __device__(auto src, auto dst, auto, auto, auto order) {
            auto v = cuda::std::get<0>(dst);
            auto p = cuda::std::get<1>(dst);

            auto word_idx = static_cast<size_t>(src) * static_cast<size_t>(num_words_per_vertex) +
                            static_cast<size_t>(p) / static_cast<size_t>(num_states_per_word);
            auto intra_word_idx = static_cast<size_t>(p) % static_cast<size_t>(num_states_per_word);
            auto src_state =
              (node_states[word_idx] >> (intra_word_idx * num_bits_per_state)) & state_mask;

            auto output_state =
              static_cast<size_t>(thrust::min(src_state, uint32_t{2}) * idx_multipliers[order]);
            // printf(
            //   "v: %d, renumber_map[v]: %d, src: %d, renumber_map[src]: %d, order: %d, "
            //   "idx_multipliers[order]: %d, src_state: %d, output_state: %d\n",
            //   v,
            //   renumber_map[v],
            //   src,
            //   renumber_map[src],
            //   static_cast<int32_t>(order),
            //   static_cast<int32_t>(idx_multipliers[order]),
            //   static_cast<int32_t>(src_state),
            //   static_cast<int32_t>(output_state));
            return output_state;
          }),
        size_t{0},
        cugraph::reduce_op::plus<size_t>{},
        cur_latch_node_output_table_indices.begin());
    }

    // Step 2: Combine old_idx (via map) + new_idx, look up table, write node_states
    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(latch_node_bucket.size()),
      [bucket_begin = latch_node_bucket.begin(),
       cur_indices  = cur_latch_node_output_table_indices.data(),
       old_latch_node_output_table_indices,
       latch_node_to_offset_map,
       node_cell_indices,
       cell_input_degrees,
       idx_multipliers,
       cell_output_tables,
       cell_output_table_offsets,
       node_states,
       num_words_per_vertex,
       num_states_per_word = num_states_per_word_,
       num_bits_per_state  = num_bits_per_state_,
       state_mask          = state_mask_,
       renumber_map] __device__(auto i) {
        auto tagged_v = *(bucket_begin + i);
        auto v        = cuda::std::get<0>(tagged_v);
        auto p        = cuda::std::get<1>(tagged_v);
        auto new_idx  = cur_indices[i];
        auto old_idx  = old_latch_node_output_table_indices[latch_node_to_offset_map[v] + p];
        auto cell_idx = node_cell_indices[v];

        auto order     = cell_input_degrees[cell_idx] + 1 /* output */;
        auto table_idx = old_idx + new_idx * idx_multipliers[order];
        auto new_state = cell_output_tables[cell_output_table_offsets[cell_idx] + table_idx];
        auto word_idx  = static_cast<size_t>(v) * static_cast<size_t>(num_words_per_vertex) +
                        static_cast<size_t>(p) / static_cast<size_t>(num_states_per_word);
        auto intra_word_idx = static_cast<size_t>(p) % static_cast<size_t>(num_states_per_word);
        auto clear_mask     = ~(state_mask << (num_bits_per_state * intra_word_idx));
        auto set_mask       = new_state << (num_bits_per_state * intra_word_idx);

        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> word(node_states[word_idx]);
        word.fetch_and(clear_mask, cuda::std::memory_order_relaxed);
        word.fetch_or(set_mask, cuda::std::memory_order_relaxed);
        // printf("v: %d, renumber_map[v]: %d, old_idx: %d, new_idx: %d, new_state: %d\n",
        //        v,
        //        renumber_map[v],
        //        static_cast<int32_t>(old_idx),
        //        static_cast<int32_t>(new_idx),
        //        static_cast<int32_t>(new_state));
      });

    // Step 3: Update old_latch_node_output_table_indices for next use
    // Read back the newly written output state and combine with new_idx to form the updated old_idx
    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(latch_node_bucket.size()),
      [bucket_begin = latch_node_bucket.begin(),
       cur_indices  = cur_latch_node_output_table_indices.data(),
       old_latch_node_output_table_indices,
       latch_node_to_offset_map,
       node_states,
       node_cell_indices,
       idx_multipliers,
       num_words_per_vertex,
       num_states_per_word = num_states_per_word_,
       num_bits_per_state  = num_bits_per_state_,
       state_mask          = state_mask_] __device__(auto i) {
        auto tagged_v = *(bucket_begin + i);
        auto v        = cuda::std::get<0>(tagged_v);
        auto p        = cuda::std::get<1>(tagged_v);
        auto new_idx  = cur_indices[i];
        auto word_idx = static_cast<size_t>(v) * static_cast<size_t>(num_words_per_vertex) +
                        static_cast<size_t>(p) / static_cast<size_t>(num_states_per_word);
        auto intra_word_idx = static_cast<size_t>(p) % static_cast<size_t>(num_states_per_word);
        auto output_state =
          (node_states[word_idx] >> (intra_word_idx * num_bits_per_state)) & state_mask;
        old_latch_node_output_table_indices[latch_node_to_offset_map[v] + p] =
          static_cast<size_t>(output_state) + new_idx * idx_multipliers[1];
      });
  }

  template <typename vertex_t,
            typename edge_t,
            typename state_t,
            typename order_t,
            typename cell_idx_t,
            typename pattern_idx_t>
  void run_simulation_cycles(
    raft::handle_t& handle,
    HighResTimer& hr_timer,
    cugraph::graph_view_t<vertex_t, edge_t, true, false> const& graph_view,
    cugraph::edge_property_view_t<edge_t, order_t const*> edge_order_view,
    SimulationInputData<vertex_t>& sim_input_data,
    NodeBuckets<vertex_t, pattern_idx_t>& node_buckets,
    LevelizedNodeBuckets<vertex_t, pattern_idx_t>& levelized_buckets,
    node_cell_map_result<vertex_t, edge_t, cell_idx_t> const& node_cell_map,
    cell_output_table_info<state_t> const& cell_output_tables_info,
    LoopDataResult<vertex_t, order_t> const& loop_data,
    CellTypeBooleanFlags const& cell_flags,
    raft::device_span<size_t const> idx_multipliers,
    raft::device_span<vertex_t const> renumber_map,
    raft::device_span<vertex_t const> filtered_loop_nodes,
    raft::device_span<uint32_t> node_states,
    size_t num_patterns,
    size_t num_levels)
  {
    auto& input_node_bucket  = node_buckets.input_node_bucket;
    auto& output_node_bucket = node_buckets.output_node_bucket;
    auto& ground_node_bucket = node_buckets.ground_node_bucket;
    auto& pwr_node_bucket    = node_buckets.pwr_node_bucket;
    auto& tie_x_node_bucket  = node_buckets.tie_x_node_bucket;
    auto& z_node_bucket      = node_buckets.z_node_bucket;
    auto& seq_node_bucket    = node_buckets.seq_node_bucket;
    auto output_node_inputs  = raft::device_span<vertex_t const>(
      node_buckets.output_node_inputs.data(), node_buckets.output_node_inputs.size());
    auto num_input_nodes       = node_buckets.num_input_nodes;
    auto num_output_nodes      = node_buckets.num_output_nodes;
    auto num_output_nodes_used = node_buckets.num_output_nodes_used;

    auto& levelized_comb_node_buckets = levelized_buckets.levelized_comb_node_buckets;
    auto& levelized_comb_node_loop_ids_buckets =
      levelized_buckets.levelized_comb_node_loop_ids_buckets;
    auto& levelized_latch_node_buckets = levelized_buckets.levelized_latch_node_buckets;
    auto latch_node_to_offset_map =
      raft::device_span<size_t const>(levelized_buckets.latch_node_to_offset_map.data(),
                                      levelized_buckets.latch_node_to_offset_map.size());

    auto node_cell_indices = raft::device_span<cell_idx_t const>(
      node_cell_map.node_cell_indices.data(), node_cell_map.node_cell_indices.size());
    auto cell_input_degrees = raft::device_span<edge_t const>(
      node_cell_map.cell_input_degrees.data(), node_cell_map.cell_input_degrees.size());

    auto cell_output_tables = raft::device_span<state_t const>(
      cell_output_tables_info.tables.data(), cell_output_tables_info.tables.size());
    auto cell_output_table_offsets = raft::device_span<size_t const>(
      cell_output_tables_info.offsets.data(), cell_output_tables_info.offsets.size());

    auto loop_id_nodes_offsets = raft::device_span<size_t const>(
      loop_data.loop_id_nodes_offsets.data(), loop_data.loop_id_nodes_offsets.size());
    auto loop_edge_srcs       = raft::device_span<vertex_t const>(loop_data.loop_edge_srcs.data(),
                                                            loop_data.loop_edge_srcs.size());
    auto loop_edge_dsts       = raft::device_span<vertex_t const>(loop_data.loop_edge_dsts.data(),
                                                            loop_data.loop_edge_dsts.size());
    auto loop_edge_orders     = raft::device_span<order_t const>(loop_data.loop_edge_orders.data(),
                                                             loop_data.loop_edge_orders.size());
    auto loop_id_edge_offsets = raft::device_span<size_t const>(
      loop_data.loop_id_edge_offsets.data(), loop_data.loop_id_edge_offsets.size());
    auto zstate_nodes_flags = raft::device_span<bool const>(cell_flags.zstate_nodes_flags.data(),
                                                            cell_flags.zstate_nodes_flags.size());
    auto latch_flags =
      raft::device_span<bool const>(cell_flags.latch_flags.data(), cell_flags.latch_flags.size());

    auto num_input_node_state_words_per_cycle_per_pattern =
      (num_input_nodes + num_states_per_word_ - 1) / num_states_per_word_;
    auto num_output_node_state_words_per_cycle_per_pattern =
      (num_output_nodes_used + num_states_per_word_ - 1) / num_states_per_word_;

    rmm::device_uvector<uint32_t>& input_node_states = sim_input_data.input_node_states;
    size_t num_cycles                                = sim_input_data.num_cycles;
    auto compare_outputs = sim_input_data.expected_output_node_states.has_value();

    /* Simulation using clock cycles */
    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Simulation using clock cycles");
    }

    auto num_words_per_vertex = (num_patterns + num_states_per_word_ - 1) / num_states_per_word_;

    /* Initialize ground nodes bucket to state_t{0}*/
    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(num_patterns * ground_node_bucket.size()),
      [node_states,
       ground_node_bucket_first = ground_node_bucket.begin(),
       num_ground_nodes         = ground_node_bucket.size(),
       num_vertices             = graph_view.number_of_vertices(),
       num_words_per_vertex,
       num_states_per_word = num_states_per_word_,
       num_bits_per_state  = num_bits_per_state_,
       state_mask          = state_mask_] __device__(auto i) {
        auto p = i / num_ground_nodes;
        auto v = *(ground_node_bucket_first + (i % num_ground_nodes));

        auto word_idx = static_cast<size_t>(v) * static_cast<size_t>(num_words_per_vertex) +
                        static_cast<size_t>(p) / static_cast<size_t>(num_states_per_word);
        auto intra_pattern_node_idx =
          static_cast<size_t>(p) % static_cast<size_t>(num_states_per_word);
        auto clear_mask = ~(state_mask << (num_bits_per_state * intra_pattern_node_idx));
        auto set_mask   = state_t{0} << (num_bits_per_state * intra_pattern_node_idx);

        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> word(node_states[word_idx]);
        word.fetch_and(clear_mask, cuda::std::memory_order_relaxed);
        word.fetch_or(set_mask, cuda::std::memory_order_relaxed);
      });

    /* Initialize power nodes bucket to state_t{1}*/
    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(num_patterns * pwr_node_bucket.size()),
      [node_states,
       pwr_node_bucket_first = pwr_node_bucket.begin(),
       num_pwr_nodes         = pwr_node_bucket.size(),
       num_vertices          = graph_view.number_of_vertices(),
       num_words_per_vertex,
       num_states_per_word = num_states_per_word_,
       num_bits_per_state  = num_bits_per_state_,
       state_mask          = state_mask_] __device__(auto i) {
        auto p = i / num_pwr_nodes;
        auto v = *(pwr_node_bucket_first + (i % num_pwr_nodes));

        auto word_idx = static_cast<size_t>(v) * static_cast<size_t>(num_words_per_vertex) +
                        static_cast<size_t>(p) / static_cast<size_t>(num_states_per_word);
        auto intra_pattern_node_idx =
          static_cast<size_t>(p) % static_cast<size_t>(num_states_per_word);
        auto clear_mask = ~(state_mask << (num_bits_per_state * intra_pattern_node_idx));
        auto set_mask   = state_t{1} << (num_bits_per_state * intra_pattern_node_idx);

        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> word(node_states[word_idx]);
        word.fetch_and(clear_mask, cuda::std::memory_order_relaxed);
        word.fetch_or(set_mask, cuda::std::memory_order_relaxed);
      });

    /* Initialize tie-x nodes bucket to state_t{2}*/
    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(num_patterns * tie_x_node_bucket.size()),
      [node_states,
       tie_x_node_bucket_first = tie_x_node_bucket.begin(),
       num_tie_x_nodes         = tie_x_node_bucket.size(),
       num_vertices            = graph_view.number_of_vertices(),
       num_words_per_vertex,
       num_states_per_word = num_states_per_word_,
       num_bits_per_state  = num_bits_per_state_,
       state_mask          = state_mask_] __device__(auto i) {
        auto p = i / num_tie_x_nodes;
        auto v = *(tie_x_node_bucket_first + (i % num_tie_x_nodes));

        auto word_idx = static_cast<size_t>(v) * static_cast<size_t>(num_words_per_vertex) +
                        static_cast<size_t>(p) / static_cast<size_t>(num_states_per_word);
        auto intra_pattern_node_idx =
          static_cast<size_t>(p) % static_cast<size_t>(num_states_per_word);
        auto clear_mask = ~(state_mask << (num_bits_per_state * intra_pattern_node_idx));
        auto set_mask   = state_t{2} << (num_bits_per_state * intra_pattern_node_idx);

        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> word(node_states[word_idx]);
        word.fetch_and(clear_mask, cuda::std::memory_order_relaxed);
        word.fetch_or(set_mask, cuda::std::memory_order_relaxed);
      });

    /* Initialize z nodes bucket to state_t{3}*/
    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(num_patterns * z_node_bucket.size()),
      [node_states,
       z_node_bucket_first = z_node_bucket.begin(),
       num_z_nodes         = z_node_bucket.size(),
       num_vertices        = graph_view.number_of_vertices(),
       num_words_per_vertex,
       num_states_per_word = num_states_per_word_,
       num_bits_per_state  = num_bits_per_state_,
       state_mask          = state_mask_] __device__(auto i) {
        auto p = i / num_z_nodes;
        auto v = *(z_node_bucket_first + (i % num_z_nodes));

        auto word_idx = static_cast<size_t>(v) * static_cast<size_t>(num_words_per_vertex) +
                        static_cast<size_t>(p) / static_cast<size_t>(num_states_per_word);
        auto intra_pattern_node_idx =
          static_cast<size_t>(p) % static_cast<size_t>(num_states_per_word);
        auto clear_mask = ~(state_mask << (num_bits_per_state * intra_pattern_node_idx));
        auto set_mask   = state_t{3} << (num_bits_per_state * intra_pattern_node_idx);

        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> word(node_states[word_idx]);
        word.fetch_and(clear_mask, cuda::std::memory_order_relaxed);
        word.fetch_or(set_mask, cuda::std::memory_order_relaxed);
      });

    auto old_seq_node_output_table_indices =
      std::move(sim_input_data.old_seq_node_output_table_indices);
    auto old_latch_node_output_table_indices =
      std::move(sim_input_data.old_latch_node_output_table_indices);

    // auto gpu_id = std::getenv("CUDA_VISIBLE_DEVICES");
    for (size_t cycle = 0; cycle < num_cycles; ++cycle) {
#if 1  // DEBUG
      std::cout << "cycle: " << cycle << std::endl;
      // RAFT_CUDA_TRY(cudaDeviceSynchronize());
      // auto time0 = std::chrono::steady_clock::now();
#endif

      /* load the pseudo-input node states */
      update_input_node_states<vertex_t, state_t>(
        handle,
        input_node_bucket,
        raft::device_span<uint32_t const>(input_node_states.data(), input_node_states.size()),
        node_states,
        cycle,
        num_patterns,
        num_input_node_state_words_per_cycle_per_pattern,
        graph_view.number_of_vertices(),
        num_words_per_vertex);

      // size_t iter{0};
      update_combinational_nodes_with_convergence<vertex_t,
                                                  edge_t,
                                                  state_t,
                                                  order_t,
                                                  cell_idx_t,
                                                  pattern_idx_t>(
        handle,
        graph_view,
        edge_order_view,
        levelized_comb_node_buckets,
        levelized_comb_node_loop_ids_buckets,
        levelized_latch_node_buckets,
        zstate_nodes_flags,
        idx_multipliers,
        node_cell_indices,
        cell_input_degrees,
        cell_output_tables,
        cell_output_table_offsets,
        latch_flags,
        raft::device_span<size_t>(old_latch_node_output_table_indices.data(),
                                  old_latch_node_output_table_indices.size()),
        latch_node_to_offset_map,
        filtered_loop_nodes,
        loop_id_nodes_offsets,
        loop_edge_srcs,
        loop_edge_dsts,
        loop_edge_orders,
        loop_id_edge_offsets,
        node_states,
        num_levels,
        num_words_per_vertex,
        renumber_map);

      /* update sequential node states */
      update_sequential_node_states<vertex_t, edge_t, state_t, order_t, cell_idx_t, pattern_idx_t>(
        handle,
        graph_view,
        edge_order_view,
        seq_node_bucket,
        idx_multipliers,
        node_cell_indices,
        cell_input_degrees,
        cell_output_tables,
        cell_output_table_offsets,
        raft::device_span<size_t>(old_seq_node_output_table_indices.data(),
                                  old_seq_node_output_table_indices.size()),
        node_states,
        num_words_per_vertex,
        renumber_map);

      update_combinational_nodes_with_convergence<vertex_t,
                                                  edge_t,
                                                  state_t,
                                                  order_t,
                                                  cell_idx_t,
                                                  pattern_idx_t>(
        handle,
        graph_view,
        edge_order_view,
        levelized_comb_node_buckets,
        levelized_comb_node_loop_ids_buckets,
        levelized_latch_node_buckets,
        zstate_nodes_flags,
        idx_multipliers,
        node_cell_indices,
        cell_input_degrees,
        cell_output_tables,
        cell_output_table_offsets,
        latch_flags,
        raft::device_span<size_t>(old_latch_node_output_table_indices.data(),
                                  old_latch_node_output_table_indices.size()),
        latch_node_to_offset_map,
        filtered_loop_nodes,
        loop_id_nodes_offsets,
        loop_edge_srcs,
        loop_edge_dsts,
        loop_edge_orders,
        loop_id_edge_offsets,
        node_states,
        num_levels,
        num_words_per_vertex,
        renumber_map);

      if (compare_outputs) {
        auto& expected_output_node_states = *sim_input_data.expected_output_node_states;
        /* Compare output node states with expected output node states */
        thrust::for_each(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator(size_t{0}),
          thrust::make_counting_iterator(num_patterns * output_node_bucket.size()),
          [node_states = raft::device_span<uint32_t>(node_states.data(), node_states.size()),
           output_node_bucket_first = output_node_bucket.begin(),
           output_node_inputs       = raft::device_span<vertex_t const>(output_node_inputs.data(),
                                                                  output_node_inputs.size()),
           num_output_nodes         = output_node_bucket.size(),
           num_vertices             = graph_view.number_of_vertices(),
           renumber_map =
             raft::device_span<vertex_t const>(renumber_map.data(), renumber_map.size()),
           num_words_per_vertex,
           num_states_per_word = num_states_per_word_,
           num_bits_per_state  = num_bits_per_state_,
           state_mask          = state_mask_] __device__(auto i) {
            auto p     = i / num_output_nodes;
            auto v_out = *(output_node_bucket_first + (i % num_output_nodes));
            auto v_in  = output_node_inputs[i % num_output_nodes];

            auto word_idx_in =
              static_cast<size_t>(v_in) * static_cast<size_t>(num_words_per_vertex) +
              static_cast<size_t>(p) / static_cast<size_t>(num_states_per_word);
            auto intra_word_idx_in =
              static_cast<size_t>(p) % static_cast<size_t>(num_states_per_word);
            auto in_state =
              (node_states[word_idx_in] >> (intra_word_idx_in * num_bits_per_state)) & state_mask;

            auto word_idx_out =
              static_cast<size_t>(v_out) * static_cast<size_t>(num_words_per_vertex) +
              static_cast<size_t>(p) / static_cast<size_t>(num_states_per_word);
            auto intra_word_idx_out =
              static_cast<size_t>(p) % static_cast<size_t>(num_states_per_word);
            auto clear_mask = ~(state_mask << (num_bits_per_state * intra_word_idx_out));
            auto set_mask   = in_state << (num_bits_per_state * intra_word_idx_out);

            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> word(node_states[word_idx_out]);
            word.fetch_and(clear_mask, cuda::std::memory_order_relaxed);
            word.fetch_or(set_mask, cuda::std::memory_order_relaxed);

            auto out_state_dbg =
              (node_states[word_idx_out] >> (intra_word_idx_out * num_bits_per_state)) & state_mask;
            auto in_state_dbg =
              (node_states[word_idx_in] >> (intra_word_idx_in * num_bits_per_state)) & state_mask;
            // printf(
            //   "v_out: %d, renumber_map[v_out]: %d, v_in: %d, renumber_map[v_in]: %d, "
            //   "out_state_dbg: %d, in_state_dbg: %d\n",
            //   v_out,
            //   renumber_map[v_out],
            //   v_in,
            //   renumber_map[v_in],
            //   static_cast<int32_t>(out_state_dbg),
            //   static_cast<int32_t>(in_state_dbg));
          });

        auto num_invalids = thrust::count_if(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator(size_t{0}),
          thrust::make_counting_iterator(num_patterns *
                                         num_output_node_state_words_per_cycle_per_pattern),
          cuda::proclaim_return_type<bool>(
            [node_states =
               raft::device_span<uint32_t const>(node_states.data(), node_states.size()),
             expected_this_cycle_output_node_states = raft::device_span<uint32_t const>(
               expected_output_node_states.data() +
                 cycle * num_patterns * num_output_node_state_words_per_cycle_per_pattern,
               num_patterns * num_output_node_state_words_per_cycle_per_pattern),
             output_node_first = output_node_bucket.begin(),
             num_output_nodes  = output_node_bucket.size(),
             num_patterns,
             num_output_node_state_words_per_cycle_per_pattern,
             num_words_per_vertex,
             num_states_per_word = num_states_per_word_,
             num_bits_per_state  = num_bits_per_state_,
             state_mask          = state_mask_] __device__(size_t i) {
              auto expected               = expected_this_cycle_output_node_states[i];
              auto pattern_idx            = i / num_output_node_state_words_per_cycle_per_pattern;
              auto intra_pattern_word_idx = i % num_output_node_state_words_per_cycle_per_pattern;
              uint32_t computed{0};
              for (size_t j = intra_pattern_word_idx * num_states_per_word;
                   j < cuda::std::min((intra_pattern_word_idx + 1) * num_states_per_word,
                                      num_output_nodes);
                   ++j) {
                auto v = *(output_node_first + j);
                auto word_idx =
                  static_cast<size_t>(v) * static_cast<size_t>(num_words_per_vertex) +
                  static_cast<size_t>(pattern_idx) / static_cast<size_t>(num_states_per_word);
                auto intra_word_idx =
                  static_cast<size_t>(pattern_idx) % static_cast<size_t>(num_states_per_word);
                auto state =
                  (node_states[word_idx] >> (intra_word_idx * num_bits_per_state)) & state_mask;
                computed |= state << ((j - intra_pattern_word_idx * num_states_per_word) *
                                      num_bits_per_state);
              }
              if (expected != computed) {
                printf("intra_pattern_word_idx: %d, expected: 0x%x, computed: 0x%x\n",
                       static_cast<int>(intra_pattern_word_idx),
                       expected,
                       computed);
              }
              return expected != computed;
            }));

        RAFT_CUDA_TRY(cudaDeviceSynchronize());
        std::cout << "num_invalids: " << num_invalids << std::endl;

        CUGRAPH_EXPECTS(num_invalids == 0, "Output node states do not match the expected outcome.");
      }
    }

    /* 3. Write node states to CSV file */
    {
      std::cout << "Writing node states to CSV file..." << std::endl;

      // Create vectors to store node IDs and states (only for first pattern)
      rmm::device_uvector<vertex_t> all_vertex_ids(graph_view.number_of_vertices(),
                                                    handle.get_stream());
      rmm::device_uvector<state_t> all_states(graph_view.number_of_vertices(),
                                              handle.get_stream());

      // Populate the vectors (using first pattern only, p=0)
      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(size_t{0}),
        thrust::make_counting_iterator(static_cast<size_t>(graph_view.number_of_vertices())),
        [all_vertex_ids =
           raft::device_span<vertex_t>(all_vertex_ids.data(), all_vertex_ids.size()),
         all_states =
           raft::device_span<state_t>(all_states.data(), all_states.size()),
         node_states =
           raft::device_span<uint32_t const>(node_states.data(), node_states.size()),
         num_words_per_vertex,
         num_vertices   = graph_view.number_of_vertices(),
         num_bits_per_state = num_bits_per_state_,
         state_mask         = state_mask_] __device__(size_t v) {
          all_vertex_ids[v] = static_cast<vertex_t>(v);

          // For pattern 0: word_idx = v * num_words_per_vertex + 0, intra_word_idx = 0
          size_t word_idx = v * num_words_per_vertex;
          size_t intra_word_idx = 0;  // pattern 0 % num_states_per_word

          uint32_t word = node_states[word_idx];
          state_t state = static_cast<state_t>(
            (word >> (intra_word_idx * num_bits_per_state)) & state_mask);
          all_states[v] = state;
        });

      // Unrenumber vertices to get original node IDs
      cugraph::unrenumber_int_vertices<vertex_t, multi_gpu_>(handle,
                                                            all_vertex_ids.data(),
                                                            all_vertex_ids.size(),
                                                            renumber_map.data(),
                                                            graph_view.vertex_partition_range_lasts());

      // Copy to host
      std::vector<vertex_t> h_vertex_ids(all_vertex_ids.size());
      std::vector<state_t> h_states(all_states.size());

      raft::update_host(h_vertex_ids.data(), all_vertex_ids.data(), all_vertex_ids.size(),
      handle.get_stream()); raft::update_host(h_states.data(), all_states.data(),
      all_states.size(), handle.get_stream());

      RAFT_CUDA_TRY(cudaStreamSynchronize(handle.get_stream()));

      std::cout<<"host vertex ids size: "<<h_vertex_ids.size()<<std::endl;
      std::cout<<"host states size: "<<h_states.size()<<std::endl;

      // Write to CSV file
      std::string output_file_name = "node_states_output.csv";
      std::ofstream outfile(output_file_name);

      if (outfile.is_open()) {
        // Write header
        outfile << "node_id,state\n";

        // Write data
        for (size_t i = 0; i < h_vertex_ids.size(); ++i) {
          outfile << h_vertex_ids[i] << "," << static_cast<int>(h_states[i]) << "\n";
        }

        outfile.close();
        std::cout << "Node states written to " << output_file_name << std::endl;
      } else {
        std::cerr << "Failed to open file for writing: " << output_file_name << std::endl;
      }
    }

    /* 4. Write sequential node output table indices to CSV file */
    {
      std::cout << "Writing sequential node output table indices to CSV file..." << std::endl;

      // seq_node_bucket contains num_patterns * num_seq_nodes entries
      // We only need the first pattern's worth of sequential nodes
      size_t num_seq_nodes = seq_node_bucket.size() / num_patterns;

      rmm::device_uvector<vertex_t> seq_vertex_ids(num_seq_nodes, handle.get_stream());
      rmm::device_uvector<size_t> seq_table_indices(num_seq_nodes, handle.get_stream());

      // Extract vertex IDs and table indices for first pattern only
      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(size_t{0}),
        thrust::make_counting_iterator(num_seq_nodes),
        [seq_vertex_ids = raft::device_span<vertex_t>(seq_vertex_ids.data(),
        seq_vertex_ids.size()),
         seq_table_indices = raft::device_span<size_t>(seq_table_indices.data(),
         seq_table_indices.size()), seq_node_bucket_begin = seq_node_bucket.begin(),
         old_seq_node_output_table_indices = raft::device_span<size_t const>(
           old_seq_node_output_table_indices.data(), old_seq_node_output_table_indices.size())]
           __device__(size_t i) {
          auto tagged_v = *(seq_node_bucket_begin + i);  // Get first pattern's entry
          seq_vertex_ids[i] = cuda::std::get<0>(tagged_v);
          seq_table_indices[i] = old_seq_node_output_table_indices[i];
        });

      // Unrenumber vertices to get original node IDs
      cugraph::unrenumber_int_vertices<vertex_t, multi_gpu_>(handle,
                                                            seq_vertex_ids.data(),
                                                            seq_vertex_ids.size(),
                                                            renumber_map.data(),
                                                            graph_view.vertex_partition_range_lasts());

      // Copy to host
      std::vector<vertex_t> h_seq_vertex_ids(seq_vertex_ids.size());
      std::vector<size_t> h_seq_table_indices(seq_table_indices.size());

      raft::update_host(h_seq_vertex_ids.data(), seq_vertex_ids.data(), seq_vertex_ids.size(),
      handle.get_stream()); raft::update_host(h_seq_table_indices.data(),
      seq_table_indices.data(), seq_table_indices.size(), handle.get_stream());

      RAFT_CUDA_TRY(cudaStreamSynchronize(handle.get_stream()));

      // Write to CSV file
      std::string output_file_name = "seq_node_output_table_indices.csv";
      std::ofstream outfile(output_file_name);

      if (outfile.is_open()) {
        // Write header
        outfile << "node_id,table_idx\n";

        // Write data
        for (size_t i = 0; i < h_seq_vertex_ids.size(); ++i) {
          outfile << h_seq_vertex_ids[i] << "," << h_seq_table_indices[i] << "\n";
        }

        outfile.close();
        std::cout << "Sequential node output table indices written to " << output_file_name <<
        std::endl;
      } else {
        std::cerr << "Failed to open file for writing: " << output_file_name << std::endl;
      }
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }
  }

  template <typename vertex_t, typename edge_t, typename cell_idx_t, typename pattern_idx_t>
  NodeBuckets<vertex_t, pattern_idx_t> create_node_buckets(
    raft::handle_t& handle,
    cugraph::graph_view_t<vertex_t, edge_t, store_transposed_, multi_gpu_> const& graph_view,
    raft::device_span<vertex_t const> renumber_map,
    raft::device_span<cell_idx_t const> node_cell_indices,
    raft::device_span<bool const> seq_cell_flags,
    size_t num_patterns)
  {
    NodeBuckets<vertex_t, pattern_idx_t> result(handle);

    {
      rmm::device_uvector<vertex_t> input_nodes(graph_view.number_of_vertices(),
                                                handle.get_stream());
      result.num_input_nodes = cuda::std::distance(
        input_nodes.begin(),
        thrust::copy_if(handle.get_thrust_policy(),
                        thrust::make_counting_iterator(vertex_t{0}),
                        thrust::make_counting_iterator(graph_view.number_of_vertices()),
                        node_cell_indices.begin(),
                        input_nodes.begin(),
                        [input_cell_idx = input_cell_idx_] __device__(auto idx) {
                          return idx == input_cell_idx;
                        }));
      input_nodes.resize(result.num_input_nodes, handle.get_stream());
      input_nodes.shrink_to_fit(handle.get_stream());

      rmm::device_uvector<vertex_t> unrenumbered_vertices(result.num_input_nodes,
                                                          handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   input_nodes.begin(),
                   input_nodes.end(),
                   unrenumbered_vertices.begin());

      result.input_node_bucket =
        cugraph::key_bucket_t<vertex_t, void, multi_gpu_, sorted_unique_key_bucket_>(
          handle, std::move(input_nodes));

      // renumber the input_nodes array
      cugraph::unrenumber_int_vertices<vertex_t, multi_gpu_>(
        handle,
        unrenumbered_vertices.data(),
        result.num_input_nodes,
        renumber_map.data(),
        graph_view.vertex_partition_range_lasts());

      rmm::device_uvector<size_t> unrenumbered_input_nodes_bucket_indices(result.num_input_nodes,
                                                                          handle.get_stream());

      thrust::sequence(handle.get_thrust_policy(),
                       unrenumbered_input_nodes_bucket_indices.begin(),
                       unrenumbered_input_nodes_bucket_indices.end(),
                       size_t{0});

      result.renumbered_input_nodes_sorted_bucket_indices_map =
        cugraph::kv_store_t<vertex_t, size_t, true>(unrenumbered_vertices.begin(),
                                                    unrenumbered_vertices.end(),
                                                    unrenumbered_input_nodes_bucket_indices.begin(),
                                                    std::numeric_limits<size_t>::max(),
                                                    false,
                                                    handle.get_stream());
    }

    {
      rmm::device_uvector<vertex_t> ground_nodes(graph_view.number_of_vertices(),
                                                 handle.get_stream());
      auto num_ground_nodes = cuda::std::distance(
        ground_nodes.begin(),
        thrust::copy_if(handle.get_thrust_policy(),
                        thrust::make_counting_iterator(vertex_t{0}),
                        thrust::make_counting_iterator(graph_view.number_of_vertices()),
                        node_cell_indices.begin(),
                        ground_nodes.begin(),
                        [ground_cell_idx = ground_cell_idx_] __device__(auto idx) {
                          return idx == ground_cell_idx;
                        }));
      ground_nodes.resize(num_ground_nodes, handle.get_stream());
      ground_nodes.shrink_to_fit(handle.get_stream());

      result.ground_node_bucket =
        cugraph::key_bucket_t<vertex_t, void, multi_gpu_, sorted_unique_key_bucket_>(
          handle, std::move(ground_nodes));
    }

    {
      rmm::device_uvector<vertex_t> pwr_nodes(graph_view.number_of_vertices(), handle.get_stream());
      auto num_pwr_nodes = cuda::std::distance(
        pwr_nodes.begin(),
        thrust::copy_if(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator(vertex_t{0}),
          thrust::make_counting_iterator(graph_view.number_of_vertices()),
          node_cell_indices.begin(),
          pwr_nodes.begin(),
          [pwr_cell_idx = pwr_cell_idx_] __device__(auto idx) { return idx == pwr_cell_idx; }));
      pwr_nodes.resize(num_pwr_nodes, handle.get_stream());
      pwr_nodes.shrink_to_fit(handle.get_stream());

      result.pwr_node_bucket =
        cugraph::key_bucket_t<vertex_t, void, multi_gpu_, sorted_unique_key_bucket_>(
          handle, std::move(pwr_nodes));
    }

    {
      rmm::device_uvector<vertex_t> tie_x_nodes(graph_view.number_of_vertices(),
                                                handle.get_stream());
      auto num_tie_x_nodes = cuda::std::distance(
        tie_x_nodes.begin(),
        thrust::copy_if(handle.get_thrust_policy(),
                        thrust::make_counting_iterator(vertex_t{0}),
                        thrust::make_counting_iterator(graph_view.number_of_vertices()),
                        node_cell_indices.begin(),
                        tie_x_nodes.begin(),
                        [tie_x_cell_idx = tie_x_cell_idx_] __device__(auto idx) {
                          return idx == tie_x_cell_idx;
                        }));
      tie_x_nodes.resize(num_tie_x_nodes, handle.get_stream());
      tie_x_nodes.shrink_to_fit(handle.get_stream());

      result.tie_x_node_bucket =
        cugraph::key_bucket_t<vertex_t, void, multi_gpu_, sorted_unique_key_bucket_>(
          handle, std::move(tie_x_nodes));
    }

    {
      rmm::device_uvector<vertex_t> z_nodes(graph_view.number_of_vertices(), handle.get_stream());
      auto num_z_nodes = cuda::std::distance(
        z_nodes.begin(),
        thrust::copy_if(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator(vertex_t{0}),
          thrust::make_counting_iterator(graph_view.number_of_vertices()),
          node_cell_indices.begin(),
          z_nodes.begin(),
          [z_cell_idx = z_cell_idx_] __device__(auto idx) { return idx == z_cell_idx; }));
      z_nodes.resize(num_z_nodes, handle.get_stream());
      z_nodes.shrink_to_fit(handle.get_stream());

      result.z_node_bucket =
        cugraph::key_bucket_t<vertex_t, void, multi_gpu_, sorted_unique_key_bucket_>(
          handle, std::move(z_nodes));
    }

    {
      rmm::device_uvector<vertex_t> output_nodes(graph_view.number_of_vertices(),
                                                 handle.get_stream());

      result.num_output_nodes = cuda::std::distance(
        output_nodes.begin(),
        thrust::copy_if(handle.get_thrust_policy(),
                        thrust::make_counting_iterator(vertex_t{0}),
                        thrust::make_counting_iterator(graph_view.number_of_vertices()),
                        node_cell_indices.begin(),
                        output_nodes.begin(),
                        [output_cell_idx = output_cell_idx_] __device__(auto idx) {
                          return idx == output_cell_idx;
                        }));
      output_nodes.resize(result.num_output_nodes, handle.get_stream());
      output_nodes.shrink_to_fit(handle.get_stream());

      rmm::device_uvector<vertex_t> unrenumbered_output_nodes(result.num_output_nodes,
                                                              handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   output_nodes.begin(),
                   output_nodes.end(),
                   unrenumbered_output_nodes.begin());

      result.output_node_bucket =
        cugraph::key_bucket_t<vertex_t, void, multi_gpu_, sorted_unique_key_bucket_>(
          handle, std::move(output_nodes));

      cugraph::unrenumber_int_vertices<vertex_t, multi_gpu_>(
        handle,
        unrenumbered_output_nodes.data(),
        result.num_output_nodes,
        renumber_map.data(),
        graph_view.vertex_partition_range_lasts());

      rmm::device_uvector<size_t> unrenumbered_output_nodes_bucket_indices(result.num_output_nodes,
                                                                           handle.get_stream());

      thrust::sequence(handle.get_thrust_policy(),
                       unrenumbered_output_nodes_bucket_indices.begin(),
                       unrenumbered_output_nodes_bucket_indices.end(),
                       size_t{0});

      result.renumbered_output_nodes_sorted_bucket_indices_map =
        cugraph::kv_store_t<vertex_t, size_t, true>(
          unrenumbered_output_nodes.begin(),
          unrenumbered_output_nodes.end(),
          unrenumbered_output_nodes_bucket_indices.begin(),
          std::numeric_limits<size_t>::max(),
          false,
          handle.get_stream());

      auto in_degrees = graph_view.compute_in_degrees(handle);
      auto num_invalids =
        thrust::count_if(handle.get_thrust_policy(),
                         output_nodes.begin(),
                         output_nodes.end(),
                         cuda::proclaim_return_type<bool>(
                           [in_degrees = raft::device_span<edge_t const>(
                              in_degrees.data(), in_degrees.size())] __device__(auto v) {
                             return in_degrees[v] != edge_t{1};
                           }));  // output nodes should have the in-degree of 1
      CUGRAPH_EXPECTS(num_invalids == 0, "Output nodes should have the in-degree of 1.");

      result.output_node_inputs.resize(result.output_node_bucket.size(), handle.get_stream());
      cugraph::per_v_transform_reduce_incoming_e(
        handle,
        graph_view,
        result.output_node_bucket,
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        cugraph::edge_dummy_property_t{}.view(),
        cuda::proclaim_return_type<vertex_t>(
          [] __device__(auto src, auto, auto, auto, auto) { return src; }),
        cugraph::invalid_vertex_id<vertex_t>::value,
        cugraph::reduce_op::any<vertex_t>{},
        result.output_node_inputs.begin());
    }

    {
      rmm::device_uvector<vertex_t> seq_nodes(graph_view.number_of_vertices(), handle.get_stream());
      auto num_seq_nodes = cuda::std::distance(
        seq_nodes.begin(),
        thrust::copy_if(handle.get_thrust_policy(),
                        thrust::make_counting_iterator(vertex_t{0}),
                        thrust::make_counting_iterator(graph_view.number_of_vertices()),
                        node_cell_indices.begin(),
                        seq_nodes.begin(),
                        [seq_cell_flags_span = seq_cell_flags] __device__(auto idx) {
                          return (idx >= 0) && seq_cell_flags_span[idx];
                        }));
      seq_nodes.resize(num_seq_nodes, handle.get_stream());
      seq_nodes.shrink_to_fit(handle.get_stream());

      auto seq_node_ids_first = cuda::make_transform_iterator(
        thrust::make_counting_iterator(size_t{0}),
        cuda::proclaim_return_type<vertex_t>(
          [seq_nodes = raft::device_span<vertex_t const>(seq_nodes.data(), seq_nodes.size()),
           num_patterns] __device__(size_t i) {
            auto v = i / num_patterns;
            return seq_nodes[v];
          }));

      auto pattern_ids_first = cuda::make_transform_iterator(
        thrust::make_counting_iterator(size_t{0}),
        cuda::proclaim_return_type<pattern_idx_t>([num_patterns] __device__(size_t i) {
          return static_cast<pattern_idx_t>(i % num_patterns);
        }));

      auto seq_node_bucket_pair = thrust::make_zip_iterator(seq_node_ids_first, pattern_ids_first);

      rmm::device_uvector<vertex_t> sorted_vertices(num_patterns * num_seq_nodes,
                                                    handle.get_stream());
      rmm::device_uvector<pattern_idx_t> sorted_patterns(num_patterns * num_seq_nodes,
                                                         handle.get_stream());

      auto sorted_pair_first =
        thrust::make_zip_iterator(sorted_vertices.begin(), sorted_patterns.begin());

      thrust::copy(handle.get_thrust_policy(),
                   seq_node_bucket_pair,
                   seq_node_bucket_pair + num_patterns * num_seq_nodes,
                   sorted_pair_first);

      result.seq_node_bucket = cugraph::key_bucket_t<vertex_t, pattern_idx_t, multi_gpu_, false>(
        handle, std::move(sorted_vertices), std::move(sorted_patterns));

      result.num_output_nodes_used = result.num_output_nodes;
    }

    return result;
  }

  template <typename vertex_t, typename edge_t, typename cell_idx_t, typename pattern_idx_t>
  LevelizedNodeBuckets<vertex_t, pattern_idx_t> create_levelized_node_buckets(
    raft::handle_t& handle,
    cugraph::graph_view_t<vertex_t, edge_t, store_transposed_, multi_gpu_> const& graph_view,
    raft::device_span<cell_idx_t const> node_cell_indices,
    raft::device_span<bool const> latch_flags,
    raft::device_span<size_t const> last_levels,
    size_t num_levels,
    std::vector<rmm::device_uvector<int32_t>> const& levelized_loop_ids,
    size_t num_patterns)
  {
    LevelizedNodeBuckets<vertex_t, pattern_idx_t> result(handle);

    // Create levelized combinational node buckets
    {
      for (size_t lvl = 0; lvl <= num_levels; lvl++) {
        rmm::device_uvector<vertex_t> this_level_comb_nodes(graph_view.number_of_vertices(),
                                                            handle.get_stream());
        auto num_comb_nodes_this_level = cuda::std::distance(
          this_level_comb_nodes.begin(),
          thrust::copy_if(handle.get_thrust_policy(),
                          thrust::make_counting_iterator(vertex_t{0}),
                          thrust::make_counting_iterator(graph_view.number_of_vertices()),
                          this_level_comb_nodes.begin(),
                          [lvl,
                           last_levels_span       = last_levels,
                           node_cell_indices_span = node_cell_indices,
                           latch_flags_span       = latch_flags] __device__(auto v) {
                            return last_levels_span[v] == lvl && node_cell_indices_span[v] >= 0 &&
                                   !latch_flags_span[node_cell_indices_span[v]];
                          }));
        this_level_comb_nodes.resize(num_comb_nodes_this_level, handle.get_stream());
        this_level_comb_nodes.shrink_to_fit(handle.get_stream());

        auto comb_node_ids_first = cuda::make_transform_iterator(
          thrust::make_counting_iterator(size_t{0}),
          cuda::proclaim_return_type<vertex_t>(
            [this_level_comb_nodes = raft::device_span<vertex_t const>(
               this_level_comb_nodes.data(), this_level_comb_nodes.size()),
             num_patterns] __device__(size_t i) {
              auto v = i / num_patterns;
              return this_level_comb_nodes[v];
            }));

        auto pattern_ids_first = cuda::make_transform_iterator(
          thrust::make_counting_iterator(size_t{0}),
          cuda::proclaim_return_type<pattern_idx_t>([num_patterns] __device__(size_t i) {
            return static_cast<pattern_idx_t>(i % num_patterns);
          }));

        auto comb_node_bucket_pair =
          thrust::make_zip_iterator(comb_node_ids_first, pattern_ids_first);

        rmm::device_uvector<vertex_t> sorted_vertices(num_patterns * num_comb_nodes_this_level,
                                                      handle.get_stream());
        rmm::device_uvector<pattern_idx_t> sorted_patterns(num_patterns * num_comb_nodes_this_level,
                                                           handle.get_stream());

        auto sorted_pair_first =
          thrust::make_zip_iterator(sorted_vertices.begin(), sorted_patterns.begin());

        thrust::copy(handle.get_thrust_policy(),
                     comb_node_bucket_pair,
                     comb_node_bucket_pair + num_patterns * num_comb_nodes_this_level,
                     sorted_pair_first);

        cugraph::key_bucket_t<vertex_t, pattern_idx_t, multi_gpu_, sorted_unique_key_bucket_>
          comb_node_bucket(handle, std::move(sorted_vertices), std::move(sorted_patterns));

        result.levelized_comb_node_buckets.push_back(std::move(comb_node_bucket));
        result.num_comb_nodes += num_comb_nodes_this_level;
      }
    }

    // Create levelized combinational node loop IDs buckets
    {
      for (size_t lvl = 0; lvl <= num_levels; lvl++) {
        auto const& this_level_comb_node_loop_ids = levelized_loop_ids[lvl];

        auto comb_node_loop_ids_first = cuda::make_transform_iterator(
          thrust::make_counting_iterator(size_t{0}),
          cuda::proclaim_return_type<vertex_t>(
            [this_level_comb_node_loop_ids = raft::device_span<int32_t const>(
               this_level_comb_node_loop_ids.data(), this_level_comb_node_loop_ids.size()),
             num_patterns] __device__(size_t i) {
              auto v = i / num_patterns;
              return this_level_comb_node_loop_ids[v];
            }));

        auto pattern_ids_first = cuda::make_transform_iterator(
          thrust::make_counting_iterator(size_t{0}),
          cuda::proclaim_return_type<pattern_idx_t>([num_patterns] __device__(size_t i) {
            return static_cast<pattern_idx_t>(i % num_patterns);
          }));

        auto comb_node_loop_ids_bucket_pair =
          thrust::make_zip_iterator(comb_node_loop_ids_first, pattern_ids_first);

        rmm::device_uvector<int32_t> sorted_loop_ids(
          num_patterns * this_level_comb_node_loop_ids.size(), handle.get_stream());
        rmm::device_uvector<pattern_idx_t> sorted_patterns(
          num_patterns * this_level_comb_node_loop_ids.size(), handle.get_stream());

        auto sorted_pair_first =
          thrust::make_zip_iterator(sorted_loop_ids.begin(), sorted_patterns.begin());

        thrust::copy(
          handle.get_thrust_policy(),
          comb_node_loop_ids_bucket_pair,
          comb_node_loop_ids_bucket_pair + num_patterns * this_level_comb_node_loop_ids.size(),
          sorted_pair_first);

        cugraph::key_bucket_t<int32_t, pattern_idx_t, multi_gpu_, false> comb_node_loop_ids_bucket(
          handle, std::move(sorted_loop_ids), std::move(sorted_patterns));

        result.levelized_comb_node_loop_ids_buckets.push_back(std::move(comb_node_loop_ids_bucket));
      }
    }

    // Build all_latch_nodes (ALL latch vertices, both loop and non-loop), latch_node_to_offset_map,
    // and levelized_latch_node_buckets (non-loop latch nodes only, since loop latches have
    // last_levels = max)
    result.all_latch_nodes.resize(graph_view.number_of_vertices(), handle.get_stream());
    result.latch_node_to_offset_map.resize(graph_view.number_of_vertices(), handle.get_stream());
    {
      result.num_all_latch_nodes = cuda::std::distance(
        result.all_latch_nodes.begin(),
        thrust::copy_if(handle.get_thrust_policy(),
                        thrust::make_counting_iterator(vertex_t{0}),
                        thrust::make_counting_iterator(graph_view.number_of_vertices()),
                        result.all_latch_nodes.begin(),
                        [node_cell_indices_span = node_cell_indices,
                         latch_flags_span       = latch_flags] __device__(auto v) {
                          return node_cell_indices_span[v] >= 0 &&
                                 latch_flags_span[node_cell_indices_span[v]];
                        }));
      result.all_latch_nodes.resize(result.num_all_latch_nodes, handle.get_stream());
      result.all_latch_nodes.shrink_to_fit(handle.get_stream());
      thrust::sort(
        handle.get_thrust_policy(), result.all_latch_nodes.begin(), result.all_latch_nodes.end());

      // latch_node_to_offset_map: vertex v -> position_in_all_latch_nodes * num_patterns
      // Used to index into old_latch_node_output_table_indices for both levelized and loop
      // processing
      thrust::fill(handle.get_thrust_policy(),
                   result.latch_node_to_offset_map.begin(),
                   result.latch_node_to_offset_map.end(),
                   std::numeric_limits<size_t>::max());

      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(size_t{0}),
        thrust::make_counting_iterator(result.num_all_latch_nodes),
        [all_latch_nodes_span = raft::device_span<vertex_t const>(result.all_latch_nodes.data(),
                                                                  result.all_latch_nodes.size()),
         latch_node_to_offset_map_span = raft::device_span<size_t>(
           result.latch_node_to_offset_map.data(), result.latch_node_to_offset_map.size()),
         num_patterns] __device__(size_t pos) {
          auto v                           = all_latch_nodes_span[pos];
          latch_node_to_offset_map_span[v] = pos * num_patterns;
        });
    }

    // Create levelized latch node buckets
    {
      for (size_t lvl = 0; lvl <= num_levels; lvl++) {
        rmm::device_uvector<vertex_t> this_level_latch_nodes(graph_view.number_of_vertices(),
                                                             handle.get_stream());
        auto num_latch_nodes_this_level = cuda::std::distance(
          this_level_latch_nodes.begin(),
          thrust::copy_if(handle.get_thrust_policy(),
                          thrust::make_counting_iterator(vertex_t{0}),
                          thrust::make_counting_iterator(graph_view.number_of_vertices()),
                          this_level_latch_nodes.begin(),
                          [lvl,
                           last_levels_span       = last_levels,
                           node_cell_indices_span = node_cell_indices,
                           latch_flags_span       = latch_flags] __device__(auto v) {
                            return last_levels_span[v] == lvl && node_cell_indices_span[v] >= 0 &&
                                   latch_flags_span[node_cell_indices_span[v]];
                          }));
        this_level_latch_nodes.resize(num_latch_nodes_this_level, handle.get_stream());
        this_level_latch_nodes.shrink_to_fit(handle.get_stream());

        auto latch_node_ids_first = cuda::make_transform_iterator(
          thrust::make_counting_iterator(size_t{0}),
          cuda::proclaim_return_type<vertex_t>(
            [this_level_latch_nodes_span = raft::device_span<vertex_t const>(
               this_level_latch_nodes.data(), this_level_latch_nodes.size()),
             num_patterns] __device__(size_t i) {
              return this_level_latch_nodes_span[i / num_patterns];
            }));

        auto pattern_ids_first = cuda::make_transform_iterator(
          thrust::make_counting_iterator(size_t{0}),
          cuda::proclaim_return_type<pattern_idx_t>([num_patterns] __device__(size_t i) {
            return static_cast<pattern_idx_t>(i % num_patterns);
          }));

        auto latch_node_bucket_pair =
          thrust::make_zip_iterator(latch_node_ids_first, pattern_ids_first);

        rmm::device_uvector<vertex_t> sorted_vertices(num_patterns * num_latch_nodes_this_level,
                                                      handle.get_stream());
        rmm::device_uvector<pattern_idx_t> sorted_patterns(
          num_patterns * num_latch_nodes_this_level, handle.get_stream());

        auto sorted_pair_first =
          thrust::make_zip_iterator(sorted_vertices.begin(), sorted_patterns.begin());

        thrust::copy(handle.get_thrust_policy(),
                     latch_node_bucket_pair,
                     latch_node_bucket_pair + num_patterns * num_latch_nodes_this_level,
                     sorted_pair_first);

        cugraph::key_bucket_t<vertex_t, pattern_idx_t, multi_gpu_, sorted_unique_key_bucket_>
          latch_node_bucket(handle, std::move(sorted_vertices), std::move(sorted_patterns));

        result.levelized_latch_node_buckets.push_back(std::move(latch_node_bucket));
        result.num_latch_nodes += num_latch_nodes_this_level;
      }
    }

    return result;
  }

  // Detects combinational loops by running SCC on the circuit graph with sequential nodes
  // masked out. Returns {filtered_loop_ids, filtered_loop_nodes} sorted by loop_id with
  // contiguous IDs starting from 0 — same format as read_and_filter_loop_id_node_id_pairs.
  template <typename vertex_t, typename edge_t, typename cell_idx_t>
  std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<vertex_t>>
  detect_combinational_loops_via_scc(
    raft::handle_t& handle,
    cugraph::graph_view_t<vertex_t, edge_t, true, false> const& graph_view,
    raft::device_span<cell_idx_t const> node_cell_indices,
    raft::device_span<bool const> seq_cell_flags,
    raft::device_span<vertex_t const> latch_flags,
    raft::device_span<vertex_t const> renumber_map)
  {
    using weight_t    = float;
    using edge_type_t = int32_t;

    auto num_vertices = graph_view.number_of_vertices();

    // 1. Run SCC on a store_transposed=false copy of the graph with sequential edges masked

    rmm::device_uvector<vertex_t> components(0, handle.get_stream());
    {
      // 1a. Decompress edges from the store_transposed=true graph
      auto [edgelist_srcs, edgelist_dsts, ignore_weights, ignore_ids, ignore_types] =
        cugraph::decompress_to_edgelist<vertex_t, edge_t, weight_t, edge_type_t, true, false>(
          handle,
          graph_view,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          std::nullopt);

      // 1b. Create vertex list [0, num_vertices) to preserve isolated vertices
      rmm::device_uvector<vertex_t> vertices(num_vertices, handle.get_stream());
      thrust::sequence(handle.get_thrust_policy(), vertices.begin(), vertices.end(), vertex_t{0});

      // 1c. Create graph with store_transposed=false for SCC
      auto [scc_graph, ignore_edge_props, ignore_renumber_map] =
        cugraph::create_graph_from_edgelist<vertex_t, edge_t, false, false>(
          handle,
          std::make_optional(std::move(vertices)),
          std::move(edgelist_srcs),
          std::move(edgelist_dsts),
          std::vector<cugraph::arithmetic_device_uvector_t>{},
          cugraph::graph_properties_t{false, true},
          false);

      auto scc_graph_view = scc_graph.view();

      // 1d. Create edge mask to exclude edges involving sequential (non-latch) nodes
      cugraph::edge_property_t<edge_t, bool> edge_mask(handle, scc_graph_view);
      cugraph::fill_edge_property(handle, scc_graph_view, edge_mask.mutable_view(), false);

      cugraph::transform_e(
        handle,
        scc_graph_view,
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        cugraph::edge_dummy_property_t{}.view(),
        cuda::proclaim_return_type<bool>(
          [node_cell_indices, seq_cell_flags, latch_flags] __device__(
            auto src, auto dst, auto, auto, auto) {
            auto src_cell = node_cell_indices[src];
            auto dst_cell = node_cell_indices[dst];
            bool src_ok   = (src_cell < 0) || !(seq_cell_flags[src_cell] || latch_flags[src_cell]);
            bool dst_ok   = (dst_cell < 0) || !(seq_cell_flags[dst_cell] || latch_flags[dst_cell]);
            return src_ok && dst_ok;
          }),
        edge_mask.mutable_view());

      scc_graph_view.attach_edge_mask(edge_mask.view());

      // 1e. Run SCC
      components = cugraph::strongly_connected_components(handle, scc_graph_view);
    }

    // 2. Identify non-singleton SCCs (loops)

    // 2a. Sort (component_id, vertex_id) pairs by component_id
    rmm::device_uvector<vertex_t> sorted_components(num_vertices, handle.get_stream());
    rmm::device_uvector<vertex_t> sorted_vertices(num_vertices, handle.get_stream());
    thrust::copy(
      handle.get_thrust_policy(), components.begin(), components.end(), sorted_components.begin());
    thrust::sequence(
      handle.get_thrust_policy(), sorted_vertices.begin(), sorted_vertices.end(), vertex_t{0});
    thrust::sort_by_key(handle.get_thrust_policy(),
                        sorted_components.begin(),
                        sorted_components.end(),
                        sorted_vertices.begin());

    // 2b. Count vertices per component
    rmm::device_uvector<vertex_t> unique_component_ids(num_vertices, handle.get_stream());
    rmm::device_uvector<int32_t> component_counts(num_vertices, handle.get_stream());
    auto reduce_ends = thrust::reduce_by_key(handle.get_thrust_policy(),
                                             sorted_components.begin(),
                                             sorted_components.end(),
                                             cuda::make_constant_iterator(int32_t{1}),
                                             unique_component_ids.begin(),
                                             component_counts.begin());
    auto num_unique_comps =
      static_cast<size_t>(cuda::std::distance(unique_component_ids.begin(), reduce_ends.first));
    unique_component_ids.resize(num_unique_comps, handle.get_stream());
    component_counts.resize(num_unique_comps, handle.get_stream());

    // 2c. Filter to non-singleton components (count > 1)
    rmm::device_uvector<vertex_t> loop_component_ids(num_unique_comps, handle.get_stream());
    auto loop_component_ids_end = thrust::copy_if(
      handle.get_thrust_policy(),
      unique_component_ids.begin(),
      unique_component_ids.end(),
      component_counts.begin(),
      loop_component_ids.begin(),
      cuda::proclaim_return_type<bool>([] __device__(int32_t count) { return count > 1; }));
    auto num_loops =
      static_cast<size_t>(cuda::std::distance(loop_component_ids.begin(), loop_component_ids_end));
    loop_component_ids.resize(num_loops, handle.get_stream());

    std::cout<<"num_loops: "<<num_loops<<std::endl;

    if (num_loops == 0) {
      return std::make_tuple(rmm::device_uvector<int32_t>(0, handle.get_stream()),
                             rmm::device_uvector<vertex_t>(0, handle.get_stream()));
    }

    // 3. Build (loop_id, node_id) pairs

    // 3a. For each (component_id, vertex_id), check if component is a loop via binary search
    rmm::device_uvector<bool> is_loop_vertex(num_vertices, handle.get_stream());
    thrust::binary_search(handle.get_thrust_policy(),
                          loop_component_ids.begin(),
                          loop_component_ids.end(),
                          sorted_components.begin(),
                          sorted_components.end(),
                          is_loop_vertex.begin());

    // 3b. Extract (component_id, vertex_id) pairs belonging to loops
    rmm::device_uvector<vertex_t> loop_component_ids_expanded(num_vertices, handle.get_stream());
    rmm::device_uvector<vertex_t> loop_vertices(num_vertices, handle.get_stream());
    auto zip_in  = thrust::make_zip_iterator(sorted_components.begin(), sorted_vertices.begin());
    auto zip_out = thrust::make_zip_iterator(loop_component_ids_expanded.begin(), loop_vertices.begin());
    auto zip_end = thrust::copy_if(handle.get_thrust_policy(),
                                   zip_in,
                                   zip_in + num_vertices,
                                   is_loop_vertex.begin(),
                                   zip_out,
                                   cuda::proclaim_return_type<bool>(
                                     [] __device__(bool is_loop) { return is_loop; }));
    auto num_loop_vertices =
      static_cast<size_t>(cuda::std::distance(zip_out, zip_end));
    loop_component_ids_expanded.resize(num_loop_vertices, handle.get_stream());
    loop_vertices.resize(num_loop_vertices, handle.get_stream());

    // 3c. Map component_id -> contiguous loop_id via lower_bound
    rmm::device_uvector<int32_t> loop_ids(num_loop_vertices, handle.get_stream());
    thrust::lower_bound(handle.get_thrust_policy(),
                        loop_component_ids.begin(),
                        loop_component_ids.end(),
                        loop_component_ids_expanded.begin(),
                        loop_component_ids_expanded.end(),
                        loop_ids.begin());

    // // 3d. Write pre-filter (loop_id, node_id) pairs to CSV (with original/unrenumbered node IDs)
    // {
    //   std::vector<int32_t> h_loop_ids(num_loop_vertices);
    //   std::vector<vertex_t> h_loop_vertices(num_loop_vertices);
    //   raft::update_host(h_loop_ids.data(), loop_ids.data(), num_loop_vertices, handle.get_stream());
    //   raft::update_host(
    //     h_loop_vertices.data(), loop_vertices.data(), num_loop_vertices, handle.get_stream());

    //   // Unrenumber: map renumbered vertex IDs back to original IDs
    //   std::vector<vertex_t> h_renumber_map(renumber_map.size());
    //   raft::update_host(
    //     h_renumber_map.data(), renumber_map.data(), renumber_map.size(), handle.get_stream());
    //   handle.sync_stream();

    //   std::ofstream ofs("scc_loop_data.csv");
    //   ofs << "loopId,nodeId\n";
    //   for (size_t i = 0; i < num_loop_vertices; ++i) {
    //     ofs << h_loop_ids[i] << "," << h_renumber_map[h_loop_vertices[i]] << "\n";
    //   }
    //   ofs.close();
    //   std::cout << "Wrote " << num_loop_vertices << " pre-filter loop pairs to scc_loop_data.csv"
    //             << std::endl;
    // }

    // 4. Filter special nodes (tie, gnd, pwr, z)

    rmm::device_uvector<int32_t> filtered_loop_ids(num_loop_vertices, handle.get_stream());
    rmm::device_uvector<vertex_t> filtered_loop_nodes(num_loop_vertices, handle.get_stream());
    auto filter_in =
      thrust::make_zip_iterator(loop_ids.begin(), loop_vertices.begin());
    auto filter_out =
      thrust::make_zip_iterator(filtered_loop_ids.begin(), filtered_loop_nodes.begin());
    auto filter_end = thrust::copy_if(
      handle.get_thrust_policy(),
      filter_in,
      filter_in + num_loop_vertices,
      filter_out,
      [node_cell_indices,
       tie_x_cell_idx_v  = dfx_logic_sim::tie_x_cell_idx,
       pwr_cell_idx_v    = dfx_logic_sim::pwr_cell_idx,
       ground_cell_idx_v = dfx_logic_sim::ground_cell_idx,
       z_cell_idx_v      = dfx_logic_sim::z_cell_idx] __device__(auto pair) {
        auto vertex_id = cuda::std::get<1>(pair);
        auto cell_idx  = node_cell_indices[vertex_id];
        return (cell_idx != tie_x_cell_idx_v) && (cell_idx != pwr_cell_idx_v) &&
               (cell_idx != ground_cell_idx_v) && (cell_idx != z_cell_idx_v);
      });

    auto filtered_count =
      static_cast<size_t>(cuda::std::distance(filter_out, filter_end));
    filtered_loop_ids.resize(filtered_count, handle.get_stream());
    filtered_loop_ids.shrink_to_fit(handle.get_stream());
    filtered_loop_nodes.resize(filtered_count, handle.get_stream());
    filtered_loop_nodes.shrink_to_fit(handle.get_stream());

    return std::make_tuple(std::move(filtered_loop_ids), std::move(filtered_loop_nodes));
  }

  template <typename vertex_t, typename edge_t>
  void run_current_test(DfxLogicSim_Usecase const& dfx_logic_sim_usecase)
  {
    using state_t       = uint8_t;  // large enough to fit 0, 1, and X
    using order_t       = edge_t;   // to use edge ID as edge order
    using cell_idx_t    = int16_t;
    using pattern_idx_t = int32_t;
    using weight_t      = float;    // dummy
    using edge_type_t   = int32_t;  // dummy
    using edge_time_t   = int32_t;  // dummy

    size_t num_patterns{1};

    static_assert(sizeof(state_t) * 8 >= num_bits_per_state_);

    static_assert(std::is_same_v<vertex_t, int32_t>);
    static_assert(std::is_same_v<edge_t, int32_t>);
    static_assert(std::is_same_v<order_t, edge_t>);
    static_assert(std::is_same_v<cell_idx_t, int16_t>);

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    // 1. read edge list and create graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Construct graph");
    }

    std::string circuit_file_path =
      cugraph::test::get_rapids_dataset_root_dir() + "/" + dfx_logic_sim_usecase.circuit_file_name;
    auto circuit_graph_result =
      dfx_logic_sim::read_circuit_graph_from_csv_file<vertex_t, edge_t, weight_t>(
        handle, circuit_file_path);
    auto& graph        = circuit_graph_result.graph;
    auto& edge_orders  = circuit_graph_result.edge_orders;
    auto& renumber_map = circuit_graph_result.renumber_map;

    auto graph_view      = graph.view();
    auto edge_order_view = edge_orders.view();

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    /* 2. Read cell (gate) output table file */

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Read cell output table file");
    }

    std::string cell_output_table_file = cugraph::test::get_rapids_dataset_root_dir() + "/" +
                                         dfx_logic_sim_usecase.cell_output_table_file_name;
    auto cell_output_table_result =
      dfx_logic_sim::read_cell_output_tables<cell_idx_t, state_t>(handle, cell_output_table_file);

    auto num_cell_types = cell_output_table_result.num_cell_types;
    auto max_table_size = cell_output_table_result.max_table_size;

    rmm::device_uvector<size_t> idx_multipliers(0, handle.get_stream());
    {
      std::vector<size_t> h_idx_multipliers{};
      size_t multiplier{1};
      while (multiplier < max_table_size) {
        h_idx_multipliers.push_back(multiplier);
        multiplier *= num_valid_values_per_state_;
      }
      CUGRAPH_EXPECTS(multiplier == max_table_size,
                      "max_table_size should be a power of num_valid_values_per_state.");
      idx_multipliers.resize(h_idx_multipliers.size(), handle.get_stream());
      raft::update_device(idx_multipliers.data(),
                          h_idx_multipliers.data(),
                          h_idx_multipliers.size(),
                          handle.get_stream());
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    /* 3. Read node to cell index map file & update cell input degrees*/

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Read node-to-cell index map file");
    }

    std::cout << "graph_view.number_of_vertices(): " << graph_view.number_of_vertices()
              << std::endl;

    std::string node2cell_map_file = cugraph::test::get_rapids_dataset_root_dir() + "/" +
                                     dfx_logic_sim_usecase.node2cell_map_file_name;
    auto node_cell_map = dfx_logic_sim::read_node_cell_map<vertex_t, edge_t, cell_idx_t>(
      handle, node2cell_map_file, graph_view, renumber_map.data(), num_cell_types);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 4. Read the sequential cell flag, latch flag and zstate flag file

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Read sequential cell flag file");
    }

    CellTypeBooleanFlags cell_flags(handle);
    cell_flags.seq_cell_flags = dfx_logic_sim::read_boolean_flags_from_csv_file<cell_idx_t>(
      handle,
      cugraph::test::get_rapids_dataset_root_dir() + "/" +
        dfx_logic_sim_usecase.seq_cell_flag_file_name,
      num_cell_types);

    cell_flags.latch_flags = dfx_logic_sim::read_boolean_flags_from_csv_file<cell_idx_t>(
      handle,
      cugraph::test::get_rapids_dataset_root_dir() + "/" +
        dfx_logic_sim_usecase.latch_flag_file_name,
      num_cell_types);

    cell_flags.zstate_nodes_flags = dfx_logic_sim::read_boolean_flags_from_csv_file<cell_idx_t>(
      handle,
      cugraph::test::get_rapids_dataset_root_dir() + "/" +
        dfx_logic_sim_usecase.zstate_nodes_flag_file_name,
      num_cell_types);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    /* 5. Optimized level implementation for combinational nodes */
    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      hr_timer.start("Level based optimization");
    }

    // Variables needed after the scope block
    rmm::device_uvector<vertex_t> filtered_loop_nodes(0, handle.get_stream());
    rmm::device_uvector<size_t> last_levels(0, handle.get_stream());
    size_t num_levels = 0;
    LoopDataResult<vertex_t, order_t> loop_data_result(handle);

    {
      /*5.1 Read loop id, node id pairs from file*/

      std::string file_name = cugraph::test::get_rapids_dataset_root_dir() + "/" +
                              dfx_logic_sim_usecase.loop_id_node_id_pairs_file_name;
      rmm::device_uvector<int32_t> filtered_loop_ids(0, handle.get_stream());
      std::tie(filtered_loop_ids, filtered_loop_nodes) =
        dfx_logic_sim::read_and_filter_loop_id_node_id_pairs<vertex_t, cell_idx_t>(
          handle,
          raft::device_span<cell_idx_t const>(node_cell_map.node_cell_indices.data(),
                                              node_cell_map.node_cell_indices.size()),
          raft::device_span<vertex_t const>(renumber_map.data(), renumber_map.size()),
          file_name,
          graph_view.number_of_vertices());


      rmm::device_uvector<int32_t> filtered_loop_ids_scc(0, handle.get_stream());
      rmm::device_uvector<vertex_t> filtered_loop_nodes_scc(0, handle.get_stream());

      std::tie(filtered_loop_ids_scc, filtered_loop_nodes_scc) = detect_combinational_loops_via_scc<vertex_t, edge_t, cell_idx_t>(
        handle,
        graph_view,
        raft::device_span<cell_idx_t const>(node_cell_map.node_cell_indices.data(),
                                            node_cell_map.node_cell_indices.size()),
        raft::device_span<bool const>(cell_flags.seq_cell_flags.data(),
                                      cell_flags.seq_cell_flags.size()),
        raft::device_span<bool const>(cell_flags.latch_flags.data(),
                                      cell_flags.latch_flags.size()),
        raft::device_span<vertex_t const>(renumber_map.data(), renumber_map.size()));


      cugraph::unrenumber_int_vertices<vertex_t, multi_gpu_>(
        handle,
        filtered_loop_nodes_scc.data(),
        filtered_loop_nodes_scc.size(),
        renumber_map.data(),
        graph_view.vertex_partition_range_lasts());

      cugraph::unrenumber_int_vertices<vertex_t, multi_gpu_>(
        handle,
        filtered_loop_nodes.data(),
        filtered_loop_nodes.size(),
        renumber_map.data(),
        graph_view.vertex_partition_range_lasts());

      // raft::print_device_vector("renumber_map", renumber_map.data(), renumber_map.size(), std::cout);

      Verify CSV-based and SCC-based loop detection identify the same node groups
      {
        auto to_sorted_groups = [&handle](rmm::device_uvector<int32_t> const& ids,
                                          rmm::device_uvector<vertex_t> const& nodes) {
          std::vector<int32_t> h_ids(ids.size());
          std::vector<vertex_t> h_nodes(nodes.size());
          raft::update_host(h_ids.data(), ids.data(), ids.size(), handle.get_stream());
          raft::update_host(h_nodes.data(), nodes.data(), nodes.size(), handle.get_stream());
          handle.sync_stream();

          std::unordered_map<int32_t, std::vector<vertex_t>> groups;
          for (size_t i = 0; i < h_ids.size(); ++i) {
            groups[h_ids[i]].push_back(h_nodes[i]);
          }
          std::vector<std::vector<vertex_t>> result;
          for (auto& [id, g] : groups) {
            std::sort(g.begin(), g.end());
            result.push_back(std::move(g));
          }
          std::sort(result.begin(), result.end());
          return result;
        };

        auto csv_groups = to_sorted_groups(filtered_loop_ids, filtered_loop_nodes);
        auto scc_groups = to_sorted_groups(filtered_loop_ids_scc, filtered_loop_nodes_scc);

        std::cout << "CSV loops: " << csv_groups.size()
                  << ", SCC loops: " << scc_groups.size() << std::endl;

        ASSERT_EQ(csv_groups, scc_groups)
          << "CSV and SCC loop detection identified different node groups.";
      }

      // //Classify loops based on whether they are purely combinational or latches
      // {
      //   rmm::device_uvector<uint8_t> is_latch(filtered_loop_ids.size(), handle.get_stream());
      //   thrust::fill(handle.get_thrust_policy(), is_latch.begin(), is_latch.end(), uint8_t{0});
      //   thrust::transform(handle.get_thrust_policy(),
      //                     filtered_loop_nodes.begin(),
      //                     filtered_loop_nodes.end(),
      //                     is_latch.begin(),
      //                     [node_cell_indices = raft::device_span<cell_idx_t const>(node_cell_map.node_cell_indices.data(), node_cell_map.node_cell_indices.size()),
      //                      latch_flags = raft::device_span<bool const>(cell_flags.latch_flags.data(), cell_flags.latch_flags.size())] __device__(auto node) {
      //                       auto cell_idx = node_cell_indices[node];
      //                       return (cell_idx >= 0) && latch_flags[cell_idx] ? uint8_t{1} : uint8_t{0};
      //                     });

      //   rmm::device_uvector<int32_t> latch_loop_ids(filtered_loop_ids.size(), handle.get_stream());
      //   rmm::device_uvector<vertex_t> latch_max_values(filtered_loop_ids.size(), handle.get_stream());
      //   auto latch_end = thrust::reduce_by_key(handle.get_thrust_policy(),
      //                         filtered_loop_ids.begin(),
      //                         filtered_loop_ids.end(),
      //                         is_latch.begin(),
      //                         latch_loop_ids.begin(),
      //                         latch_max_values.begin(),
      //                         cuda::std::equal_to<int32_t>{},
      //                         thrust::maximum<vertex_t>{});

      //   auto num_loops_total = cuda::std::distance(latch_loop_ids.begin(), latch_end.first);

      //   latch_loop_ids.resize(num_loops_total, handle.get_stream());
      //   latch_max_values.resize(num_loops_total, handle.get_stream());
      //   latch_loop_ids.shrink_to_fit(handle.get_stream());
      //   latch_max_values.shrink_to_fit(handle.get_stream());

      //   auto num_latch_loops = thrust::count(handle.get_thrust_policy(),
      //                                        latch_max_values.begin(),
      //                                        latch_max_values.end(),
      //                                        uint8_t{1});
        
      //   auto num_combinational_loops = num_loops_total - num_latch_loops;

      //   std::cout << "Number of latch loops: " << num_latch_loops << std::endl;
      //   std::cout << "Number of combinational loops: " << num_combinational_loops << std::endl;
      // }

      /* 5.2: Create coarsen graph */
      auto coarsen_result = create_coarsen_graph<vertex_t, edge_t, weight_t, cell_idx_t>(
        handle,
        graph_view,
        raft::device_span<int32_t const>(filtered_loop_ids.data(), filtered_loop_ids.size()),
        raft::device_span<vertex_t const>(filtered_loop_nodes.data(), filtered_loop_nodes.size()),
        raft::device_span<cell_idx_t const>(node_cell_map.node_cell_indices.data(),
                                            node_cell_map.node_cell_indices.size()));

      /* 5.3: Perform levelization */
      std::tie(last_levels, num_levels) = perform_levelization<vertex_t, edge_t, cell_idx_t>(
        handle,
        coarsen_result,
        raft::device_span<bool const>(cell_flags.seq_cell_flags.data(),
                                      cell_flags.seq_cell_flags.size()),
        graph_view.number_of_vertices());

      /* 5.4: Build loop data structures */
      loop_data_result =
        build_loop_data_structures<vertex_t, edge_t, weight_t, order_t, edge_type_t, cell_idx_t>(
          handle,
          graph_view,
          edge_order_view,
          coarsen_result,
          raft::device_span<int32_t const>(filtered_loop_ids.data(), filtered_loop_ids.size()),
          raft::device_span<vertex_t const>(filtered_loop_nodes.data(), filtered_loop_nodes.size()),
          raft::device_span<size_t>(last_levels.data(), last_levels.size()),
          num_levels,
          raft::device_span<cell_idx_t const>(node_cell_map.node_cell_indices.data(),
                                              node_cell_map.node_cell_indices.size()),
          raft::device_span<bool const>(cell_flags.seq_cell_flags.data(),
                                        cell_flags.seq_cell_flags.size()));
    }

    auto& levelized_loop_ids = loop_data_result.levelized_loop_ids;

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    /* 7. Create key buckets for input/ground/sequential/combinational nodes */

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Create key buckets for input/sequential/combinational nodes");
    }

    auto node_buckets = create_node_buckets<vertex_t, edge_t, cell_idx_t, pattern_idx_t>(
      handle,
      graph_view,
      raft::device_span<vertex_t const>(renumber_map.data(), renumber_map.size()),
      raft::device_span<cell_idx_t const>(node_cell_map.node_cell_indices.data(),
                                          node_cell_map.node_cell_indices.size()),
      raft::device_span<bool const>(cell_flags.seq_cell_flags.data(),
                                    cell_flags.seq_cell_flags.size()),
      num_patterns);

    auto levelized_buckets =
      create_levelized_node_buckets<vertex_t, edge_t, cell_idx_t, pattern_idx_t>(
        handle,
        graph_view,
        raft::device_span<cell_idx_t const>(node_cell_map.node_cell_indices.data(),
                                            node_cell_map.node_cell_indices.size()),
        raft::device_span<bool const>(cell_flags.latch_flags.data(), cell_flags.latch_flags.size()),
        raft::device_span<size_t const>(last_levels.data(), last_levels.size()),
        num_levels,
        levelized_loop_ids,
        num_patterns);

#if 1  // DEBUG
    std::cout << "num_seq_nodes: " << node_buckets.seq_node_bucket.size() << std::endl;
    std::cout << "num_comb_nodes: " << levelized_buckets.num_comb_nodes << std::endl;
    std::cout << "num_all_latch_nodes: " << levelized_buckets.num_all_latch_nodes << std::endl;
    std::cout << "num_latch_nodes (levelized): " << levelized_buckets.num_latch_nodes << std::endl;
#endif

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    /* 8. Read simulation input data from files and initialize node_states */
    auto num_words_per_vertex = (num_patterns + num_states_per_word_ - 1) / num_states_per_word_;
    rmm::device_uvector<uint32_t> node_states(
      graph_view.number_of_vertices() * num_words_per_vertex, handle.get_stream());

    auto sim_input_data = dfx_logic_sim::
      read_simulation_input_data<vertex_t, edge_t, state_t, order_t, cell_idx_t, pattern_idx_t>(
        handle,
        hr_timer,
        graph_view,
        dfx_logic_sim_usecase,
        num_patterns,
        node_buckets,
        node_cell_map,
        levelized_buckets,
        raft::device_span<vertex_t const>(renumber_map.data(), renumber_map.size()),
        raft::device_span<uint32_t>(node_states.data(), node_states.size()),
        raft::device_span<size_t const>(idx_multipliers.data(), idx_multipliers.size()));

    /* 9. Run simulation */
    run_simulation_cycles(
      handle,
      hr_timer,
      graph_view,
      edge_order_view,
      sim_input_data,
      node_buckets,
      levelized_buckets,
      node_cell_map,
      cell_output_table_result,
      loop_data_result,
      cell_flags,
      raft::device_span<size_t const>(idx_multipliers.data(), idx_multipliers.size()),
      raft::device_span<vertex_t const>(renumber_map.data(), renumber_map.size()),
      raft::device_span<vertex_t const>(filtered_loop_nodes.data(), filtered_loop_nodes.size()),
      raft::device_span<uint32_t>(node_states.data(), node_states.size()),
      num_patterns,
      num_levels);
  }
};

using Tests_DfxLogicSim_File = Tests_DfxLogicSim;

TEST_P(Tests_DfxLogicSim_File, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(file_test,
                         Tests_DfxLogicSim_File,
                         ::testing::Values(DfxLogicSim_Usecase{
#if 1
                           "GBA_test/NV_gba_sm0_verific_fixed_circuit.csv",
                           "common/global_truth_table.csv",
                           "GBA_test/NV_gba_sm0_verific_fixed_nodeId_cellIdx_map.csv",
                           "common/global_cell_sequential_flags.csv",
                           "common/global_cell_latch_flags.csv",
                           "common/zstate_node_flags.csv",
                           "GBA_test/NV_gba_sm0_loop_data.csv",
                           "GBA_test/input_patterns_setup.csv",
                           std::nullopt, //"GBA_test/output_patterns_setup.csv"
                           std::nullopt, //  "GBA_test/scan_output_indices.csv",
                           std::nullopt, //  "GBA_test/node_states_output.csv",
                           std::nullopt, //  "GBA_test/seq_node_output_table_indices.csv"
// #else
//                            "latches/250_circuit.csv",
//                            "latches/global_truth_table.csv",
//                            "latches/250_nodeId_cellIdx_map.csv",
//                            "latches/global_cell_sequential_flags.csv",
//                            "latches/global_cell_latch_flags.csv",
//                            "latches/zstate_node_flags.csv",
//                            "latches/25_combinational_loop_data.csv",
//                            "latches/250_test_patterns_inputs.csv",
//                            "latches/250_test_patterns_outputs.csv",
//                            std::nullopt,
//                            std::nullopt,
//                            std::nullopt,
//  "GBA_test/scan_output_indices.csv",
//  "GBA_test/node_states_output.csv",
//  "GBA_test/seq_node_output_table_indices.csv"

#else
                           "combinational_loops/5000/5000_circuit.csv",
                           "common/global_truth_table.csv",
                           "combinational_loops/5000/5000_nodeid_cellIdx_map.csv",
                           "common/global_cell_sequential_flags.csv",
                           "common/global_cell_latch_flags.csv",
                           "common/zstate_node_flags.csv",
                           "combinational_loops/5000/5000_combinational_loop_data.csv",
                           "combinational_loops/5000/5000_patterns_inputs.csv",
                           "combinational_loops/5000/5000_patterns_outputs.csv",
                           std::nullopt,
                           std::nullopt,
                           std::nullopt,

// #else
//                            "gloam/jpeg_circuit.csv",
//                            "gloam/global_truth_tables.csv",
//                            "gloam/jpeg_nodeId_cellIdx_map.csv",
//                            "gloam/global_cell_sequential_flags.csv",
//                            "gloam/zstate_node_flags.csv",
//                            "gloam/jpeg_combinational_loop_data.csv",
//                            "gloam/jpeg_test_patterns_inputs.csv",
//                            "zstates_test/1000_test_patterns_outputs.csv",
//                            "GBA_test/scan_output_indices.csv",
//                            "GBA_test/node_states_output.csv",
//                            "GBA_test/seq_node_output_table_indices.csv"

#endif
                         }));

#pragma GCC diagnostic pop

CUGRAPH_TEST_PROGRAM_MAIN()
