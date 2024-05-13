/*******************************************************************************
* Copyright 2021-2024 Intel Corporation
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
*******************************************************************************/
#include "backend/elyzor/elyzor_backend.hpp"
#include "backend/elyzor/elyzor_partition_impl.hpp"
#include "interface/allocator.hpp"
#include "interface/graph.hpp"
#include "interface/partition.hpp"
#include "test_utils.hpp"

#include <gtest/gtest.h>

// the tests below use getters that are only defined under '!NDEBUG'
#ifndef NDEBUG

// Verify that after calling 'elyzor_partition_impl_t::infer_shape' the input shapes
// are propagated to the 'copied_ops_' field (the one that is used to dump the graph to JSON)
TEST(ElyzorGraphTest, CompleteInputShapes) {
    utils::id_generator id_gen;
    impl::graph_t agraph;
    impl::dims initial_shapes = {};
    // constructing a graph providing no shapes
    construct_mul_quantize_subgraph(&agraph, id_gen, initial_shapes);
    agraph.finalize();

    // map<tensor_id, input_shape>
    std::unordered_map<size_t, impl::dims> input_shapes
            = {{0, {10}}, {1, {10}}};
    std::unordered_map<size_t, impl::dims> expected_complete_shapes
            = {{0, {10}}, {1, {10}}, {2, {10}}, {3, {10}}};

    auto &backend_ptr = impl::elyzor::elyzor_backend_t::get_singleton();
    backend_ptr.get_partitions(agraph, impl::partition_policy::fusion);
    auto partitions = agraph.get_partitions();
    for (auto &part : partitions) {
        std::vector<std::shared_ptr<graph::logical_tensor_t>> inputs_ptr,
                outputs_ptr;
        std::vector<const graph::logical_tensor_t *> inputs;
        std::vector<graph::logical_tensor_t *> outputs;
        for (auto &lt : part->get_inputs()) {
            // verify that the input shapes are unknown
            ASSERT_EQ(lt.ndims, DNNL_GRAPH_UNKNOWN_NDIMS);
            ASSERT_EQ(lt.dims[0], DNNL_GRAPH_UNKNOWN_DIM);

            auto new_lt = std::make_shared<impl::logical_tensor_t>(
                    utils::logical_tensor_init(
                            lt.id, input_shapes[lt.id], lt.data_type));
            ASSERT_NE(new_lt->ndims, DNNL_GRAPH_UNKNOWN_NDIMS);
            ASSERT_NE(new_lt->dims[0], DNNL_GRAPH_UNKNOWN_DIM);

            inputs_ptr.push_back(new_lt);
            inputs.push_back(new_lt.get());
        }
        for (auto &lt : part->get_outputs()) {
            // verify that the output shapes are unknown
            ASSERT_EQ(lt.ndims, DNNL_GRAPH_UNKNOWN_NDIMS);
            ASSERT_EQ(lt.dims[0], DNNL_GRAPH_UNKNOWN_DIM);
            auto new_lt = std::make_shared<impl::logical_tensor_t>(
                    utils::logical_tensor_init(lt.id, lt.data_type));
            ASSERT_EQ(new_lt->ndims, DNNL_GRAPH_UNKNOWN_NDIMS);
            ASSERT_EQ(new_lt->dims[0], DNNL_GRAPH_UNKNOWN_DIM);
            outputs_ptr.push_back(new_lt);
            outputs.push_back(new_lt.get());
        }

        part->infer_shape(inputs, outputs);
        // getting the field containing a graph to be converted to JSON
        auto ops = std::static_pointer_cast<
                impl::elyzor::elyzor_partition_impl_t>(part)
                           ->get_copied_ops();
        // iterating over the graph and verifying that all the shapes are complete
        for (auto &op : ops) {
            for (auto &in_val : op->get_input_values()) {
                auto lt = in_val->get_logical_tensor();
                ASSERT_NE(lt.ndims, DNNL_GRAPH_UNKNOWN_NDIMS);
                ASSERT_NE(lt.dims[0], DNNL_GRAPH_UNKNOWN_DIM);
                auto &complete_shape = expected_complete_shapes[lt.id];
                for (size_t i = 0; i < complete_shape.size(); i++) {
                    ASSERT_EQ(lt.dims[i], complete_shape[i]);
                }
            }
            for (auto &out_val : op->get_output_values()) {
                auto lt = out_val->get_logical_tensor();
                ASSERT_NE(lt.ndims, DNNL_GRAPH_UNKNOWN_NDIMS);
                ASSERT_NE(lt.dims[0], DNNL_GRAPH_UNKNOWN_DIM);
                auto &complete_shape = expected_complete_shapes[lt.id];
                for (size_t i = 0; i < complete_shape.size(); i++) {
                    ASSERT_EQ(lt.dims[i], complete_shape[i]);
                }
            }
        }
    }
}

#endif
