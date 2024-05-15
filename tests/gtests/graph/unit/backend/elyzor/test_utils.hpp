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
#ifndef BACKEND_ELYZOR_TEST_UTILS_HPP
#define BACKEND_ELYZOR_TEST_UTILS_HPP

#include "interface/graph.hpp"
#include "src/common/utils.hpp"

#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

namespace impl = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;

using ltsr_vec = std::vector<impl::logical_tensor_t>;

// ONEDNN_ENABLE_ELYZOR_GRAPH_COMPILER_LIB_TESTS=0 - skip tests that require 'libgraph_compiler.so' loading
#define SKIP_WHEN_NO_LIB_TESTS() \
    if (!dnnl::impl::getenv_int_user("ENABLE_ELYZOR_GRAPH_COMPILER_LIB_TESTS", \
                /*default_value=*/1)) { \
        GTEST_SKIP(); \
        return; \
    }

#define DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(name) \
    name.set_attr(graph::op_attr::scales, std::vector<float>({0.12f})); \
    name.set_attr(graph::op_attr::zps, std::vector<int64_t>({2})); \
    name.set_attr(graph::op_attr::qtype, std::string("per_tensor")); \
    name.set_attr(graph::op_attr::axis, (int64_t)0);

inline void construct_mul_quantize_subgraph(graph::graph_t *agraph,
        utils::id_generator &id_gen, const graph::dims &input_shape,
        bool is_mixed_precision = false) {
    bool known_shapes = input_shape.size() != 0;
    auto dtype = is_mixed_precision ? graph::data_type::bf16
                                    : graph::data_type::f32;

    dnnl::impl::graph::logical_tensor_t mul_in, smooth_quant_scale, mul_out,
            quant_out;
    if (known_shapes) {
        graph::dims smooth_quant_scales_shape {
                input_shape[input_shape.size() - 1]};
        mul_in = utils::logical_tensor_init(
                id_gen.get_id(), input_shape, dtype);
        smooth_quant_scale = utils::logical_tensor_init(id_gen.get_id(),
                smooth_quant_scales_shape, graph::data_type::f32);
        mul_out = utils::logical_tensor_init(
                id_gen.get_id(), input_shape, graph::data_type::f32);
        quant_out = utils::logical_tensor_init(
                id_gen.get_id(), input_shape, graph::data_type::u8);
    } else {
        mul_in = utils::logical_tensor_init(id_gen.get_id(), dtype);
        smooth_quant_scale = utils::logical_tensor_init(
                id_gen.get_id(), graph::data_type::f32);
        mul_out = utils::logical_tensor_init(
                id_gen.get_id(), graph::data_type::f32);
        quant_out = utils::logical_tensor_init(
                id_gen.get_id(), graph::data_type::u8);
    }

    graph::op_t mul {id_gen.get_id(), graph::op_kind::Multiply, "mul"};
    mul.set_attr(graph::op_attr::auto_broadcast, std::string("numpy"));
    graph::op_t quantize {
            id_gen.get_id(), graph::op_kind::Quantize, "quantize"};
    DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(quantize);

    mul.add_input(mul_in);
    mul.add_input(smooth_quant_scale);
    mul.add_output(mul_out);
    quantize.add_input(mul_out);
    quantize.add_output(quant_out);

    agraph->add_op(&mul);
    agraph->add_op(&quantize);
}

std::vector<dnnl_graph_compiler_tensor> test_tensor_to_gc_tensor(
        std::vector<test_tensor> &in) {
    std::vector<dnnl_graph_compiler_tensor> res;
    res.reserve(in.size());

    for (auto &test_tensr : in) {
        auto tensr = test_tensr.get();
        auto lt = tensr.get_logical_tensor();
        res.emplace_back(dnnl_graph_compiler_tensor {.id = lt.id,
                .ndims = static_cast<uint8_t>(lt.ndims),
                .dims = lt.dims,
                .data = tensr.get_data_handle()});
    }

    return res;
}

static void compile_execution_pipeline(impl::graph_t &agraph,
        int expected_part_size,
        std::function<void(ltsr_vec &, ltsr_vec &)> dynamic_callback
        = nullptr) {
    auto &elyzor_backend_ptr = impl::elyzor::elyzor_backend_t::get_singleton();
    elyzor_backend_ptr.get_partitions(agraph, impl::partition_policy::fusion);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), static_cast<size_t>(expected_part_size));
    if (dynamic_callback) { ASSERT_EQ(expected_part_size, 1); }
    // TODO(yifei): generalize the logic here
    // sort partitions to run forward first according to num ops
    std::sort(partitions.begin(), partitions.end(),
            [](std::shared_ptr<impl::partition_impl_t> a,
                    std::shared_ptr<impl::partition_impl_t> b) {
                return a->get_ops().size() < b->get_ops().size();
            });

    std::unordered_map<size_t, impl::logical_tensor_t> lt_info_map;

    for (size_t i = 0; i < partitions.size(); ++i) {
        impl::partition_t p;
        p.init(partitions[i]);
        auto partition_inputs = p.get_inputs();
        auto partition_outputs = p.get_outputs();

        // replace partition inputs info if needed
        for (size_t i = 0; i < partition_inputs.size(); ++i) {
            if (lt_info_map.find(partition_inputs[i].id) != lt_info_map.end()) {
                partition_inputs[i] = lt_info_map[partition_inputs[i].id];
            }
        }

        std::vector<const impl::logical_tensor_t *> inputs;
        std::vector<const impl::logical_tensor_t *> outputs;
        for (auto &lt : partition_inputs) {
            inputs.push_back(&lt);
        }
        for (auto &lt : partition_outputs) {
            outputs.push_back(&lt);
        }
        impl::compiled_partition_t cp(p);
        impl::engine_t &eng = *get_engine();
        ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

        std::vector<test_tensor> execution_inputs;
        std::vector<test_tensor> execution_outputs;
        partition_outputs.clear();
        for (auto &lt : outputs) {
            impl::logical_tensor_t compiled_output;
            cp.query_logical_tensor(lt->id, &compiled_output);
            partition_outputs.push_back(compiled_output);
            assert(compiled_output.ndims > -1);
        }
        if (dynamic_callback) {
            dynamic_callback(partition_inputs, partition_outputs);
        }
        for (auto &lt : partition_inputs) {
            assert(lt.ndims > -1);
            lt_info_map[lt.id] = lt;
        }
        for (auto &lt : partition_outputs) {
            assert(lt.ndims > -1);
            lt_info_map[lt.id] = lt;
        }

        for (auto &lt : partition_inputs) {
            test_tensor placeholder(lt, &eng);
            execution_inputs.push_back(placeholder);
        }
        for (auto &lt : partition_outputs) {
            test_tensor placeholder(lt, &eng);
            execution_outputs.push_back(placeholder);
        }

        impl::stream_t &strm = *get_stream();
        ASSERT_EQ(cp.execute(&strm,
                          test_tensor::to_graph_tensor(execution_inputs),
                          test_tensor::to_graph_tensor(execution_outputs)),
                impl::status::success);
        strm.wait();
    }
}

#endif
