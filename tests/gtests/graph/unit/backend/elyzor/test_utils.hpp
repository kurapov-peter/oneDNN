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

#include "backend/elyzor/elyzor_partition_impl.hpp"
#include "interface/graph.hpp"

#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

namespace impl = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;

using ltsr_vec = std::vector<impl::logical_tensor_t>;

#define DEFINE_DEFAULT_PER_TENSOR_QUANT_ATTR(name) \
    name.set_attr(graph::op_attr::scales, std::vector<float>({0.12f})); \
    name.set_attr(graph::op_attr::zps, std::vector<int64_t>({2})); \
    name.set_attr(graph::op_attr::qtype, std::string("per_tensor")); \
    name.set_attr(graph::op_attr::axis, (int64_t)0);

class test_elyzor_partition_impl_t {
public:
    static std::vector<std::shared_ptr<impl::op_t>> get_copied_ops(
            std::shared_ptr<impl::elyzor::elyzor_partition_impl_t> part) {
        return part->copied_ops_;
    }
};

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

#endif
