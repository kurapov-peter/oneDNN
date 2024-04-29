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
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "compiler_partition_impl.hpp"

#include "common/rw_mutex.hpp"
#include "common/verbose.hpp"
#include "compiler_loader.hpp"
#include "graph/interface/graph.hpp"
#include "graph/utils/debug.hpp"
#include "graph/utils/utils.hpp"
#include "utils.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace elyzor {

graph::status_t compiler_partition_impl_t::infer_shape(
        std::vector<const graph::logical_tensor_t *> &inputs,
        std::vector<graph::logical_tensor_t *> &outputs) const {
    std::lock_guard<std::mutex> lck(mtx_);
    // construct a temp graph
    copied_ops_ = graph::graph_t::deep_copy(ops_);
    graph::graph_t temp_graph(copied_ops_);
    auto output_ops = temp_graph.get_output_ops();
    auto ret = topo_order_visit(output_ops, [&](op_t *cur_op) {
        const graph::op_schema_t *cur_op_schema
                = graph::op_schema_registry_t::get_op_schema(
                        cur_op->get_kind());
        assertm(cur_op_schema, "Can't infer shape for cur op: no schema");
        auto get_logical_tensor = [&](const std::shared_ptr<value_t> &val)
                -> graph::logical_tensor_t {
            logical_tensor_t lt = val->get_logical_tensor();
            auto in_pos = std::find_if(inputs.begin(), inputs.end(),
                    [&](const graph::logical_tensor_t *alt) -> bool {
                        return alt->id == lt.id;
                    });
            if (in_pos != inputs.end()) { return **in_pos; }
            return lt;
        };
        graph::op_t temp_node = graph::op_t(cur_op->get_kind());
        temp_node.merge_attributes(cur_op->get_attributes());
        std::vector<graph::logical_tensor_t> ordered_inputs_holder
                = utils::func_map(
                        cur_op->get_input_values(), get_logical_tensor);
        std::vector<graph::logical_tensor_t> ordered_outputs_holder
                = utils::func_map(
                        cur_op->get_output_values(), get_logical_tensor);
        std::vector<graph::logical_tensor_t *> ordered_inputs;
        ordered_inputs.reserve(ordered_inputs_holder.size());
        for (auto &tsr : ordered_inputs_holder) {
            ordered_inputs.emplace_back(&tsr);
        }
        std::vector<graph::logical_tensor_t *> ordered_outputs;
        ordered_outputs.reserve(ordered_outputs_holder.size());
        for (auto &tsr : ordered_outputs_holder) {
            ordered_outputs.emplace_back(&tsr);
        }
        graph::status_t ret = cur_op_schema->shape_infer(
                &temp_node, ordered_inputs, ordered_outputs);
        if (ret != graph::status::success) return ret;

        auto set_new_lt = [](graph::logical_tensor_t lt,
                                  std::shared_ptr<value_t> val) {
            val->set_logical_tensor(lt);
            // if layout is any; let's respect it
            // if layout is strided; shape_infer will fill the stride
            // if layout is undef; convert it to strided and fill the stride
            if (lt.layout_type == graph::layout_type::undef) {
                // force set strided dense layout
                graph::dims shape(lt.dims, lt.dims + lt.ndims);
                graph::dims strides = utils::get_dense_strides(shape);
                val->set_strides(strides);
            }
        };

        for (size_t i = 0; i < cur_op->get_output_values().size(); ++i) {
            auto output_lt = *ordered_outputs[i];
            auto cur_val = cur_op->get_output_values()[i];
            set_new_lt(output_lt, cur_val);
            // only write back inferred info to outputs
            // shall not modify the layout type
            auto out_pos = std::find_if(outputs.begin(), outputs.end(),
                    [&](graph::logical_tensor_t *alt) -> bool {
                        return alt->id == ordered_outputs[i]->id;
                    });
            if (out_pos != outputs.end()) {
                auto cur_lt = cur_val->get_logical_tensor();
                // ensure layout type not modified
                if ((**out_pos).layout_type == graph::layout_type::any)
                    cur_lt.layout_type = (**out_pos).layout_type;
                **out_pos = cur_lt;
            }
        }
        // additionaly fill shapes for input values of the subgraph
        for (size_t i = 0; i < cur_op->get_input_values().size(); ++i) {
            if (!cur_op->get_input_values()[i]->has_producer()) {
                auto input_lt = *ordered_inputs[i];
                auto cur_val = cur_op->get_input_values()[i];
                set_new_lt(input_lt, cur_val);
            }
        }
        return graph::status::success;
    });
    return ret;
}

graph::status_t compiler_partition_impl_t::compile(
        graph::compiled_partition_t *compiled_partition,
        const std::vector<graph::logical_tensor_t> &inputs,
        const std::vector<graph::logical_tensor_t> &outputs,
        const graph::engine_t *aengine) const {
    try {
        std::cout << "Trying to compile using Elyzor..." << std::endl;
        graph::status_t res = status::success;
        // here we call infer_shape since logical tensor info
        // may be incomplete for the graph corresponding to the
        // partition
        std::vector<const graph::logical_tensor_t *> input_ref;
        std::vector<graph::logical_tensor_t *> output_ref;
        input_ref.reserve(inputs.size());
        output_ref.reserve(outputs.size());
        for (auto &t : inputs) {
            input_ref.push_back(const_cast<graph::logical_tensor_t *>(&t));
        }
        for (auto &t : outputs) {
            output_ref.push_back(const_cast<graph::logical_tensor_t *>(&t));
        }
        res = this->infer_shape(input_ref, output_ref);
        if (res != status::success) { return res; }

        COMPILE_ASSERT(aengine->kind() == graph::engine_kind_t::dnnl_cpu,
                "Graph compiler backend only supports cpu engine");

        // TODO: maybe cache ctx and gc in a static variable
        auto &cl = compiler_backend_t::get_singleton()
                           .get_graph_compiler_loader();

        const graph_compiler_context ctx {
                /*num_threads=*/std::thread::hardware_concurrency()};
        const graph_compiler *gc;
        WRAP_GC_CALL(cl.create_gc(&ctx, &gc),
                "Failed to create graph compiler object");

        std::stringstream json_stream;
        graph::graph_t(copied_ops_).serialize(json_stream);

        std::string json = json_stream.str();

        const graph_compiler_executable *exe;
        WRAP_GC_CALL(cl.compile(gc, json.data(), &exe),
                "Failed to compile partition");

        auto pimpl = std::make_shared<compiler_compiled_partition_impl_t>(
                *aengine, inputs, outputs,
                // inplace pairs are always empty in the original GC
                std::vector<graph::inplace_pair_t> {},
                /*executor=*/exe, /*graph_compiler=*/gc);
        compiled_partition->init(pimpl);
        return res;
    } catch (const std::exception &e) {
        VERROR(graph, elyzor, "%s", e.what());
        return graph::status::unimplemented;
    }
}

std::shared_ptr<graph::partition_impl_t>
compiler_partition_impl_t::clone() const {
    auto ret = std::make_shared<compiler_partition_impl_t>(
            get_engine_kind(), get_fpmath_mode(), get_kind(), get_name());
    ret->ops_ = graph::graph_t::deep_copy(ops_);
    ret->inputs_ = inputs_;
    ret->outputs_ = outputs_;
    ret->id_ = id_;
    ret->is_init_ = is_init_;
    return ret;
}

bool compiler_partition_impl_t::is_initialized() const {
    return is_init_;
}

compiler_compiled_partition_impl_t::compiler_compiled_partition_impl_t(
        const graph::engine_t &engine,
        const std::vector<graph::logical_tensor_t> &inputs,
        const std::vector<graph::logical_tensor_t> &outputs,
        const std::vector<graph::inplace_pair_t> &inplace_pairs,
        const graph_compiler_executable *exe, const graph_compiler *gc)
    : graph::compiled_partition_impl_t(engine, inputs, outputs, inplace_pairs)
    , exe_(exe)
    , gc_(gc) {}

compiler_compiled_partition_impl_t::~compiler_compiled_partition_impl_t() {
    // isn't it too dangerous to call this in a descructor??
    auto &cl = compiler_backend_t::get_singleton().get_graph_compiler_loader();

    cl.destroy_exe(gc_, exe_);
    cl.destroy_gc(gc_);
}

graph::status_t compiler_compiled_partition_impl_t::execute(
        const graph::stream_t *astream,
        const std::vector<graph::tensor_t> &inputs,
        const std::vector<graph::tensor_t> &outputs) {
    try {
        UNUSED(astream);
        std::vector<graph_compiler_tensor> args;
        args.reserve(inputs.size() + outputs.size());

        for (auto &in : inputs) {
            auto lt = in.get_logical_tensor();
            graph_compiler_tensor tmp {lt.id, static_cast<uint8_t>(lt.ndims),
                    {}, in.get_data_handle()};
            std::copy(lt.dims, lt.dims + lt.ndims, tmp.dims);
            args.push_back(tmp);
        }
        for (auto &out : outputs) {
            auto lt = out.get_logical_tensor();
            graph_compiler_tensor tmp {lt.id, static_cast<uint8_t>(lt.ndims),
                    {}, out.get_data_handle()};
            std::copy(lt.dims, lt.dims + lt.ndims, tmp.dims);
            args.push_back(tmp);
        }

        auto &cl = compiler_backend_t::get_singleton()
                           .get_graph_compiler_loader();
        WRAP_GC_CALL(
                cl.execute(gc_, exe_, args.data(), args.data() + inputs.size()),
                "Failed to execute partition");
        return status::success;
    } catch (const std::exception &e) {
        VERROR(graph, elyzor, "%s", e.what());
        return graph::status::runtime_error;
    }
}
} // namespace elyzor
} // namespace graph
} // namespace impl
} // namespace dnnl
