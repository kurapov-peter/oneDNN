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
#ifndef BACKEND_ELYZOR_UTILS_HPP
#define BACKEND_ELYZOR_UTILS_HPP

#include <functional>
#include <memory>
#include <numeric>
#include <vector>
#include <unordered_map>

#include "include/dnnl_graph_compiler.h"
#include <graph/interface/c_types_map.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace elyzor {
namespace utils {

// gcc4.8.5 can 't support enum class as key
struct enum_hash_t {
    template <typename T>
    size_t operator()(const T &t) const {
        return static_cast<size_t>(t);
    }
};

template <class F, class T>
inline auto func_map(const T &inputs, const F &fn)
        -> std::vector<decltype(fn(*inputs.begin()))> {
    std::vector<decltype(fn(*inputs.begin()))> r;
    r.reserve(inputs.size());
    for (const auto &input : inputs)
        r.push_back(fn(input));
    return r;
}

inline dims get_dense_strides(const dims &shape) {
    dims strides(shape.size());
    for (auto it = shape.begin(); it < shape.end(); ++it) {
        const auto val = std::accumulate(
                std::next(it), shape.end(), 1, std::multiplies<dim_t>());
        const auto dist = std::distance(shape.begin(), it);
        strides[static_cast<size_t>(dist)] = val;
    }
    return strides;
}

const std::unordered_map<op_kind_t, std::string, enum_hash_t> &
get_supported_ops();
std::vector<op_kind_t> get_supported_op_kinds();

#define COMPILE_ASSERT(cond, ...) \
    if (!(cond)) { \
        ::std::stringstream ss; \
        ss << __FILE__ << "[" << __LINE__ << "]: " << __VA_ARGS__ << "\n"; \
        throw ::std::runtime_error(ss.str()); \
    }

#define WRAP_GC_CALL(call, msg) \
    COMPILE_ASSERT(call == dnnl_status_t::dnnl_success, msg)

} // namespace utils
} // namespace elyzor
} // namespace graph
} // namespace impl
} // namespace dnnl

std::ostream &operator<<(
        std::ostream &os, const dnnl_graph_compiler_version &v);
std::ostream &operator<<(
        std::ostream &os, const dnnl_graph_compiler_version::version &v);

#endif
