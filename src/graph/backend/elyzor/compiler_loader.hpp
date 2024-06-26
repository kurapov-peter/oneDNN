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
#ifndef ELYZOR_GRAPH_COMPILER_LOADER_H
#define ELYZOR_GRAPH_COMPILER_LOADER_H

#include <iostream>
#include "common/verbose.hpp"
#include "include/dnnl_graph_compiler.h"

namespace dnnl {
namespace impl {
namespace graph {
namespace elyzor {

typedef const dnnl_graph_compiler_version *(*dnnl_graph_compiler_get_version_t)(
        void);

typedef dnnl_status_t (*dnnl_graph_compiler_create_t)(
        const struct dnnl_graph_compiler_context *ctx,
        const struct dnnl_graph_compiler **gc);

typedef void (*dnnl_graph_compiler_destroy_t)(
        const struct dnnl_graph_compiler *gc);

typedef dnnl_status_t (*dnnl_graph_compiler_compile_t)(
        const struct dnnl_graph_compiler *gc, const char *graph_json,
        const struct dnnl_graph_compiler_executable **exe);

typedef void (*dnnl_graph_compiler_destroy_executable_t)(
        const struct dnnl_graph_compiler_executable *exe);

typedef dnnl_status_t (*dnnl_graph_compiler_execute_t)(
        const struct dnnl_graph_compiler_executable *exe,
        dnnl_graph_compiler_tensor *inputs,
        dnnl_graph_compiler_tensor *outputs);

struct dnnl_graph_compiler_vtable {
    dnnl_graph_compiler_get_version_t dnnl_graph_compiler_get_version;
    dnnl_graph_compiler_create_t dnnl_graph_compiler_create;
    dnnl_graph_compiler_destroy_t dnnl_graph_compiler_destroy;
    dnnl_graph_compiler_compile_t dnnl_graph_compiler_compile;
    dnnl_graph_compiler_destroy_executable_t
            dnnl_graph_compiler_destroy_executable;
    dnnl_graph_compiler_execute_t dnnl_graph_compiler_execute;
};

class graph_compiler_loader {
public:
    static const dnnl_graph_compiler_vtable &get_vtable() {
        static graph_compiler_loader ins;
        return ins.vtable_;
    }
    ~graph_compiler_loader();
    static bool is_supported_api_version(
            const dnnl_graph_compiler_version::version &api_v);

private:
    graph_compiler_loader();
    graph_compiler_loader(graph_compiler_loader &) = delete;
    graph_compiler_loader(graph_compiler_loader &&) = delete;
    graph_compiler_loader &operator=(const graph_compiler_loader &) = delete;
    graph_compiler_loader &operator=(graph_compiler_loader &&) = delete;

    void *handle_;
    dnnl_graph_compiler_vtable vtable_;
    static const dnnl_graph_compiler_version::version supported_api_v_;
};

} // namespace elyzor
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
