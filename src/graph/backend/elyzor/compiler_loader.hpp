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

#include <dlfcn.h>
#include <iostream>
#include "common/verbose.hpp"
#include "include/dnnl_graph_compiler.h"

namespace dnnl {
namespace impl {
namespace graph {
namespace elyzor {

typedef dnnl_status_t (*dnnl_graph_compiler_get_api_version_t)(
        dnnl_graph_compiler_api_version *v);

typedef dnnl_status_t (*dnnl_graph_compiler_create_t)(
        const struct dnnl_graph_compiler_context *ctx,
        const struct dnnl_graph_compiler **gc);

typedef void (*dnnl_graph_compiler_destroy_t)(
        const struct dnnl_graph_compiler *gc);

typedef dnnl_status_t (*dnnl_graph_compiler_compile_t)(
        const struct dnnl_graph_compiler *gc, const char *graph_json,
        const struct dnnl_graph_compiler_executable **exe);

typedef void (*dnnl_graph_compiler_destroy_executable_t)(
        const struct dnnl_graph_compiler *gc,
        const struct dnnl_graph_compiler_executable *exe);

typedef dnnl_status_t (*dnnl_graph_compiler_execute_t)(
        const struct dnnl_graph_compiler *gc,
        const struct dnnl_graph_compiler_executable *exe,
        dnnl_graph_compiler_tensor *inputs,
        dnnl_graph_compiler_tensor *outputs);

struct dnnl_graph_compiler_vtable {
    dnnl_graph_compiler_get_api_version_t dnnl_graph_compiler_get_api_version;
    dnnl_graph_compiler_create_t dnnl_graph_compiler_create;
    dnnl_graph_compiler_destroy_t dnnl_graph_compiler_destroy;
    dnnl_graph_compiler_compile_t dnnl_graph_compiler_compile;
    dnnl_graph_compiler_destroy_executable_t
            dnnl_graph_compiler_destroy_executable;
    dnnl_graph_compiler_execute_t dnnl_graph_compiler_execute;
};

class graph_compiler_loader {
public:
    graph_compiler_loader() : handle_(nullptr) {}
    graph_compiler_loader(graph_compiler_loader &) = delete;
    graph_compiler_loader(graph_compiler_loader &&) = delete;
    graph_compiler_loader &operator=(const graph_compiler_loader &) = delete;
    graph_compiler_loader &operator=(graph_compiler_loader &&) = delete;

    template <typename func_ptr_type>
    func_ptr_type load_func(const char *func_name) {
        if (!handle_) {
            throw std::runtime_error(
                    "Can't load symbols from an invalid handle.");
        }
        func_ptr_type func
                = reinterpret_cast<func_ptr_type>(dlsym(handle_, func_name));
        if (!func) {
            std::stringstream ss;
            ss << "Failed to load \'" << func_name << "\' function from "
               << libname;
            throw std::runtime_error(ss.str());
        }
        return func;
    }

    ~graph_compiler_loader() {
        if (handle_) dlclose(handle_);
    }

    const dnnl_graph_compiler_vtable &get_vtable() {
        maybe_load_module();
        return vtable_;
    }

    bool is_version_supported(dnnl_graph_compiler_api_version version) {
        return version.major == supported_version_.major
                && version.minor == supported_version_.minor
                && version.patch == supported_version_.patch;
    }

private:
    void maybe_load_module();

    std::mutex mtx_;
    void *handle_;
    static const char *libname;
    dnnl_graph_compiler_vtable vtable_;
    static const dnnl_graph_compiler_api_version supported_version_;
};

} // namespace elyzor
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
