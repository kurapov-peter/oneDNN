/*******************************************************************************
 * Copyright 2021-2023 Intel Corporation
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
#include "compiler_loader.hpp"
#include "elyzor_backend.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace elyzor {

const char *graph_compiler_loader::libname = "libgraph_compiler.so";

void graph_compiler_loader::maybe_load_module() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (handle_) return;
    handle_ = dlopen(libname, RTLD_LAZY | RTLD_DEEPBIND);
    if (!handle_) {
        std::stringstream ss;
        ss << "Failed to load library: " << dlerror();
        throw std::runtime_error(ss.str());
    }

    try {
        // check for the version
        vtable_.dnnl_graph_compiler_get_version
                = load_func<dnnl_graph_compiler_get_version_t>(
                        "dnnl_graph_compiler_get_version");

        dnnl_graph_compiler_version v;
        vtable_.dnnl_graph_compiler_get_version(&v);
        if (!is_version_supported(v)) {
            std::stringstream ss;
            ss << "Unsupported version of " << libname << " (" << v.major << "."
               << v.minor << "." << v.patch << ")";
            throw std::runtime_error(ss.str());
        }

        vtable_.dnnl_graph_compiler_create
                = load_func<dnnl_graph_compiler_create_t>(
                        "dnnl_graph_compiler_create");
        vtable_.dnnl_graph_compiler_destroy
                = load_func<dnnl_graph_compiler_destroy_t>(
                        "dnnl_graph_compiler_destroy");
        vtable_.dnnl_graph_compiler_compile
                = load_func<dnnl_graph_compiler_compile_t>(
                        "dnnl_graph_compiler_compile");
        vtable_.dnnl_graph_compiler_destroy_executable
                = load_func<dnnl_graph_compiler_destroy_executable_t>(
                        "dnnl_graph_compiler_destroy_executable");
        vtable_.dnnl_graph_compiler_execute
                = load_func<dnnl_graph_compiler_execute_t>(
                        "dnnl_graph_compiler_execute");
    } catch (const std::runtime_error &e) {
        dlclose(handle_);
        handle_ = nullptr;
        throw e;
    }
}

} // namespace elyzor
} // namespace graph
} // namespace impl
} // namespace dnnl

#define LOAD_AND_CALL(fn_name, ...) \
    dnnl::impl::graph::elyzor::elyzor_backend_t::get_singleton() \
            .get_graph_compiler_loader() \
            .get_vtable() \
            .fn_name(__VA_ARGS__);

DNNL_API dnnl_status_t dnnl_graph_compiler_get_version(
        dnnl_graph_compiler_version *v) {
    return LOAD_AND_CALL(dnnl_graph_compiler_get_version, v);
}

DNNL_API dnnl_status_t dnnl_graph_compiler_create(
        const struct dnnl_graph_compiler_context *ctx,
        const struct dnnl_graph_compiler **gc) {
    return LOAD_AND_CALL(dnnl_graph_compiler_create, ctx, gc);
}

DNNL_API void dnnl_graph_compiler_destroy(
        const struct dnnl_graph_compiler *gc) {
    return LOAD_AND_CALL(dnnl_graph_compiler_destroy, gc);
}

DNNL_API dnnl_status_t dnnl_graph_compiler_compile(
        const struct dnnl_graph_compiler *gc, const char *graph_json,
        const struct dnnl_graph_compiler_executable **exe) {
    return LOAD_AND_CALL(dnnl_graph_compiler_compile, gc, graph_json, exe);
}

DNNL_API void dnnl_graph_compiler_destroy_executable(
        const struct dnnl_graph_compiler *gc,
        const struct dnnl_graph_compiler_executable *exe) {
    LOAD_AND_CALL(dnnl_graph_compiler_destroy_executable, gc, exe);
}

DNNL_API dnnl_status_t dnnl_graph_compiler_execute(
        const struct dnnl_graph_compiler *gc,
        const struct dnnl_graph_compiler_executable *exe,
        dnnl_graph_compiler_tensor *inputs,
        dnnl_graph_compiler_tensor *outputs) {
    return LOAD_AND_CALL(dnnl_graph_compiler_execute, gc, exe, inputs, outputs);
}
