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

namespace dnnl {
namespace impl {
namespace graph {
namespace elyzor {

const char *graph_compiler_loader::libname = "libgraph_compiler.so";

void graph_compiler_loader::maybe_load_module() {
    if (handle_) return;
    handle_ = dlopen(libname, RTLD_LAZY);
    if (!handle_) {
        std::stringstream ss;
        ss << "Failed to load library: " << dlerror();
        throw std::runtime_error(ss.str());
    }

    try {
        create_gc_ = load_func<graph_compiler_create_func>(
                "graph_compiler_create");
        destroy_gc_ = load_func<graph_compiler_destroy_func>(
                "graph_compiler_destroy");
        compile_ = load_func<graph_compiler_compile_func>(
                "graph_compiler_compile");
        destroy_exe_ = load_func<graph_compiler_destroy_executable_func>(
                "graph_compiler_destroy_executable");
        execute_ = load_func<graph_compiler_execute_func>(
                "graph_compiler_execute");
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