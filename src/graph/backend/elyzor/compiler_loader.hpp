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
#ifndef GRAPH_COMPILER_LOADER_H
#define GRAPH_COMPILER_LOADER_H

#include <dlfcn.h>
#include <iostream>
#include "common/verbose.hpp"
#include "graph_compiler.h"

namespace dnnl {
namespace impl {
namespace graph {
namespace elyzor {

typedef graph_compiler_status (*graph_compiler_create_func)(
        const struct graph_compiler_context *ctx,
        const struct graph_compiler **gc);

typedef void (*graph_compiler_destroy_func)(const struct graph_compiler *gc);

typedef graph_compiler_status (*graph_compiler_compile_func)(
        const struct graph_compiler *gc, const char *graph_json,
        const struct graph_compiler_executable **exe);

typedef void (*graph_compiler_destroy_executable_func)(
        const struct graph_compiler *gc,
        const struct graph_compiler_executable *exe);

typedef graph_compiler_status (*graph_compiler_execute_func)(
        const struct graph_compiler *gc,
        const struct graph_compiler_executable *exe,
        graph_compiler_tensor *inputs, graph_compiler_tensor *outputs);

typedef size_t (*graph_compiler_hints_func)(
        const struct graph_compiler_executable *exe,
        const struct graph_compiler_hint **hints);

struct graph_compiler_loader {
    graph_compiler_create_func create_gc;
    graph_compiler_destroy_func destroy_gc;
    graph_compiler_compile_func compile;
    graph_compiler_destroy_executable_func destroy_exe;
    graph_compiler_execute_func execute;

    graph_compiler_loader() {
        handle_ = dlopen("libgraph_compiler.so", RTLD_LAZY);
        if (!handle_) {
            std::stringstream ss;
            ss << "Failed to load library: " << dlerror();
            throw std::runtime_error(ss.str());
        }

        create_gc = (graph_compiler_create_func)dlsym(
                handle_, "graph_compiler_create");
        destroy_gc = (graph_compiler_destroy_func)dlsym(
                handle_, "graph_compiler_destroy");
        compile = (graph_compiler_compile_func)dlsym(
                handle_, "graph_compiler_compile");
        destroy_exe = (graph_compiler_destroy_executable_func)dlsym(
                handle_, "graph_compiler_destroy_executable");
        execute = (graph_compiler_execute_func)dlsym(
                handle_, "graph_compiler_execute");

        if (!create_gc || !destroy_gc || !compile || !destroy_exe || !execute) {
            dlclose(handle_);
            throw std::runtime_error("Failed to load one or more graph compiler's functions.");
        }
    }
    graph_compiler_loader(graph_compiler_loader &) = delete;
    graph_compiler_loader(graph_compiler_loader &&) = delete;
    graph_compiler_loader &operator=(const graph_compiler_loader &) = delete;
    graph_compiler_loader &operator=(graph_compiler_loader &&) = delete;

    ~graph_compiler_loader() {
        if (handle_) dlclose(handle_);
    }

private:
    void *handle_;
};

} // namespace elyzor
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
