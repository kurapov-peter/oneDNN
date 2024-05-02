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

class graph_compiler_loader {
public:
    graph_compiler_loader() : handle_(nullptr) {}
    graph_compiler_loader(graph_compiler_loader &) = delete;
    graph_compiler_loader(graph_compiler_loader &&) = delete;
    graph_compiler_loader &operator=(const graph_compiler_loader &) = delete;
    graph_compiler_loader &operator=(graph_compiler_loader &&) = delete;

    graph_compiler_status create_gc(const struct graph_compiler_context *ctx,
            const struct graph_compiler **gc) {
        maybe_load_module();
        return create_gc_(ctx, gc);
    }

    void destroy_gc(const struct graph_compiler *gc) {
        maybe_load_module();
        destroy_gc_(gc);
    }

    graph_compiler_status compile(const struct graph_compiler *gc,
            const char *graph_json,
            const struct graph_compiler_executable **exe) {
        maybe_load_module();
        return compile_(gc, graph_json, exe);
    }

    void destroy_exe(const struct graph_compiler *gc,
            const struct graph_compiler_executable *exe) {
        maybe_load_module();
        destroy_exe_(gc, exe);
    }

    graph_compiler_status execute(const struct graph_compiler *gc,
            const struct graph_compiler_executable *exe,
            graph_compiler_tensor *inputs, graph_compiler_tensor *outputs) {
        maybe_load_module();
        return execute_(gc, exe, inputs, outputs);
    }

    ~graph_compiler_loader() {
        if (handle_) dlclose(handle_);
    }

private:
    void maybe_load_module();

    template <typename func_ptr_type>
    func_ptr_type load_func(const char *func_name) {
        if (!handle_) {
            throw std::runtime_error("Can't load symbols from an invalid handle.");
        }
        func_ptr_type func = (func_ptr_type)dlsym(handle_, func_name);
        if (!func) {
            std::stringstream ss;
            ss << "Failed to load \'" << func_name << "\' function from "
               << libname;
            throw std::runtime_error(ss.str());
        }
        return func;
    }

    std::mutex mtx_;
    void *handle_;
    static const char *libname;
    graph_compiler_create_func create_gc_;
    graph_compiler_destroy_func destroy_gc_;
    graph_compiler_compile_func compile_;
    graph_compiler_destroy_executable_func destroy_exe_;
    graph_compiler_execute_func execute_;
};

} // namespace elyzor
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
