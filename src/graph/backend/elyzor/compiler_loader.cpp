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
#include "compiler_loader.hpp"
#include "elyzor_backend.hpp"

#ifdef _WIN32
#include <windows.h>
#define DNNL_GC_LIB_NAME "GcDnnlApi.dll"
#define dlopen(libname, flags) LoadLibrary(libname)
#define dlsym(handle, funcname) GetProcAddress((HMODULE)handle, funcname)
#define dlclose(handle) FreeLibrary((HMODULE)handle)
#define dlerror() "an error occured when working with " #DNNL_GC_LIB_NAME
#elif __APPLE__
#include <dlfcn.h>
#define DNNL_GC_LIB_NAME "libGcDnnlApi.dylib"
#else
#include <dlfcn.h>
#define DNNL_GC_LIB_NAME "libGcDnnlApi.so"
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace elyzor {

template <typename func_ptr_type>
func_ptr_type load_func(void *handle, const char *func_name) {
    if (!handle) {
        throw std::runtime_error("Can't load symbols from an invalid handle.");
    }
    func_ptr_type func
            = reinterpret_cast<func_ptr_type>(dlsym(handle, func_name));
    if (!func) {
        std::stringstream ss;
        ss << "Failed to load \'" << func_name << "\' function.";
        throw std::runtime_error(ss.str());
    }
    return func;
}

const dnnl_graph_compiler_version::version
        graph_compiler_loader::supported_api_v_ {DNNL_GC_API_V_MAJOR,
                DNNL_GC_API_V_MINOR, DNNL_GC_API_V_PATCH, .hash = ""};

bool graph_compiler_loader::is_supported_api_version(
        const dnnl_graph_compiler_version::version &api_v) {
    return api_v.major == supported_api_v_.major
            && api_v.minor == supported_api_v_.minor
            && api_v.patch == supported_api_v_.patch;
}

graph_compiler_loader::graph_compiler_loader() {
    handle_ = dlopen(DNNL_GC_LIB_NAME, RTLD_LAZY | RTLD_DEEPBIND);
    if (!handle_) {
        std::stringstream ss;
        ss << "Failed to load library: " << dlerror();
        throw std::runtime_error(ss.str());
    }

    try {
        vtable_.dnnl_graph_compiler_get_version
                = load_func<dnnl_graph_compiler_get_version_t>(
                        handle_, "dnnl_graph_compiler_get_version");

        auto lib_v = vtable_.dnnl_graph_compiler_get_version();
        if (!is_supported_api_version(lib_v->api_version)) {
            std::stringstream ss;
            ss << "Unsupported GC API version found in " << DNNL_GC_LIB_NAME
               << "; Supported API version: " << supported_api_v_
               << "; Recieved: " << lib_v->api_version;
            throw std::runtime_error(ss.str());
        }

        vtable_.dnnl_graph_compiler_create
                = load_func<dnnl_graph_compiler_create_t>(
                        handle_, "dnnl_graph_compiler_create");
        vtable_.dnnl_graph_compiler_destroy
                = load_func<dnnl_graph_compiler_destroy_t>(
                        handle_, "dnnl_graph_compiler_destroy");
        vtable_.dnnl_graph_compiler_compile
                = load_func<dnnl_graph_compiler_compile_t>(
                        handle_, "dnnl_graph_compiler_compile");
        vtable_.dnnl_graph_compiler_destroy_executable
                = load_func<dnnl_graph_compiler_destroy_executable_t>(
                        handle_, "dnnl_graph_compiler_destroy_executable");
        vtable_.dnnl_graph_compiler_execute
                = load_func<dnnl_graph_compiler_execute_t>(
                        handle_, "dnnl_graph_compiler_execute");
    } catch (...) {
        dlclose(handle_);
        throw;
    }
}

graph_compiler_loader::~graph_compiler_loader() {
    if (handle_) dlclose(handle_);
}

} // namespace elyzor
} // namespace graph
} // namespace impl
} // namespace dnnl

#define LOAD_AND_CALL(fn_name, ...) \
    dnnl::impl::graph::elyzor::graph_compiler_loader::get_vtable().fn_name( \
            __VA_ARGS__);

const dnnl_graph_compiler_version *dnnl_graph_compiler_get_version(void) {
    return LOAD_AND_CALL(dnnl_graph_compiler_get_version);
}

dnnl_status_t dnnl_graph_compiler_create(
        const struct dnnl_graph_compiler_context *ctx,
        const struct dnnl_graph_compiler **gc) {
    return LOAD_AND_CALL(dnnl_graph_compiler_create, ctx, gc);
}

void dnnl_graph_compiler_destroy(const struct dnnl_graph_compiler *gc) {
    return LOAD_AND_CALL(dnnl_graph_compiler_destroy, gc);
}

dnnl_status_t dnnl_graph_compiler_compile(const struct dnnl_graph_compiler *gc,
        const char *graph_json,
        const struct dnnl_graph_compiler_executable **exe) {
    return LOAD_AND_CALL(dnnl_graph_compiler_compile, gc, graph_json, exe);
}

void dnnl_graph_compiler_destroy_executable(
        const struct dnnl_graph_compiler_executable *exe) {
    LOAD_AND_CALL(dnnl_graph_compiler_destroy_executable, exe);
}

dnnl_status_t dnnl_graph_compiler_execute(
        const struct dnnl_graph_compiler_executable *exe,
        dnnl_graph_compiler_tensor *inputs,
        dnnl_graph_compiler_tensor *outputs) {
    return LOAD_AND_CALL(dnnl_graph_compiler_execute, exe, inputs, outputs);
}
