#ifndef DNNL_GRAPH_COMPILER_H
#define DNNL_GRAPH_COMPILER_H

#include <cstddef>
#include <cstdint>
#include "dnnl_types.h"
#include "dnnl_version.h"

/*
 * Public API for integration with third-party graph compilers.
 */

#ifdef __cplusplus
extern "C" {
#endif

// graph compiler's API version following semver
#define DNNL_GC_API_V_MAJOR 0
#define DNNL_GC_API_V_MINOR 1
#define DNNL_GC_API_V_PATCH 0
#ifdef DNNL_VERSION_HASH
#define DNNL_GC_API_V_HASH DNNL_VERSION_HASH
#else
#define DNNL_GC_API_V_HASH "N/A"
#endif

struct dnnl_graph_compiler;
struct dnnl_graph_compiler_executable;

struct dnnl_graph_compiler_version {
    struct version {
        uint8_t major;
        uint8_t minor;
        uint8_t patch;
        const char *hash;
    };
    // version of the gc API that was used to compile gc,
    // we only bump it if the API in this file changes
    version api_version;
    // version of the graph compiler itself, we only bump it if
    // the core graph compiler releases a new version (with new
    // compiling features, etc) that does not necessarily change
    // the API in this file
    version gc_version;
    // Why having two versions?
    // The versions are bumped independently, so we want to know both versions of the '.so':
    //     1. API version - to check whether it's compatible at all with the current oneDNN version.
    //     2. GC version - to know which features/compilation patterns are supported in order to
    //        dispatch backend's logic based on this information (register certain patterns or not)
};

struct dnnl_graph_compiler_context {
    uint32_t num_threads;
};

struct dnnl_graph_compiler_tensor {
    size_t id;
    uint8_t ndims;
    int64_t *dims;
    void *data;
};

const dnnl_graph_compiler_version *dnnl_graph_compiler_get_version(void);

dnnl_status_t dnnl_graph_compiler_create(
        const struct dnnl_graph_compiler_context *ctx,
        const struct dnnl_graph_compiler **gc);

void dnnl_graph_compiler_destroy(const struct dnnl_graph_compiler *gc);

dnnl_status_t dnnl_graph_compiler_compile(const struct dnnl_graph_compiler *gc,
        const char *graph_json,
        const struct dnnl_graph_compiler_executable **exe);

void dnnl_graph_compiler_destroy_executable(
        const struct dnnl_graph_compiler_executable *exe);

dnnl_status_t dnnl_graph_compiler_execute(
        const struct dnnl_graph_compiler_executable *exe,
        dnnl_graph_compiler_tensor *inputs,
        dnnl_graph_compiler_tensor *outputs);

#ifdef __cplusplus
}
#endif
#endif // DNNL_GRAPH_COMPILER_H
