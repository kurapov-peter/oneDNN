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
        size_t major;
        size_t minor;
        size_t patch;
        const char *hash;
    };
    // version of the gc API that was used to compile gc
    version api_version;
    // version of the graph compiler itself
    version gc_version;
};

struct dnnl_graph_compiler_context {
    uint32_t num_threads;

    void *(*allocator)(size_t size);

    void (*deallocator)(void *ptr);
};

struct dnnl_graph_compiler_tensor {
    size_t id;
    uint8_t ndims;
    size_t *dims;
    void *data;
};

DNNL_API const dnnl_graph_compiler_version *dnnl_graph_compiler_get_version(
        void);

DNNL_API dnnl_status_t dnnl_graph_compiler_create(
        const struct dnnl_graph_compiler_context *ctx,
        const struct dnnl_graph_compiler **gc);

DNNL_API void dnnl_graph_compiler_destroy(const struct dnnl_graph_compiler *gc);

DNNL_API dnnl_status_t dnnl_graph_compiler_compile(
        const struct dnnl_graph_compiler *gc, const char *graph_json,
        const struct dnnl_graph_compiler_executable **exe);

DNNL_API void dnnl_graph_compiler_destroy_executable(
        const struct dnnl_graph_compiler *gc,
        const struct dnnl_graph_compiler_executable *exe);

DNNL_API dnnl_status_t dnnl_graph_compiler_execute(
        const struct dnnl_graph_compiler *gc,
        const struct dnnl_graph_compiler_executable *exe,
        dnnl_graph_compiler_tensor *inputs,
        dnnl_graph_compiler_tensor *outputs);

#ifdef __cplusplus
}
#endif
#endif // DNNL_GRAPH_COMPILER_H
