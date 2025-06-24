# llama.cpp Technical Deep Dive for CS50 Students

## Core Architecture: Function Pointers as Virtual Tables

llama.cpp uses C function pointers to simulate object-oriented behavior. Think of it like CS50's sorting algorithms where you pass a comparison function - but scaled up to an entire system.

### Backend Interface Structure
```c
// From ggml-backend-impl.h
struct ggml_backend_i {
    const char * (*get_name)(ggml_backend_t backend);
    void (*free)(ggml_backend_t backend);
    void (*set_tensor_async)(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    void (*get_tensor_async)(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
    void (*synchronize)(ggml_backend_t backend);
    enum ggml_status (*graph_compute)(ggml_backend_t backend, struct ggml_cgraph * cgraph);
    // ... more function pointers
};
```

This is essentially a **vtable** (virtual function table) - each backend implements these functions differently:
- CUDA backend: calls CUDA functions
- CPU backend: calls optimized CPU functions  
- Your NPU backend: would call XDNA2 functions

### Backend Structure
```c
struct ggml_backend {
    struct ggml_backend_i iface;  // Function pointers (the "vtable")
    void * context;               // Backend-specific data (like CUDA context)
};
```

When you call `backend->iface.graph_compute(backend, graph)`, it's like calling different sorting algorithms based on a function pointer.

## Memory Management System

### Buffer Types and Memory Hierarchy
```c
struct ggml_backend_buffer_type_i {
    const char * (*get_name)(ggml_backend_buffer_type_t buft);
    ggml_backend_buffer_t (*alloc_buffer)(ggml_backend_buffer_type_t buft, size_t size);
    size_t (*get_alignment)(ggml_backend_buffer_type_t buft);  // Memory alignment requirements
    size_t (*get_max_size)(ggml_backend_buffer_type_t buft);   // Max allocation size
    bool (*is_host)(ggml_backend_buffer_type_t buft);          // Is this regular RAM?
};
```

**Memory Types:**
- **Host Memory**: Regular RAM (malloc/free territory)
- **Device Memory**: GPU VRAM, NPU memory, etc.
- **Unified Memory**: Shared between CPU/GPU (like Apple Silicon)

### Buffer Interface
```c
struct ggml_backend_buffer_i {
    void (*free_buffer)(ggml_backend_buffer_t buffer);
    void * (*get_base)(ggml_backend_buffer_t buffer);          // Get raw memory pointer
    void (*set_tensor)(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    void (*get_tensor)(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
    bool (*cpy_tensor)(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst);
    void (*clear)(ggml_backend_buffer_t buffer, uint8_t value);
};
```

Think of this as custom `malloc/free` with additional operations for different memory types.

## Tensor System

### Tensor Structure (Simplified)
```c
struct ggml_tensor {
    enum ggml_type type;      // Data type (float32, int8, etc.)
    
    int64_t ne[GGML_MAX_DIMS]; // Number of elements in each dimension
    size_t  nb[GGML_MAX_DIMS]; // Number of bytes (stride) for each dimension
    
    enum ggml_op op;          // Operation type (ADD, MUL, MATMUL, etc.)
    
    void * data;              // Pointer to actual data
    
    struct ggml_tensor * src[GGML_MAX_SRC]; // Input tensors
    
    // Backend-specific data
    void * buffer;
    size_t offset;
};
```

**Key Concept**: Tensors are n-dimensional arrays with metadata. Like CS50's 2D arrays, but generalized:
- `ne[]`: dimensions (like rows/cols)  
- `nb[]`: byte stride (how to navigate memory)
- `data`: actual memory location

## Compute Graph System

### Graph Structure
```c
struct ggml_cgraph {
    int size;                    // Max number of nodes
    int n_nodes;                 // Current number of nodes
    int n_leafs;                 // Number of input nodes
    
    struct ggml_tensor ** nodes; // Array of tensor pointers
    struct ggml_tensor ** leafs; // Input tensors
    
    struct ggml_hash_set visited_hash_set; // For cycle detection
    
    // Backend-specific optimizations
    enum ggml_cgraph_eval_order order;
};
```

**Execution Flow:**
1. Build compute graph (like building a dependency tree)
2. Optimize graph for target backend
3. Execute operations in topological order
4. Handle memory transfers between different backends

## Backend Registration System

### How Backends are Discovered
```c
// From ggml-backend-reg.cpp
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL  
#include "ggml-metal.h"
#endif

// Your future NPU backend:
#ifdef GGML_USE_XDNA2
#include "ggml-xdna2.h"
#endif
```

Backends register themselves at startup. Each backend provides:
```c
// Example registration function
ggml_backend_reg_t ggml_backend_cuda_reg(void) {
    static ggml_backend_reg registration = {
        .get_name = ggml_backend_cuda_reg_get_name,
        .init     = ggml_backend_cuda_reg_init,
        .get_device_count = ggml_backend_cuda_reg_get_device_count,
        .get_device_description = ggml_backend_cuda_reg_get_device_description,
        // ... other callbacks
    };
    return &registration;
}
```

## Operation Dispatch System

### How Operations are Executed
When you call `ggml_backend_graph_compute(backend, graph)`:

1. **Graph Analysis**: Walk through compute graph nodes
2. **Operation Dispatch**: For each tensor operation:
   ```c
   switch (tensor->op) {
       case GGML_OP_ADD:
           return backend->iface.add(backend, tensor);
       case GGML_OP_MUL:  
           return backend->iface.mul(backend, tensor);
       case GGML_OP_MATMUL:
           return backend->iface.matmul(backend, tensor);
       // ... hundreds of operations
   }
   ```
3. **Memory Management**: Handle data transfers between backends
4. **Synchronization**: Wait for async operations to complete

### CUDA Backend Example
```c
// Simplified CUDA matrix multiplication
static enum ggml_status ggml_backend_cuda_matmul(ggml_backend_t backend, struct ggml_tensor * dst) {
    // Get CUDA context
    ggml_backend_cuda_context_t * cuda_ctx = (ggml_backend_cuda_context_t *)backend->context;
    
    // Get tensor data pointers
    const void * src0_data = dst->src[0]->data;
    const void * src1_data = dst->src[1]->data;
    void * dst_data = dst->data;
    
    // Launch CUDA kernel
    ggml_cuda_matmul_kernel<<<grid, block, shared_mem, cuda_ctx->stream>>>(
        src0_data, src1_data, dst_data, 
        /* dimensions and other params */
    );
    
    return GGML_STATUS_SUCCESS;
}
```

## Quantization System

### Data Types
```c
enum ggml_type {
    GGML_TYPE_F32  = 0,  // 32-bit float
    GGML_TYPE_F16  = 1,  // 16-bit float  
    GGML_TYPE_Q4_0 = 2,  // 4-bit quantized (block of 32)
    GGML_TYPE_Q4_1 = 3,  // 4-bit quantized (different encoding)
    GGML_TYPE_Q5_0 = 6,  // 5-bit quantized
    GGML_TYPE_Q8_0 = 8,  // 8-bit quantized
    // ... many more
};
```

**Quantization**: Compress model weights from 32-bit floats to lower precision (4-bit, 8-bit) to save memory and increase speed.

**Block-based Quantization**: Instead of quantizing individual values, quantize blocks of 32 values together with shared scaling factors.

---

# Adding AMD XDNA2 NPU Backend

## Step 1: Backend Structure Setup

### Create Header File: `ggml/include/ggml-xdna2.h`
```c
#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// Initialize XDNA2 backend
GGML_API ggml_backend_t ggml_backend_xdna2_init(int device);

// Check if XDNA2 is available
GGML_API bool ggml_backend_is_xdna2(ggml_backend_t backend);

// Get device count
GGML_API int ggml_backend_xdna2_get_device_count(void);

// Buffer type for XDNA2 memory
GGML_API ggml_backend_buffer_type_t ggml_backend_xdna2_buffer_type(int device);

// Host buffer type (for CPU memory that XDNA2 can access)
GGML_API ggml_backend_buffer_type_t ggml_backend_xdna2_host_buffer_type(void);

#ifdef  __cplusplus
}
#endif
```

### Create Implementation: `ggml/src/ggml-xdna2/ggml-xdna2.cpp`
```c
#include "ggml-xdna2.h"
#include "ggml-backend-impl.h"
#include <xrt/xrt.h>  // AMD XRT library headers

// Backend context - holds XDNA2 device state
struct ggml_backend_xdna2_context {
    int device_id;
    xrtDeviceHandle device;
    xrtContextHandle context;  
    // ... other XDNA2-specific state
};

// Forward declarations
static const char * ggml_backend_xdna2_name(ggml_backend_t backend);
static void ggml_backend_xdna2_free(ggml_backend_t backend);
static enum ggml_status ggml_backend_xdna2_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph);

// Backend interface implementation  
static struct ggml_backend_i xdna2_backend_i = {
    /* .get_name                = */ ggml_backend_xdna2_name,
    /* .free                    = */ ggml_backend_xdna2_free,
    /* .get_default_buffer_type = */ ggml_backend_xdna2_buffer_type,
    /* .set_tensor_async        = */ NULL, // Optional
    /* .get_tensor_async        = */ NULL, // Optional  
    /* .cpy_tensor_async        = */ NULL, // Optional
    /* .synchronize             = */ ggml_backend_xdna2_synchronize,
    /* .graph_plan_create       = */ NULL, // Optional
    /* .graph_plan_free         = */ NULL, // Optional
    /* .graph_plan_compute      = */ NULL, // Optional
    /* .graph_compute           = */ ggml_backend_xdna2_graph_compute,
    /* .supports_op             = */ ggml_backend_xdna2_supports_op,
    /* .supports_buft           = */ ggml_backend_xdna2_supports_buft,
    /* .offload_op              = */ NULL, // Optional
    /* .event_record            = */ NULL, // Optional
    /* .event_wait              = */ NULL, // Optional
};

// Implementation functions
static const char * ggml_backend_xdna2_name(ggml_backend_t backend) {
    return "XDNA2";
}

static void ggml_backend_xdna2_free(ggml_backend_t backend) {
    ggml_backend_xdna2_context_t * ctx = (ggml_backend_xdna2_context_t *)backend->context;
    
    // Cleanup XDNA2 resources
    xrtContextClose(ctx->context);
    xrtDeviceClose(ctx->device);
    
    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_xdna2_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_xdna2_context_t * ctx = (ggml_backend_xdna2_context_t *)backend->context;
    
    // Main compute loop
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        
        switch (node->op) {
            case GGML_OP_ADD:
                ggml_xdna2_add(ctx, node);
                break;
            case GGML_OP_MUL:  
                ggml_xdna2_mul(ctx, node);
                break;
            case GGML_OP_MATMUL:
                ggml_xdna2_matmul(ctx, node);
                break;
            default:
                // Fallback to CPU for unsupported operations
                return GGML_STATUS_FAILED;
        }
    }
    
    return GGML_STATUS_SUCCESS;
}

// Public API implementation
ggml_backend_t ggml_backend_xdna2_init(int device) {
    // Initialize XDNA2 device
    ggml_backend_xdna2_context_t * ctx = new ggml_backend_xdna2_context_t;
    ctx->device_id = device;
    
    // Open XDNA2 device using XRT
    ctx->device = xrtDeviceOpen(device);
    if (!ctx->device) {
        delete ctx;
        return nullptr;
    }
    
    ctx->context = xrtContextOpen(ctx->device);
    if (!ctx->context) {
        xrtDeviceClose(ctx->device);
        delete ctx;
        return nullptr;
    }
    
    ggml_backend_t backend = new ggml_backend {
        /* .iface   = */ xdna2_backend_i,
        /* .context = */ ctx
    };
    
    return backend;
}

int ggml_backend_xdna2_get_device_count(void) {
    // Query number of XDNA2 devices
    return xrtProbe();  // XRT function to count devices
}
```

## Step 2: Buffer Management

### XDNA2 Buffer Implementation
```c
// Buffer type for XDNA2 device memory
struct ggml_backend_xdna2_buffer_context {
    int device;
    xrtDeviceHandle device_handle;
    xrtBufferHandle buffer_handle;
    void * ptr;  // Host-accessible pointer (if supported)
    size_t size;
};

static void ggml_backend_xdna2_buffer_free(ggml_backend_buffer_t buffer) {
    ggml_backend_xdna2_buffer_context_t * ctx = (ggml_backend_xdna2_buffer_context_t *)buffer->context;
    
    xrtBufferFree(ctx->buffer_handle);
    delete ctx;
}

static void * ggml_backend_xdna2_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_xdna2_buffer_context_t * ctx = (ggml_backend_xdna2_buffer_context_t *)buffer->context;
    return ctx->ptr;
}

static void ggml_backend_xdna2_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_xdna2_buffer_context_t * ctx = (ggml_backend_xdna2_buffer_context_t *)buffer->context;
    
    // Copy from host to XDNA2 device memory
    xrtBufferWrite(ctx->buffer_handle, data, size, offset);
}

static void ggml_backend_xdna2_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_xdna2_buffer_context_t * ctx = (ggml_backend_xdna2_buffer_context_t *)buffer->context;
    
    // Copy from XDNA2 device memory to host
    xrtBufferRead(ctx->buffer_handle, data, size, offset);
}

static struct ggml_backend_buffer_i xdna2_backend_buffer_i = {
    /* .free_buffer  = */ ggml_backend_xdna2_buffer_free,
    /* .get_base     = */ ggml_backend_xdna2_buffer_get_base,
    /* .init_tensor  = */ NULL, // Optional
    /* .memset_tensor= */ NULL, // Optional  
    /* .set_tensor   = */ ggml_backend_xdna2_buffer_set_tensor,
    /* .get_tensor   = */ ggml_backend_xdna2_buffer_get_tensor,
    /* .cpy_tensor   = */ NULL, // Optional
    /* .clear        = */ ggml_backend_xdna2_buffer_clear,
    /* .reset        = */ NULL, // Optional
};
```

## Step 3: Operation Implementation

### Matrix Multiplication Example
```c
static void ggml_xdna2_matmul(ggml_backend_xdna2_context_t * ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // First matrix
    const struct ggml_tensor * src1 = dst->src[1];  // Second matrix
    
    // Get matrix dimensions
    const int64_t m = src0->ne[1];  // rows of src0
    const int64_t n = src1->ne[1];  // cols of src1  
    const int64_t k = src0->ne[0];  // cols of src0 = rows of src1
    
    // Load kernel for matrix multiplication
    xrtKernelHandle matmul_kernel = xrtPLLoadKernel(ctx->context, "matmul_kernel");
    
    // Set kernel arguments
    xrtKernelSetArg(matmul_kernel, 0, src0->data);  // Matrix A
    xrtKernelSetArg(matmul_kernel, 1, src1->data);  // Matrix B  
    xrtKernelSetArg(matmul_kernel, 2, dst->data);   // Result matrix C
    xrtKernelSetArg(matmul_kernel, 3, sizeof(int64_t), &m);
    xrtKernelSetArg(matmul_kernel, 4, sizeof(int64_t), &n);
    xrtKernelSetArg(matmul_kernel, 5, sizeof(int64_t), &k);
    
    // Execute kernel
    xrtRunHandle run = xrtKernelRun(matmul_kernel);
    xrtRunWait(run);  // Wait for completion
    
    xrtRunClose(run);
    xrtKernelClose(matmul_kernel);
}
```

## Step 4: Registration and Build Integration

### Add to Backend Registry: Modify `ggml-backend-reg.cpp`
```cpp
#ifdef GGML_USE_XDNA2
#include "ggml-xdna2.h"
#endif

// In backend registration function:
#ifdef GGML_USE_XDNA2
    if (ggml_backend_xdna2_get_device_count() > 0) {
        register_backend(ggml_backend_xdna2_reg());
    }
#endif
```

### CMake Integration: Add to `ggml/src/CMakeLists.txt`
```cmake
# XDNA2 backend
option(GGML_XDNA2 "Enable XDNA2 backend" OFF)

if (GGML_XDNA2)
    find_package(XRT REQUIRED)
    
    add_subdirectory(ggml-xdna2)
    
    target_compile_definitions(ggml PRIVATE GGML_USE_XDNA2)
    target_link_libraries(ggml PRIVATE ggml-xdna2)
endif()
```

### Create `ggml/src/ggml-xdna2/CMakeLists.txt`
```cmake
find_package(XRT REQUIRED)

add_library(ggml-xdna2
    ggml-xdna2.cpp
    # Add operation implementation files
    xdna2-matmul.cpp
    xdna2-add.cpp
    # ... other operations
)

target_include_directories(ggml-xdna2 PRIVATE 
    ${XRT_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(ggml-xdna2 PRIVATE ${XRT_LIBRARIES})
target_compile_definitions(ggml-xdna2 PRIVATE GGML_USE_XDNA2)

set_target_properties(ggml-xdna2 PROPERTIES POSITION_INDEPENDENT_CODE ON)
```

## Key Implementation Challenges

### 1. **Memory Management**
- XDNA2 may have different memory types (DDR, HBM, etc.)
- Handle memory alignment requirements  
- Implement efficient host-device transfers

### 2. **Operation Mapping**
- Map GGML operations to XDNA2 kernel calls
- Handle quantized data types (Q4_0, Q8_0, etc.)
- Implement fallback to CPU for unsupported ops

### 3. **Performance Optimization**
- Kernel fusion (combine multiple operations)
- Memory access pattern optimization
- Asynchronous execution with proper synchronization

### 4. **Error Handling**
- XRT error codes to GGML status conversion
- Graceful fallback mechanisms
- Memory leak prevention

This gives you the complete technical foundation for implementing an XDNA2 backend. The key is understanding that llama.cpp uses C-style "objects" with function pointers for abstraction - exactly the kind of system design that builds on CS50's C fundamentals.