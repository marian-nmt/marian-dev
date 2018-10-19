#ifndef VALUE_PTR_H_62B23520_7C8E_11DE_8A39_0800200C9A66
#define VALUE_PTR_H_62B23520_7C8E_11DE_8A39_0800200C9A66

#if defined(_MSC_VER) ||                                            \
    (defined(__GNUC__) && (__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || \
     (__GNUC__ >= 4))  // GCC supports "pragma once" correctly since 3.4
#pragma once
#endif

#include "yaml-cpp/dll.h"

#include "common/icky_ptr.h"
#include "common/sticky_ptr.h"

#include <memory>

namespace YAML {
namespace detail {
class node;
class node_ref;
class node_data;
class memory;
class memory_holder;

typedef StickyPtr<node> shared_node;
void stickyPtrAddRef(node*);
void stickyPtrRelease(node*);

typedef StickyPtr<node_ref> shared_node_ref;
void stickyPtrAddRef(node_ref*);
void stickyPtrRelease(node_ref*);

typedef StickyPtr<node_data> shared_node_data;
void stickyPtrAddRef(node_data*);
void stickyPtrRelease(node_data*);

typedef StickyPtr<memory_holder> shared_memory_holder;
void stickyPtrAddRef(memory_holder*);
void stickyPtrRelease(memory_holder*);

typedef StickyPtr<memory> shared_memory;
void stickyPtrAddRef(memory*);
void stickyPtrRelease(memory*);

}
}

#endif  // VALUE_PTR_H_62B23520_7C8E_11DE_8A39_0800200C9A66
