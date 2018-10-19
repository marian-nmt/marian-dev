#ifndef VALUE_PTR_H_62B23520_7C8E_11DE_8A39_0800200C9A66
#define VALUE_PTR_H_62B23520_7C8E_11DE_8A39_0800200C9A66

#if defined(_MSC_VER) ||                                            \
    (defined(__GNUC__) && (__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || \
     (__GNUC__ >= 4))  // GCC supports "pragma once" correctly since 3.4
#pragma once
#endif

#include "yaml-cpp/dll.h"
#include "common/intrusive_ptr.h"

#include <memory>

namespace YAML {
namespace detail {
class node;
class node_ref;
class node_data;
class memory;
class memory_holder;

typedef IntrusivePtr<node> shared_node;
void intrusivePtrAddRef(node*);
void intrusivePtrRelease(node*);

typedef IntrusivePtr<node_ref> shared_node_ref;
void intrusivePtrAddRef(node_ref*);
void intrusivePtrRelease(node_ref*);

typedef IntrusivePtr<node_data> shared_node_data;
void intrusivePtrAddRef(node_data*);
void intrusivePtrRelease(node_data*);

typedef IntrusivePtr<memory_holder> shared_memory_holder;
void intrusivePtrAddRef(memory_holder*);
void intrusivePtrRelease(memory_holder*);

typedef IntrusivePtr<memory> shared_memory;
void intrusivePtrAddRef(memory*);
void intrusivePtrRelease(memory*);

}
}

#endif  // VALUE_PTR_H_62B23520_7C8E_11DE_8A39_0800200C9A66
