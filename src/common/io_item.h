#pragma once

#include "common/shape.h"
#include "common/types.h"

#include <string>

namespace marian {
namespace io {

struct Item {
  std::vector<char> bytes;
  const char* ptr{0};
  bool mapped{false};

  std::string name;
  Shape shape;
  Type type{Type::float32};

  const char* data() const {
    if(mapped)
      return ptr;
    else
      return bytes.data();
  }

  size_t size() const {
    if(mapped)
      return shape.elements() * sizeOf(type);
    else
      return bytes.size();
  }

  void append(const Item& other) {
    ABORT_IF(mapped, "Memory-mapped items cannot be appended");
    ABORT_IF(type != other.type, "Only item of same type can be appended");

    // abort if any of the shapes is not a flat array, i.e. the number of elements in the
    // last dimension has to correspond to the number of bytes.
    ABORT_IF(shape[-1] * sizeOf(type) != bytes.size(), "Only flat items can be appended");
    ABORT_IF(other.shape[-1] * sizeOf(type) != other.bytes.size(), "Only flat items can be appended");

    shape.set(-1, shape[-1] + other.shape[-1]);
    bytes.insert(bytes.end(), other.bytes.begin(), other.bytes.end());
  }
};

}  // namespace io
}  // namespace marian
