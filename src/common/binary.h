#pragma once

#include "common/io_item.h"

#include <string>
#include <vector>

namespace marian {

const static uint64_t BINARY_FILE_VERSION = 1;
const static uint64_t BINARY_FILE_VERSION_WITH_ENCRYPTION = 2;

namespace io {
namespace binary {

void loadItems(const void* current,
               std::vector<io::Item>& items,
               bool mapped = false);
void loadItems(const std::string& fileName,
               std::vector<io::Item>& items);

io::Item getItem(const void* current, const std::string& vName);
io::Item getItem(const std::string& fileName, const std::string& vName);

void saveItems(const std::string& fileName,
               const std::vector<io::Item>& items);

}  // namespace binary
}  // namespace io
}  // namespace marian
