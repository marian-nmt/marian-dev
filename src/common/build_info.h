#pragma once

#include <string>

namespace marian {

// Returns list of non-advanced cache variables used by CMake
std::string cmake_cache();

// Returns list of advanced cache variables used by CMake
std::string cmake_cache_advanced();

} // namespace marian
