#pragma once

namespace marian {

class BLASInitializer {
public:
  BLASInitializer();
};

// Declare a global instance
extern BLASInitializer blasInitializer;

}  // namespace marian
