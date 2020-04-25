#pragma once

namespace marian {
namespace server {
struct TranslationOptions {
  size_t nbest{1};
  // size_t beam{0}; // use default ### NOT YET IMPLEMENTED!
};
}// end of namespace
}// end of namespace
