#pragma once
#include "3rd_party/rapidjson/include/rapidjson/document.h"
namespace marian{
namespace server{

struct OutputOptions {
  bool withWordAlignment{false};
  bool withSoftAlignment{false};
  bool withTokenization{false};
  bool withSentenceScore{false};
  bool withWordScores{false};
  bool withOriginal{false};

  // Return true if all detail options are false:
  bool noDetails() const;
};

}} // end of namespace marian::server
