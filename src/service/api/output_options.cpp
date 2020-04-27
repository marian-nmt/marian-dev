#include "common/logging.h"
#include "output_options.h"
#include "rapidjson_utils.h"
namespace marian {
namespace server {

bool OutputOptions::noDetails() const {
  // we don't need to check to withWordAlignment and withSoftAlignment, because
  // these imply withTokenization
  return !(withTokenization ||
           withSentenceScore ||
           withWordScores ||
           withWordAlignment ||
           withSoftAlignment ||
           withOriginal);
}

}}// end of namespace marian::server
