#pragma once
#include "../../translation_service.h"
#include "../rapidjson_utils.h"
#include "3rd_party/rapidjson/include/rapidjson/document.h"
#include "3rd_party/rapidjson/include/rapidjson/writer.h"
#include "3rd_party/rapidjson/include/rapidjson/stringbuffer.h"

namespace marian{
namespace server{
namespace bergamot{

template<class Service>
class JsonRequestHandlerV1{
  Service& service_;
public:
  JsonRequestHandlerV1(Service& service)
    : service_(service) {}


  rapidjson::Document
  error(char const* msg) const{
    rapidjson::Document D;

    D.AddMember("error", {}, D.GetAllocator())
      .SetString(msg, strlen(msg), D.GetAllocator());
    return D;
  }

  rapidjson::Document
  operator()(std::string const& body,
             std::string const payload_field_name,
             std::string const options_field_name) const{
    rapidjson::Document D;
    D.Parse(body.c_str());
    if (!D.IsObject()) {
      return error("Invalid Json");
    }
    LOG(debug, "PARSED: {}", server::serialize(D));
    server::NodeTranslation<>
      job(&D, service_, payload_field_name, options_field_name);
    job.finish(D.GetAllocator());
    return D;
  }
};
}}} // end of namespace marian::server::bergamot
