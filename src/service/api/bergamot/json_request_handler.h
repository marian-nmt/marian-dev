#pragma once
#include "service/api/rapidjson_utils.h"
#include "node_translation.h"
// Do not include this file directly. It is included by ../json_request_handler.h
namespace marian{
namespace server{

template<class Service>
class BergamotJsonRequestHandlerV1
  : public JsonRequestHandlerBaseClass<Service>{
  std::string const payload_field_name_;
  std::string const options_field_name_;
public:
  BergamotJsonRequestHandlerV1(Service& service,
                               std::string const& payload_field="text",
                               std::string const& options_field="options")
    : JsonRequestHandlerBaseClass<Service>(service),
    payload_field_name_(payload_field),
    options_field_name_(options_field)
  {}

  Ptr<rapidjson::Document>
  operator()(std::string const& body,
             std::string const payload_field_name,
             std::string const options_field_name) const{
    Ptr<rapidjson::Document> D(new rapidjson::Document());
    D->Parse(body.c_str());
    if (!D->IsObject()) {
      return this->error("Invalid Json");
    }
    LOG(debug, "PARSED: {}", serialize(*D));
    NodeTranslation
      job(D.get(), this->service, payload_field_name, options_field_name);
    job.finish(D->GetAllocator());
    return D;
  }

  Ptr<rapidjson::Document>
  operator()(std::string const& body) const override {
    return (*this)(body, payload_field_name_, options_field_name_);
  }
};
}} // end of namespace marian::server::bergamot
