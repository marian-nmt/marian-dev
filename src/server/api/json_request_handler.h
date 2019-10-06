#pragma once
#include "../translation_service.h"
#include "rapidjson_utils.h"
#include "3rd_party/rapidjson/include/rapidjson/document.h"
#include "3rd_party/rapidjson/include/rapidjson/writer.h"
#include "3rd_party/rapidjson/include/rapidjson/stringbuffer.h"

namespace marian{
namespace server{

template<class Service>
class JsonRequestHandlerBaseClass{
protected:
  Service& service_;
public:
  JsonRequestHandlerBaseClass(Service& service)
    : service_(service){}

  virtual
  Ptr<rapidjson::Document>
  error(char const* msg) const{
    Ptr<rapidjson::Document> D(new rapidjson::Document());

    D->AddMember("error", {}, D->GetAllocator())
      .SetString(msg, strlen(msg), D->GetAllocator());
    return D;
  }

  virtual
  Ptr<rapidjson::Document>
  operator()(std::string const& body) const = 0;
};
}} // end of namespace marian::server

#include "elg/json_request_handler.h"
#include "bergamot/json_request_handler.h"
