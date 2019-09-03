// -*- mode: c++; indent-tabs-mode: nil; tab-width: 2 -*-
#pragma once
#include "../translation_service.h"
#include <crow.h>
#include "3rd_party/rapidjson/include/rapidjson/document.h"
#include "3rd_party/rapidjson/include/rapidjson/writer.h"
#include "3rd_party/rapidjson/include/rapidjson/stringbuffer.h"

#define USE_CROW_JSON 0

namespace marian {
namespace server {
namespace elg {

enum class api { v1 };

template<class Service>
Ptr<rapidjson::Document>
translate_v1(Service& service, rapidjson::Value const& request){
  auto response = std::make_shared<rapidjson::Document>();
  auto alloc = response->GetAllocator();
  // Copy metadata from request.
  // @TODO: Metadata should be a required field.
  if (request.HasMember("metadata")){
    response->AddMember("metadata").CopyFrom(request["metadata"]);
  }
  else{ // Metadata should be a required field.
    // @TODO: return error.
  }
  if (request.HasMember("request") &&
      request["request"].HasMember("content")){
    std::string payload = request["request"]["content"].GetString();
    std::string translation = service.translate(payload);
    auto& r = response->AddMember("response",rapidjson::kObjectType(),alloc);
    r.AddMember("type","texts",alloc);
    r.AddMember("texts",kArrayType(),alloc).PushBack(translation,alloc);
  }
  else{ // error
    auto& r = response->AddMember("failure",rapidjson::kObjectType(),alloc);
    r.AddMember("errors",rapidjson::kArrayType(),alloc)
      .PushBack("Invalid request format");
  }
  return response;
}


// template<class Service>
// crow::json::wvalue
// translate_v1(Service& service, crow::json::rvalue const& request)
// {
//   crow::json::wvalue response;
//   if (request.has("metadata")){
//     response["metadata"] = request["metadata"];
//   }
//   if (request.has("request") && request["request"].has("content")){
//     auto payload = request["request"]["content"].s();
//     std::string translation = service.translate(payload);
//     crow::json::wvalue& r = response["response"];
//     r["type"] = "texts";
//     r["texts"][0]["text"] = translation;
//   }
//   else{
//     crow::json::wvalue& r = response["failure"];
//     std::vector<std::string> e = {"Invalid request format."};
//     r["errors"] = e;
//   }
//   return std::move(response);
// }


}}} // end of namespace
