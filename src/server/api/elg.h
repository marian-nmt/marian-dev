// -*- mode: c++; indent-tabs-mode: nil; tab-width: 2 -*-
#pragma once
#include "../translation_service.h"
#include <crow.h>
#include "3rd_party/rapidjson/include/rapidjson/document.h"
#include "3rd_party/rapidjson/include/rapidjson/writer.h"
#include "3rd_party/rapidjson/include/rapidjson/stringbuffer.h"
#include "rapidjson_utils.h"
#include "elg/json_request_handler.h"
#include "elg/crow_request_handler.h"

#define USE_CROW_JSON 0

namespace marian {
namespace server {
namespace elg {

enum class api { v1 };

// template<class Service>
// Ptr<rapidjson::Value>
// translate_node_v1(Service& service, rapidjson::Value const& c){

// }







// template<class Service>
// Ptr<rapidjson::Document>
// translate_v1(Service& service, rapidjson::Value const& request){
//   auto response = std::make_shared<rapidjson::Document>();
//   auto& alloc = response->GetAllocator();
//   rapidjson::Document& D = *response;
//   D.SetObject();
//   // Copy metadata from request.
//   if (request.HasMember("metadata")){
//     D.AddMember("metadata", {}, alloc);
//     D["metadata"].CopyFrom(request["metadata"], alloc);
//   }
//   if (request.HasMember("request") &&
//       request["request"].HasMember("content")){
//     auto& c = request["request"]["content"];
//     if (c.IsArray()) {
//     }
//     else if (c.IsString()) {
//       std::string payload = request["request"]["content"].GetString();
//       std::string translation = service.translate(payload)->await();
//       auto& r = D.AddMember("response",{},alloc)["response"].SetObject();
//       r.AddMember("type", "texts", alloc);
//       rapidjson::Value x(rapidjson::kObjectType);
//       x.AddMember("text", {}, alloc)["text"]
//         .SetString(translation.c_str(), translation.size(), alloc);
//       r.AddMember("texts", {}, alloc)["texts"].SetArray().PushBack(x,alloc);
//       return response;
//     }
//   }
//   // error
//   auto& r = D.AddMember("failure",{},alloc)["failure"].SetObject();
//   auto& e = r.AddMember("errors",{},alloc)["errors"].SetArray();
//   e.PushBack("Invalid request format.",alloc);
//   return response;
// }



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
