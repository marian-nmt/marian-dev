#pragma once
// #include "3rd_party/rapidjson/include/rapidjson/rapidjson.h"
// #include "3rd_party/rapidjson/include/rapidjson/fwd.h"
#include "3rd_party/rapidjson/include/rapidjson/allocators.h"
#include "3rd_party/rapidjson/include/rapidjson/fwd.h"
#include "3rd_party/rapidjson/include/rapidjson/document.h"
#include "3rd_party/rapidjson/include/rapidjson/stringbuffer.h"
#include "3rd_party/rapidjson/include/rapidjson/writer.h"

#include "service/common/translation_service.h"
#include "service/common/translation_job.h"
#include "service/api/output_options.h"

// Utility functions for dealing with Rapidjson
namespace rapidjson {
  Value*
  ensure_path(Value& node, MemoryPoolAllocator<>& alloc, char const* key);

  template<typename ... Rest>
  Value*
  ensure_path(Value& node, MemoryPoolAllocator<>& alloc, char const* key,
              Rest ... restpath){
    if (!node.IsObject())
      return NULL;
    auto m = node.FindMember(key);
    if (m != node.MemberEnd())
      return ensure_path(m->value, alloc, restpath ...);
    Value k(key,alloc);
    auto& x = node.AddMember(k, Value().Move(), alloc)[key].SetObject();
    return ensure_path(x, alloc, restpath ...);
  }

  std::string get(const Value& D, const char* key, std::string const& dflt);
  int get(const Value& D, const char* key, int const& dflt);

  Value job2json(const marian::server::Job& job,
                 const marian::server::TranslationService& service,
                 const marian::server::OutputOptions& opts,
                 MemoryPoolAllocator<>& alloc);

  std::string serialize(Document const& D);

  // void dump(rapidjson::Value& v, std::ostream& out) {
  //   if (v.IsString()) { out << v.GetString() << std::endl; }
  //   else if (v.IsArray()) { for (auto& c: v.GetArray()) dump(c,out); }
  // }

  // Override values from specs in Json object v.
  // Return false if there's a problem with v (not a JSON object type)
  bool setOptions(marian::server::OutputOptions& opts, const rapidjson::Value& v);


} // end of namespace rapidjson
