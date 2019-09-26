#pragma once
#include "3rd_party/rapidjson/include/rapidjson/document.h"
// #include "3rd_party/rapidjson/include/rapidjson/writer.h"
#include "3rd_party/rapidjson/include/rapidjson/stringbuffer.h"

// Utility functions for dealing with Rapidjson
namespace rapidjson {
  Value*
  ensure_path(Value& node, MemoryPoolAllocator<>& alloc, char const* key);

  template<typename ... Rest>
  Value*
  ensure_path(Value& node, MemoryPoolAllocator<>& alloc, char const* key, Rest ... restpath){
    if (!node.IsObject())
      return NULL;
    auto m = node.FindMember(key);
    if (m != node.MemberEnd())
      return ensure_path(m->value, alloc, restpath ...);
    Value k(key,alloc);
    auto& x = node.AddMember(k, Value().Move(), alloc)[key].SetObject();
    return ensure_path(x, alloc, restpath ...);
  }

}
