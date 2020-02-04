#include "rapidjson_utils.h"
namespace rapidjson {

  Value*
  ensure_path(Value& node, MemoryPoolAllocator<>& alloc, char const* key){
    if (!node.IsObject())
      return NULL;
    auto m = node.FindMember(key);
    if (m != node.MemberEnd())
      return &(m->value);
    return &(node.AddMember(StringRef(key),{},alloc)[key]);
  }

}
