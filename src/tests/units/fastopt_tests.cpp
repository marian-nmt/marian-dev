#include "catch.hpp"
#include "common/fast_opt.h"
#include "3rd_party/yaml-cpp/yaml.h"

using namespace marian;

TEST_CASE("FastOpt can be constructed from a YAML node", "[fastopt]") {
  YAML::Node node;

  SECTION("from a simple node") {
    YAML::Node node = YAML::Load("{foo: bar}");
    FastOpt o(node);
    CHECK( o.has("foo") );
    CHECK_FALSE( o.has("bar") );
    CHECK_FALSE( o.has("baz") );
  }

  SECTION("from a sequence node") {
    YAML::Node node = YAML::Load("{foo: [bar, baz]}");
    FastOpt o(node);
    CHECK( o.has("foo") );
  }

  SECTION("from nested nodes") {
    YAML::Node node = YAML::Load("{foo: {bar: 123, baz}}");
    FastOpt o(node);
    CHECK( o.has("foo") );
  }
}

TEST_CASE("Options can be accessed", "[fastopt]") {
  YAML::Node node = YAML::Load("{"
      "foo: bar,"
      "seq: [1, 2, 3],"
      "subnode: {"
      "  baz: 111,"
      "  qux: 222,"
      "  }"
      "}");

  FastOpt o(node);

  SECTION("using operator[]") {
    auto& oo = o["subnode"];
    CHECK( oo.has("baz") );
    CHECK( oo.has("qux") );
    CHECK_NOTHROW( o["subnode"]["baz"] );
  }

  SECTION("using as<T>()") {
    CHECK( o["foo"].as<std::string>() == "bar" );
    //CHECK( o["subnode"]["baz"].as<int>() == 111 );
  }

  //SECTION("using as<std::vector<T>>()") {
    //CHECK( o["seq"].as<std::vector<int>>() == std::vector<int>({1, 2, 3}) );
  //}
}
