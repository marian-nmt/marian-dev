#include "catch.hpp"
#include "common/fast_opt.h"
#include "3rd_party/yaml-cpp/yaml.h"

using namespace marian;

TEST_CASE("FastOpt can be constructed from a YAML node", "[fastopt]") {
  YAML::Node node;
  const FastOpt o;

  SECTION("from a simple node") {
    YAML::Node node = YAML::Load("{foo: bar}");
    const_cast<FastOpt&>(o).reset(node);

    CHECK( o.has("foo") );
    CHECK_FALSE( o.has("bar") );
    CHECK_FALSE( o.has("baz") );
  }

  SECTION("from a sequence node") {
    YAML::Node node = YAML::Load("{foo: [bar, baz]}");
    const_cast<FastOpt&>(o).reset(node);
    CHECK( o.has("foo") );
  }

  SECTION("from nested nodes") {
    YAML::Node node = YAML::Load("{foo: {bar: 123, baz}}");
    const_cast<FastOpt&>(o).reset(node);
    CHECK( o.has("foo") );
    CHECK( o["foo"].has("bar") );
    CHECK( o["foo"].has("baz") );    
    CHECK( o["foo"]["bar"].as<int>() == 123 );
    CHECK( o["foo"]["baz"].type() == FastOpt::NodeType::Null );
  }
}

TEST_CASE("Options can be accessed", "[fastopt]") {
  YAML::Node node = YAML::Load("{"
      "foo: bar,"
      "seq: [1, 2, 3],"
      "subnode: {"
      "  baz: 111.5,"
      "  qux: 222,"
      "  }"
      "}");

  const FastOpt o(node);

  SECTION("using operator[]") {
    auto& oo = o["subnode"];
    CHECK( oo.has("baz") );
    CHECK( oo.has("qux") );
    CHECK_NOTHROW( o["subnode"]["baz"] );
  }

  SECTION("using as<T>()") {
    CHECK( o["foo"].as<std::string>() == "bar" );
    CHECK( o["subnode"]["baz"].as<int>() == 111 );
    CHECK( o["subnode"]["baz"].as<float>() == 111.5f );
  }

  SECTION("using as<std::vector<T>>()") {
    CHECK( o["seq"].as<std::vector<double>>() == std::vector<double>({1, 2, 3}) );
  }  
  
}
