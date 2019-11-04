#include "catch.hpp"
#include "common/fastopt.h"
#include "3rd_party/yaml-cpp/yaml.h"

using namespace marian;

TEST_CASE("FastOpt can be constructed from a YAML node", "[fastopt]") {
  YAML::Node node;
  
  SECTION("from a simple node") {
    YAML::Node node = YAML::Load("{foo: bar}");
    const FastOpt o(node);

    CHECK( o.has("foo") );
    CHECK_FALSE( o.has("bar") );
    CHECK_FALSE( o.has("baz") );
  }

  SECTION("from a sequence node") {
    YAML::Node node = YAML::Load("{foo: [bar, baz]}");
    const FastOpt o(node);
    CHECK( o.has("foo") );
  }

  SECTION("from nested nodes") {
    YAML::Node node = YAML::Load("{foo: {bar: 123, baz}}");
    const FastOpt o(node);
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
      "  baz: [ 111.5, False ],"
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
    CHECK( o["subnode"]["baz"][0].as<float>() == 111.5f );
    CHECK( o["subnode"]["baz"][1].as<bool>() == false );
    CHECK( o["subnode"]["baz"][0].as<int>() == 111 );
  }

  // for(auto k : o[subnode].keys())
  //   o[subnode][k].type()

  SECTION("using as<std::vector<T>>()") {
    CHECK( o["seq"].as<std::vector<double>>() == std::vector<double>({1, 2, 3}) );
  }  

  YAML::Node yml = YAML::Load("{"
    "filter: ["
      "dense : { dim-in: 512, dim-out: 2048 },"
      "relu,"
      "dense : { dim-in: 2048, dim-out: 512 }"
      "]"
    "}");

  const FastOpt graph(yml);
  std::cerr << yml << std::endl;
  std::cerr << yml["filter"][0].begin()->first.as<std::string>() << std::endl;
  
}
