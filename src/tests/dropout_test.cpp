#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <map>

#include "common/logging.h"
#include "common/const_map.h"


int main(int argc, char** argv) {

  std::map<std::string, std::map<std::string, float>> test;
  test["key1"] = {{"sub1", 1.2f}};
  test["key2"] = {{"sub1", -2.f}, {"sub2", 3.14f}};

  FastOpt opt(test);

  // FastOpt opt("test : [3, 5], test2: { test3 : lala }");

  // opt["test"][0].as<int>() // 3
  // opt["test"][1].as<int>() // 5

  // opt["test"].as<std::vector<int>>()

  // opt["test"]["test2"]["test3"].as<std::string>() // lala

  const auto& kNode = opt["key1"];
  if(kNode) {
    float test = kNode["sub1"].as<float>();
  }

  std::cerr << opt["key1"]["sub1"].as<float>() << std::endl;
  std::cerr << opt["key2"]["sub1"].as<float>() << std::endl;
  std::cerr << opt["key2"]["sub2"].as<float>() << std::endl;

  // for(let set : opt["valid-sets"]) {
  //   let srcPath = set[0].as<std::string>();
  //   let tgtPath = set[1].as<std::string>();
  // }

  // isLeaf(), isString() isInt() isFloat()

  // template <class K, class V>
  // Node(const std::map<K, V>& m) {
  //   std::vector<K> keys;
  //   std::vector<Node> values;
  //   for(const auto& it : m) {
  //     keys.push_back(it.first);
  //     values.push_back(Node(it.second));
  //   }
  //   return Node(keys, values);
  // }

  // std::map<std::string, std::vector<std::vector<float>>> test;
  // test["test"] = {{ 0.2, 0.3 }, { 0.4 }};

  // auto stdmap = FastOpt::fromYaml("{test : [0.5, 0.33], test2 : [6.6, -10]}").convert<std::map<std::string, std::vector<float>>>();
  // auto fopt = FastOpt::fromStd(stdmap);

  // stdmap["test2"][0] // 6.6
  // fopt["test2"][0] // 6.6

  // for(let it : opt.map()) {
  //   std::cerr << it.first.as<std::string>() << " " << it.second.isString() ? it.second.as<std::string()> : "-";
  // }

  // Expr x = graph->const({256, 256}, inits::zeros);
  // if(opt.has("encoder"))
  //   if(opt["encoder"].isMap())
  //     for(let layer : opt["encoder"])
  //       x = Dense(x, layer);

  // auto c = New<Config>(argc, argv);

  // auto type = c->get<size_t>("cpu-threads") > 0
  //   ? DeviceType::cpu
  //   : DeviceType::gpu;
  // DeviceId deviceId{0, type};

  // auto g = New<ExpressionGraph>();
  // g->setDevice(deviceId);
  // g->reserveWorkspaceMB(512);

  // for(int i = 0; i < 10; ++i) {
  //   g->clear();
  //   auto mask1 = g->dropout(0.2, {10, 3072});
  //   auto mask2 = g->dropout(0.3, {1, 3072});
  //   auto mask = mask1 + mask2;
  //   debug(mask1, "mask1");
  //   debug(mask2, "mask2");
  //   debug(mask, "mask");
  //   g->forward();
  // }

  return 0;
}
