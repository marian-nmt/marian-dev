#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <fstream>

#include <map>

#include "common/logging.h"
#include "common/fast_opt.h"


int main(int argc, char** argv) {
  const FastOpt& opt1 = YAML::LoadFile("test.yml");

  std::cerr << opt1[1]["powers"][1]["name"].as<std::string>() << std::endl;

  // YAML::Node yaml = opt1;

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

/*
- name: Ogre
  position: [0, 5, 0]
  powers:
    - name: Club
      damage: 10
    - name: Fist
      damage: 8
- name: Dragon
  position: [1, 0, 10]
  powers:
    - name: Fire Breath
      damage: 25
    - name: Claws
      damage: 15
- name: Wizard
  position: [5, -3, 0]
  powers:
    - name: Acid Rain
      damage: 50
    - name: Staff
      damage: 3
*/
