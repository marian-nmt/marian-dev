#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "marian.h"
#include "functional/tensor.h"

namespace f = marian::functional;

int main(int argc, char** argv) {

  std::vector<float> v(2*2*5);
  std::iota(v.begin(), v.end(), 0);

  f::View<float, 3> out(v.data(), {{2,2,5}});

  std::cerr << out.debug() << std::endl;

  std::cerr << out[{0,0,3}] << std::endl;
  std::cerr << out[{0,1,3}] << std::endl;
  int ind = out.shape().index({0,1,3});

  std::cerr << out[{0,1,3}] << " == " << out[ind] << " ind: " << ind << std::endl;

  auto out2 = f::slice(out, {1}, {1}, {2,5});

  std::cerr << out2.shape() << std::endl;

  std::cerr << out2[{0,0,0}] << std::endl;
  std::cerr << out2[{0,0,1}] << std::endl;
  std::cerr << out2[{0,0,2}] << std::endl;

  // auto out4 = f::reshape(out3, {5});
  // auto out5 = f::broadcast(out4, {5,5,5});

  // std::cerr << out2[{1,0}] << std::endl;
  // std::cerr << out2[{1,1}] << std::endl;
  // std::cerr << out2[{1,2}] << std::endl;

  // auto out3 = f::slice(out2, {}, {1});
  // std::cerr << out3.shape() << std::endl;

  // std::cerr << out3[{0,0}] << std::endl;
  // std::cerr << out3[{1,0}] << std::endl;

  // std::cerr << out3[0] << std::endl;

  // std::cerr << out3[1] << std::endl;

  //auto out2 = view(out, { r(1) , r(2) , r(0, 2) , r(3, 5) });
  // shape: 1x1x2x2 , offsets: (1,2,0,3) stride: orig
  // out2[{0,0,1,0}] == 88





  // auto c = New<Config>(argc, argv);

  // auto type = c->get<size_t>("cpu-threads") > 0
  //   ? DeviceType::cpu
  //   : DeviceType::gpu;
  // DeviceId deviceId{0, type};

  // auto g = New<ExpressionGraph>();
  // g->setDevice(deviceId);
  // g->reserveWorkspaceMB(512);

  // for(int i = 0; i < 10; ++i) {
  //   g->clear();q
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
