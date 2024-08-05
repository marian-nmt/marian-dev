/* All or part of this file was contributed by NVIDIA under license:
 *   Copyright (C) 2020 NVIDIA Corporation
 *   SPDX-License-Identifier: MIT
 */
#include "catch.hpp"
#include "graph/expression_graph.h"
#include "graph/expression_operators.h"
#include "layers_new/transformer.h"

#ifdef CUDA_FOUND
#include "tensors/gpu/backend.h"
#endif

#include <cmath>

using namespace marian;

template <typename T>
void tests(DeviceType device, Type floatType = Type::float32) {

// Checking for FP16 support and skipping if not supported.
#ifdef CUDA_FOUND
  if(device == DeviceType::gpu && floatType == Type::float16) {
    auto gpuBackend = New<gpu::Backend>(DeviceId({0, device}), /*seed=*/1234);
    auto cudaCompute = gpuBackend->getCudaComputeCapability();
    if(cudaCompute.major < 6) return;
  }
#endif

  auto floatApprox  = [](T x, T y) -> bool { return x == Approx(y).margin(0.001f); };
  auto floatApprox2 = [](T x, T y) -> bool { return x == Approx(y).margin(0.01f); };
  auto floatEqual   = [](T x, T y) -> bool { return x == y; };

  Config::seed = 4321;
  auto graph = New<ExpressionGraph>();
  
  graph->setInference(true);
  graph->setDefaultElementType(floatType);
  graph->setDevice({0, device});
  graph->reserveWorkspaceMB(16);

  std::vector<T> values;

  SECTION("Test equivalence of layers and specialized operators") {
    graph->clear();
    values.clear();

    std::vector<T> vecState = {
      0.82858741, 0.97615969, 0.67942131, 0.17952891,
      0.65630823, 0.38350773, 0.74830967, 0.67770803,
      0.00955211, 0.02345274, 0.02023151, 0.97143453,
      0.89971799, 0.50413132, 0.62781775, 0.59496081,
      0.14006306, 0.46450409, 0.91360050, 0.10497642,
      0.25477138, 0.63996094, 0.53658444, 0.88240266,
      0.37349635, 0.38880551, 0.18208119, 0.62951839,
      0.04330675, 0.59304160, 0.20436798, 0.74339235,
      0.32903627, 0.81596214, 0.44163024, 0.92444748,
      0.80231488, 0.52994978, 0.13350771, 0.40195912,
      0.55303711, 0.55137914, 0.98701674, 0.54963994,
      0.45657760, 0.57295781, 0.58645976, 0.74960953,
      0.77174628, 0.06652048, 0.68104792, 0.84806365,
      0.75292617, 0.82063907, 0.96599948, 0.63845992,
      0.47047511, 0.48726216, 0.95756608, 0.01479877,
      0.75449765, 0.55964196, 0.66664016, 0.34928808
    };

    auto state = graph->constant({2, 2, 4, 4}, inits::fromVector(vecState));

    using namespace marian::nn;

    auto rnn = New<RNN<SSRU>>(graph, state->shape()[-1], /*transformer-rnn-projection*/true);
    auto output = rnn->apply(state);

    auto iProj = rnn->cell->iProj->weight;
    auto iBias = rnn->cell->iProj->bias;

    auto fProj = rnn->cell->fProj->weight;
    auto fBias = rnn->cell->fProj->bias;

    auto oProj = rnn->oProj->weight;
    auto oBias = rnn->oProj->bias;
    
#if 0
    debug(output, "output");

    auto x = affine(state, iProj, iBias);
    auto f = affine(state, fProj, fBias);

    auto ssruFwd = [=](Expr out, const std::vector<Expr>& inputs) {
      auto x = inputs[0];
      auto f = inputs[1];
      
      SSRUScanForward(out->val(), x->val(), f->val());
    };

    auto output2 = lambda({x, f}, x->shape(), x->value_type(), ssruFwd);
    
    output2 = relu(output2);
    output2 = affine(output, oProj, oBias);
    debug(output2, "output2");
#endif 

    graph->forward();

    std::vector<T> expected = {
      -0.23135981,  0.04476057,  0.16183880, -0.13936377,
      -0.47255400, -0.00786887,  0.10853745, -0.06822529,
      -0.51970947, -0.10289559, -0.06798580,  0.10712720,
      -0.58211476, -0.10762983, -0.06099827,  0.10525966,
      -0.33873928,  0.07430670,  0.24815071, -0.21479189,
      -0.50458324, -0.01065392,  0.11723585, -0.07428676,
      -0.47146145, -0.07140756, -0.01806587,  0.05478236,
      -0.49719882, -0.10403568, -0.07004700,  0.10721481,
      -0.31213918, -0.07793316, -0.06812444,  0.09076738,
      -0.26403564, -0.08575443, -0.10109652,  0.11913717,
      -0.57269764, -0.03178894,  0.08730030, -0.03967147,
      -0.63041478, -0.07102037,  0.02447471,  0.02596882,
      -0.40184090, -0.07519485, -0.04389046,  0.07439522,
      -0.62908661, -0.03906321,  0.08765715, -0.03556710,
      -0.54157418,  0.06784889,  0.27720353, -0.22676750,
      -0.50410551,  0.02381870,  0.17982434, -0.13504542
    };

    output->val()->get(values);

    CHECK(values.size() == expected.size());
    // CHECK(std::equal(values.begin(), values.end(), expected.begin(), floatApprox));
  }
}

#ifdef CUDA_FOUND
TEST_CASE("Expression graph supports basic math operations (gpu)", "[operator]") {
  tests<float>(DeviceType::gpu);
}

#if COMPILE_FP16
TEST_CASE("Expression graph supports basic math operations (gpu fp16)", "[operator]") {
  tests<float16>(DeviceType::gpu, Type::float16);
}
#endif
#endif

#ifdef BLAS_FOUND
TEST_CASE("Expression graph supports basic math operations (cpu)", "[operator]") {
  tests<float>(DeviceType::cpu);
}
#endif
