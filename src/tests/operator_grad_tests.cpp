#include "catch.hpp"
#include "graph/expression_graph.h"
#include "graph/expression_operators.h"

using namespace marian;

template <typename T>
void tests(DeviceType device, Type floatType = Type::float32) {

  auto floatApprox = [](T x, T y) { return x == Approx(y).epsilon(0.1); };

  Config::seed = 1234;
  auto graph = New<ExpressionGraph>();
  graph->setParameterType(floatType);
  graph->setDevice({0, device});
  graph->reserveWorkspaceMB(16);

  std::vector<T> values;

  float eps = 1e-7; // smaller does not work with fp16

  // random noise is OK for elementwise functions
  auto randomNoise = [eps](Expr x) {
    return constant_like(x, inits::uniform(eps, 10.f * eps));
  };

  // only noise one column, ok for softmax and layerNorm and similar row-wise
  // gradients
  auto colNoise = [randomNoise](Expr x) {
    auto noise = randomNoise(x);

    auto graph = x->graph();
    int cols = x->shape()[-1];
    std::vector<float> vMask(cols, 0.f);
    // @TODO: choose column randomly
    vMask[2] = 1.f;
    auto colMask  = graph->constant({1, cols}, inits::fromVector(vMask));
    return colMask * noise;
  };

  // only noise one element, ok for matrix product
  // gradients
  auto elemNoise = [eps,colNoise](Expr x) {
    auto graph = x->graph();
    int rows = x->shape()[-2];
    
    auto noise = colNoise(x);
    
    std::vector<float> vMask(rows, 0.f);
    vMask[0] = 1.f;
    auto rowMask  = graph->constant({rows, 1}, inits::fromVector(vMask));
    return rowMask * noise;
  };

  auto diffQuot = [](std::function<Expr(Expr)> f, Expr x, 
                     std::function<Expr(Expr)> noise) {
    auto fx = f(x);
    auto h = noise(x);
    
    auto graph = x->graph();
    graph->setInference(true);
    auto fxh = f(x + h);
    auto dq  = (fxh - fx) / h;
    graph->setInference(false);
    
    return dq;
  };

  auto diffQuotAll = [diffQuot, randomNoise](std::function<Expr(Expr)> f, Expr x) {
    return diffQuot(f, x, randomNoise);
  };


  // auto diffQuotCol = [diffQuot, colNoise](std::function<Expr(Expr)> f, Expr x) {
  //   return diffQuot(f, x, colNoise);
  // };

  auto diffQuotElem = [diffQuot, elemNoise](std::function<Expr(Expr)> f, Expr x) {
    return diffQuot(f, x, elemNoise);
  };

  SECTION("Unary operators and functions") {

    std::vector<std::function<Expr(Expr)>> unaryFunctions;

    unaryFunctions.push_back([](Expr x) { return x * 0.5f; });
    unaryFunctions.push_back([](Expr x) { return x + 2.71f; });
    unaryFunctions.push_back([](Expr x) { return x / 0.5f; });
    unaryFunctions.push_back([](Expr x) { return x - 2.71f; });
    unaryFunctions.push_back([](Expr x) { return x * x; });
    unaryFunctions.push_back([](Expr x) { return x + x; });
    unaryFunctions.push_back([](Expr x) { return x - x; });
    unaryFunctions.push_back([](Expr x) { return exp(x); });
    unaryFunctions.push_back([](Expr x) { return sqrt(x); });
    unaryFunctions.push_back([](Expr x) { return square(x); });
    unaryFunctions.push_back([](Expr x) { return transpose(x); });
    unaryFunctions.push_back([](Expr x) { return sigmoid(x); });
    unaryFunctions.push_back([](Expr x) { return tanh(x); });
    //unaryFunctions.push_back([](Expr x) { return relu(x); });
    
    unaryFunctions.push_back([](Expr x) { return swish(x); });
    unaryFunctions.push_back([](Expr x) { return x * sigmoid(x); });

    unaryFunctions.push_back([](Expr x) { return softmax(x); });
    unaryFunctions.push_back([](Expr x) { return logsoftmax(x); });
    unaryFunctions.push_back([](Expr x) { return exp(x) / sum(exp(x), -1); });

    for(auto f : unaryFunctions) {

      graph->clear();
      auto x = graph->param("x", {128, 128}, inits::uniform(-1.f, 1.f));
      
      auto fx = f(x);
      auto dq = diffQuotAll(f, x);

      graph->forward();
      graph->backward();

      std::vector<T> grad1;
      x->grad()->get(grad1);

      std::vector<T> grad2;
      dq->val()->get(grad2);

      // std::cerr << fx->type() << std::endl;
      //  std::cerr << fx->val()->debug(4, 32) << std::endl;
      //  std::cerr << x->grad()->debug(4, 32) << std::endl;
      //  std::cerr << dq->val()->debug(4, 32) << std::endl;

      CHECK( std::equal(grad1.begin(), grad1.end(),
                        grad2.begin(), floatApprox) );

    }
  }

  SECTION("Binary operators and functions") {

    std::vector<std::function<Expr(Expr, Expr)>> binaryFunctions;

    binaryFunctions.push_back([](Expr x, Expr y) { return x + y; });
    binaryFunctions.push_back([](Expr x, Expr y) { return x * y; });
    binaryFunctions.push_back([](Expr x, Expr y) { return exp(sigmoid(x) - 5.2 * y); });
    // binaryFunctions.push_back([](Expr x, Expr y) { return x / y; }); // this fails, numeric issues.
    
    binaryFunctions.push_back([](Expr x, Expr y) { return dot(x, y, false, false); });
    // binaryFunctions.push_back([](Expr x, Expr y) { return dot(x, y, true, false); });
    // binaryFunctions.push_back([](Expr x, Expr y) { return dot(x, y, false, true); });
    // binaryFunctions.push_back([](Expr x, Expr y) { return dot(x, y, true, true); });
    
    for(auto f : binaryFunctions) {

      graph->clear();
      auto x = graph->param("x", {5, 5}, inits::uniform(-1.f, 1.f));
      auto y = graph->param("y", {5, 5}, inits::uniform(-1.f, 1.f));
      
      auto fxy = f(x, y);

      auto f_x = [f,y](Expr x) { return f(x, y); };
      auto f_y = [f,x](Expr y) { return f(x, y); };

      auto dq_x = diffQuotElem(f_x, x);
      auto dq_y = diffQuotElem(f_y, y);

      graph->forward();
      graph->backward();

      std::vector<T> grad1x;
      x->grad()->get(grad1x);

      std::vector<T> grad1y;
      y->grad()->get(grad1y);

      std::vector<T> grad2x;
      dq_x->val()->get(grad2x);

      std::vector<T> grad2y;
      dq_y->val()->get(grad2y);

      std::cerr << fxy->type() << std::endl;
      std::cerr << fxy->val()->debug(4, 5) << std::endl;
      std::cerr << x->val()->debug(4, 5) << std::endl;
      std::cerr << x->grad()->debug(4, 5) << std::endl;
      std::cerr << dq_x->val()->debug(4, 5) << std::endl;
      std::cerr << y->val()->debug(4, 5) << std::endl;
      std::cerr << y->grad()->debug(4, 5) << std::endl;
      std::cerr << dq_y->val()->debug(4, 5) << std::endl;
      
      CHECK( std::equal(grad1x.begin(), grad1x.end(),
                        grad2x.begin(), floatApprox) );

      CHECK( std::equal(grad1y.begin(), grad1y.end(),
                        grad2y.begin(), floatApprox) );

    }
  }
}

#ifdef CUDA_FOUND
TEST_CASE("Expression graph supports basic math operations (gpu)", "[operator]") {
  tests<float>(DeviceType::gpu);
}

TEST_CASE("Expression graph supports basic math operations (gpu fp16)", "[operator]") {
  tests<float16>(DeviceType::gpu, Type::float16);
}
#endif

#ifdef BLAS_FOUND
TEST_CASE("Expression graph supports basic math operations (cpu)", "[operator]") {
  tests<float>(DeviceType::cpu);
}
#endif
