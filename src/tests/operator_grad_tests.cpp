#include "catch.hpp"
#include "graph/expression_graph.h"
#include "graph/expression_operators.h"

using namespace marian;

void tests(DeviceType device, Type floatType = Type::float32) {

  Config::seed = 1234;
  auto graph = New<ExpressionGraph>();
  graph->setParameterType(Type::float32); // use float32 unless specified differently
  graph->setDevice({0, device});
  graph->reserveWorkspaceMB(16);

  float eps = 1e-6; // smaller does not work with fp16
  auto diffQuot = [eps](std::function<Expr(Expr)> f, Expr x) {

    // this only works if graph->backward() has been called!
    auto xGrad  = constant_like(x, inits::fromTensor(x->grad()));
    auto h      = constant_like(x, inits::uniform(eps, 10.f * eps));

    auto diff   = sum(flatten(f(x + h) - f(x))) - sum(flatten(xGrad * h));

    return diff;
  };

  struct FunArg1 {
    std::string name;
    float eps{0.001};
    std::function<Expr(Expr)> f;

    FunArg1(const std::string& nameArg, std::function<Expr(Expr)> fArg)
    : name(nameArg), eps(0.001), f(fArg) {}

    FunArg1(const std::string& nameArg, float epsArg, std::function<Expr(Expr)> fArg)
    : name(nameArg), eps(epsArg), f(fArg) {}

    Expr operator()(Expr x) const { return f(x); }
  };

  SECTION("Unary operators and functions") {

    std::vector<FunArg1> unaryFunctions;

    unaryFunctions.emplace_back("scalar mult", [](Expr x) { return x * 10.f; });
    unaryFunctions.emplace_back("scalar add", [](Expr x) { return x + 2.71f; });
    unaryFunctions.emplace_back("scalar div", [](Expr x) { return x / 0.5f; });
    unaryFunctions.emplace_back("scalar sub", [](Expr x) { return x - 2.71f; });
    unaryFunctions.emplace_back("x * x", [](Expr x) { return x * x; });
    unaryFunctions.emplace_back("x + x", [](Expr x) { return x + x; });
    unaryFunctions.emplace_back("x - x", [](Expr x) { return x - x; });
    //unaryFunctions.emplace_back("scalar mult", [](Expr x) { return x / x; }); // fails for fp16
    unaryFunctions.emplace_back("exp(x)", [](Expr x) { return exp(x); });
    unaryFunctions.emplace_back("sqrt(relu(x) + 1e-3)", [](Expr x) { return sqrt(relu(x) + 1e-3); });
    unaryFunctions.emplace_back("square(x)", [](Expr x) { return square(x); });
    unaryFunctions.emplace_back("transpose(x)", [](Expr x) { return transpose(x); });
    unaryFunctions.emplace_back("sigmoid(x)", [](Expr x) { return sigmoid(x); });
    unaryFunctions.emplace_back("tanh(x)", [](Expr x) { return tanh(x); });
    unaryFunctions.emplace_back("relu(x)", [](Expr x) { return relu(x); });

    unaryFunctions.emplace_back("swish(x)", [](Expr x) { return swish(x); });
    unaryFunctions.emplace_back("x * sigmoid(x)", [](Expr x) { return x * sigmoid(x); });

    unaryFunctions.emplace_back("softmax(x)", [](Expr x) { return softmax(x); });
    unaryFunctions.emplace_back("logsoftmax(x)", [](Expr x) { return logsoftmax(x); });
    unaryFunctions.emplace_back("exp(x) / sum(exp(x), -1)", [](Expr x) { return exp(x) / sum(exp(x), -1); });

    for(auto f : unaryFunctions) {
      std::cerr << f.name << " " << floatType << " " << DeviceId({0, device}) << std::endl;

      graph->clear();
      auto x = graph->param("x", {128, 128}, inits::uniform(-5.f, 5.f), Type::float32);

      auto fx = sum(flatten(f(cast(x, floatType))));

      graph->forward();
      graph->backward();

      auto diff = diffQuot(f, x);

      graph->forwardNext();

      CHECK( diff->scalar() == Approx(0.f).epsilon(f.eps) );
    }
  }

  struct FunArg2 {
    std::string name;
    float eps{0.001};
    std::function<Expr(Expr, Expr)> f;

    FunArg2(const std::string& nameArg, std::function<Expr(Expr, Expr)> fArg)
    : name(nameArg), eps(0.001), f(fArg) {}

    FunArg2(const std::string& nameArg, float epsArg, std::function<Expr(Expr, Expr)> fArg)
    : name(nameArg), eps(epsArg), f(fArg) {}

    Expr operator()(Expr x, Expr y) const { return f(x, y); }
  };

  SECTION("Binary operators and functions") {

    std::vector<FunArg2> binaryFunctions;

    binaryFunctions.emplace_back("x + y", [](Expr x, Expr y) { return x + y; });
    binaryFunctions.emplace_back("x * y", [](Expr x, Expr y) { return x * y; });
    binaryFunctions.emplace_back("exp(sigmoid(x) - 0.2 * y)", 0.02, [](Expr x, Expr y) { return exp(sigmoid(x) - 0.2 * y); });
    binaryFunctions.emplace_back("x / (relu(y) + 0.001)", [](Expr x, Expr y) { return x / (relu(y) + 0.01); });
    binaryFunctions.emplace_back("dot(x, y, false, false)", 0.02, [](Expr x, Expr y) { return dot(x, y, false, false);});
    binaryFunctions.emplace_back("dot(x, y, true, false)", 0.02, [](Expr x, Expr y) { return dot(x, y, true, false); });
    binaryFunctions.emplace_back("dot(x, y, false, true)", 0.02, [](Expr x, Expr y) { return dot(x, y, false, true); });
    binaryFunctions.emplace_back("dot(x, y, true, true)", 0.02, [](Expr x, Expr y) { return dot(x, y, true, true); });

    for(auto f : binaryFunctions) {
      std::cerr << f.name << " " << floatType << " " << DeviceId({0, device}) << std::endl;

      graph->clear();
      auto x = graph->param("x", {128, 128}, inits::uniform(-5.f, 5.f));
      auto y = graph->param("y", {128, 128}, inits::uniform(-5.f, 5.f));

      auto fxy = f(cast(x, floatType), cast(y, floatType));

      auto fx = [f,y](Expr x) { return f(x, y); };
      auto fy = [f,x](Expr y) { return f(x, y); };

      graph->forward();
      graph->backward(); // required to compute gradient of x

      auto diffx = diffQuot(fx, x);
      auto diffy = diffQuot(fy, y);

      graph->forwardNext();

      std::vector<float> vDiffx;
      diffx->val()->get(vDiffx); // this should be zero mostly

      std::vector<float> vDiffy;
      diffy->val()->get(vDiffy); // this should be zero mostly

      CHECK( diffx->scalar() == Approx(0.f).epsilon(f.eps) );
      CHECK( diffy->scalar() == Approx(0.f).epsilon(f.eps) );

    }
  }
}

#ifdef CUDA_FOUND
TEST_CASE("Expression graph supports basic math operations (gpu)", "[operator]") {
  tests(DeviceType::gpu);
}

TEST_CASE("Expression graph supports basic math operations (gpu fp16)", "[operator]") {
  tests(DeviceType::gpu, Type::float16);
}
#endif

#ifdef BLAS_FOUND
TEST_CASE("Expression graph supports basic math operations (cpu)", "[operator]") {
  tests(DeviceType::cpu);
}
#endif
