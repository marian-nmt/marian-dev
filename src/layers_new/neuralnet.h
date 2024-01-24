#pragma once

#include "layers_new/interface.h"
#include "graph/node_initializers.h"

namespace marian {
namespace nn {

static inline Expr swapTimeBatch(Expr input) { return swapAxes(atleast_4d(input), -2, -3); }

/**
 * A generic Activation function layer. Any unary Marian operator or function accepted by
 * `std::function<Expr(Expr)>` can be turned into an activation function like this:
 ```
 auto reluLayer = New<Activation>(graph, (Expr(*)(Expr))relu)
 ```
 * The function pointer cast may be required to disambiguate the operator name if operators
 * of the same name but with a different sets of parameters exist, otherwise it can be dropped
 * or replaced with a more readable lambda function.
 *
 * `Activation` will also accept lambdas for more complex activations:
 ```
 // a reasonably accurate approximation of GELU
 auto geluApprox = New<Activation>(graph, [](Expr x) { return x * sigmoid(1.702f * x); });
 ```
 */
class Activation : public Layer, public IUnaryLayer {
private:
  std::function<Expr(Expr)> actFn;

public:
  Activation(Ptr<ExpressionGraph> graph,
             const std::function<Expr(Expr)>& actFn)
    : Layer(graph), actFn(actFn) {}

  virtual ~Activation() = default;

  Expr apply(Expr x) const override {
    return actFn(x);
  }
};

// A ReLU activation function layer defined via `Activation`.
struct ReLU final : public Activation {
  ReLU(Ptr<ExpressionGraph> graph)    : Activation(graph, (Expr(*)(Expr))relu) {}
};

// A GELU activation function layer defined via `Activation`.
struct GELU final : public Activation {
  GELU(Ptr<ExpressionGraph> graph)    : Activation(graph, (Expr(*)(Expr))gelu) {}
};

// A Tanh activation function layer defined via `Activation`.
struct Tanh final : public Activation {
  Tanh(Ptr<ExpressionGraph> graph)    : Activation(graph, (Expr(*)(Expr))tanh) {}
};

// A Sigmoid activation function layer defined via `Activation`.
struct Sigmoid final : public Activation {
  Sigmoid(Ptr<ExpressionGraph> graph) : Activation(graph, (Expr(*)(Expr))sigmoid) {}
};

// A Swish activation function layer defined via `Activation`.
struct Swish final : public Activation {
  Swish(Ptr<ExpressionGraph> graph)   : Activation(graph, (Expr(*)(Expr))swish) {}
};

// Factory for activation function layers from name as string.
Ptr<Activation> activationLayerByName(Ptr<ExpressionGraph> graph, const std::string& actName);

// Applies a linear transformation to the incoming data: y = xA^T + b
struct Linear : public Layer, public IUnaryLayer {
  Expr weight;
  Expr bias;

  int dimOut;
  bool useBias{true};
  bool transposed{false};
  Ptr<inits::NodeInitializer> init;

  // Typical constructor that can take an initializer function
  Linear(Ptr<ExpressionGraph> graph,
         int dimOut,
         bool useBias = true,
         bool transposed = false,
         Ptr<inits::NodeInitializer> init = inits::glorotUniform())
    : Layer(graph), dimOut(dimOut), useBias(useBias), init(init)
  {}

  // Alternate constructor which takes a weight parameter that will be re-used, e.g. for tied output weights.
  // Since the weights are already initialized there is no initializer. Output dimension is initialized from
  // the given weight parameter.
  Linear(Ptr<ExpressionGraph> graph,
         Expr tiedWeight,
         bool useBias = true,
         bool transposed = false)
    : Layer(graph), weight(tiedWeight), dimOut(weight->shape()[-1]), useBias(useBias), init(nullptr)
  {}

  virtual ~Linear() = default;

  Expr apply(Expr x) const override {
    int dimIn = x->shape()[-1];

    // if weight is already initialized nothing happens here
    if(transposed) {
      registerParameterLazy(weight, Shape({ dimOut, dimIn }), init);
    } else {
      registerParameterLazy(weight, Shape({ dimIn, dimOut }), init);
    }

    if(useBias) {
      registerParameterLazy(bias, Shape({ dimOut }), inits::zeros());
    }

    Type outputType = x->value_type();
    if(useBias)
      return marian::affine(x,
                            marian::cast(weight, outputType),
                            marian::cast(bias, outputType),
                            /*transA=*/false,
                            /*transB=*/transposed);
    else
      return marian::dot(x,
                         marian::cast(weight, outputType),
                         /*transA=*/false,
                         /*transB=*/transposed);
  }
};

struct Dropout final : public Layer, public IUnaryLayer {
  float dropoutProbability;
  Shape::Axes dropoutAxes{{-2, -1}};

  Dropout(Ptr<ExpressionGraph> graph,
          float dropoutProbability,
          const Shape::Axes& dropoutAxes)
    : Layer(graph), dropoutProbability(dropoutProbability), dropoutAxes(dropoutAxes)
  {}

  Dropout(Ptr<ExpressionGraph> graph,
          float dropoutProbability)
    : Layer(graph), dropoutProbability(dropoutProbability)
  {}

  Expr apply(Expr input) const override {
    if(getMode() == Mode::eval)
      return input;

    if(dropoutProbability > 0.f) {
      return marian::dropout(input, dropoutProbability, dropoutAxes);
    } else {
      return input;
    }
  }

  virtual void clear() override {}
};

struct LinearReluDropout final : public Linear {
  using Linear::weight;
  using Linear::bias;

  using Linear::dimOut;
  using Linear::useBias;
  using Linear::transposed;
  using Linear::init;

  float dropoutProbability;
  Shape::Axes dropoutAxes{{-2, -1}};

  // Typical constructor that can take an initializer function
  LinearReluDropout(Ptr<ExpressionGraph> graph,
                    int dimOut,
                    float dropoutProbability,
                    bool useBias = true,
                    bool transposed = false,
                    Ptr<inits::NodeInitializer> init = inits::glorotUniform())
    : Linear(graph, dimOut, useBias, transposed, init),
      dropoutProbability(dropoutProbability) {}

  // Typical constructor that can take an initializer function
  LinearReluDropout(Ptr<ExpressionGraph> graph,
                    int dimOut,
                    float dropoutProbability,
                    const Shape::Axes& dropoutAxes,
                    bool useBias = true,
                    bool transposed = false,
                    Ptr<inits::NodeInitializer> init = inits::glorotUniform())
    : Linear(graph, dimOut, useBias, transposed, init),
      dropoutProbability(dropoutProbability), dropoutAxes(dropoutAxes) {}

  Expr apply(Expr x) const override {
    int dimIn = x->shape()[-1];

    // if weight is already initialized nothing happens here
    if(transposed) {
      registerParameterLazy(weight, Shape({ dimOut, dimIn }), init);
    } else {
      registerParameterLazy(weight, Shape({ dimIn, dimOut }), init);
    }

    if(useBias) {
      registerParameterLazy(bias, Shape({ dimOut }), inits::zeros());
    }

    Expr output;
    if(useBias)
      output = marian::affine(x, weight, bias, /*transA=*/false, /*transB=*/transposed);
    else
      output = marian::dot(x, weight, /*transA=*/false, /*transB=*/transposed);

    if(getMode() == Mode::eval) {
      return marian::dropoutReluInplace(output); // no dropout
    } else {
      return marian::dropoutReluInplace(output, dropoutProbability, dropoutAxes);
    }
  }

  virtual void clear() override {}
};

struct Norm : public Layer, public IUnaryLayer {
  Expr weight{nullptr}; // = scale
  Expr bias{nullptr};

  bool useScale{true};
  bool useBias{true};
  bool elementwise{true};
  float eps{1e-5f};

  Norm(Ptr<ExpressionGraph> graph,
       bool useScale = true,
       bool useBias = true,
       bool elementwise = true,
       float eps = 1e-5f)
    : Layer(graph),
      useScale(useScale),
      useBias(useBias),
      elementwise(elementwise),
      eps(eps) {}

  virtual Expr getScale(int dimModel) const {
    Expr scaleVector = nullptr;
    if(useScale) {
      registerParameterLazy(weight, Shape({ elementwise ? dimModel : 1 }), inits::ones());
      // if elementwise==false we multiply with a vector of 1s - that's a trick to make gradient computation faster
      scaleVector = elementwise ? weight : weight * graph()->ones({dimModel}); // @TODO: make this obsolete
    }
    return scaleVector;
  }

  virtual Expr getBias(int dimModel) const {
    Expr biasVector = nullptr;
    if(useBias) {
      registerParameterLazy(bias,  Shape({ elementwise ? dimModel : 1 }), inits::zeros());
      // if elementwise==false we multiply with a vector of 1s - that's a trick to make gradient computation faster
      biasVector = elementwise ? bias : bias * graph()->ones({dimModel}); // @TODO: make this obsolete
    }
    return biasVector;
  }

  Expr apply(Expr x) const override = 0;
};

struct LayerNorm : public Norm {
  LayerNorm(Ptr<ExpressionGraph> graph,
            bool useScale = true,
            bool useBias = true,
            bool elementwise = true,
            float eps = 1e-5f)
   : Norm(graph, useScale, useBias, elementwise, eps)
  {}

  Expr apply(Expr x) const override {
    int dimModel = x->shape()[-1];
    return marian::layerNorm(x, getScale(dimModel), getBias(dimModel), eps);
  }

  virtual void clear() override {}
};

struct RMSNorm : public Norm {
  RMSNorm(Ptr<ExpressionGraph> graph,
          bool useScale = true,
          bool useBias = true,
          bool elementwise = true,
          float eps = 1e-5f)
   : Norm(graph, useScale, useBias, elementwise, eps)
  {}

  Expr apply(Expr x) const override {
    int dimModel = x->shape()[-1];
    return marian::rmsNorm(x, getScale(dimModel), getBias(dimModel), eps);
  }
};

} // namespace nn
} // namespace marian
