#pragma once

#include "common/utils.h"
#include "graph/expression_graph.h"
#include "graph/expression_operators.h"
#include "graph/node_initializers.h"

#include <type_traits>

namespace marian {
namespace nn {

// Interface: provides a class member to return the class name (type) as a string
struct IClassName {
  virtual std::string className() const {
    return utils::cxxTypeName(*this);
  }
};

// Interface: Unary function
struct IUnaryLayer {
  virtual Expr apply(Expr) const = 0;
};

// Interface: Binary function
struct IBinaryLayer {
  virtual Expr apply(Expr, Expr) const = 0;
};

// Interface: Ternary function
struct ITernaryLayer {
  virtual Expr apply(Expr, Expr, Expr) const = 0;
};

// Interface: 4ary function
struct IQuaternaryLayer {
  virtual Expr apply(Expr, Expr, Expr, Expr) const = 0;
};

// Interface: N-Ary function
struct INaryLayer {
  virtual Expr apply(const std::vector<Expr>& list) const = 0;
};

// Interface: implement a clearing function
struct IClearable {
  virtual void clear() = 0;
};


// Helper macro to turn parameter C++ variable name into a string.
#define registerParameter(paramArg, shape, init) \
do { \
  if(!paramArg) { \
    paramArg = this->param(#paramArg, shape, init); \
  } \
} while(0);

// Helper macro to turn parameter C++ variable name into a string.
// This version is meant to be used in apply(...) functions for lazy parameter inits 
// hence has to cast away constness.
#define registerParameterLazy(paramArg, shape, init) \
do { \
  using ThisLayerType = std::decay<decltype(*this)>::type; \
  ThisLayerType* thisLayer = const_cast<ThisLayerType*>(this); \
  if(!thisLayer->paramArg) { \
    thisLayer->paramArg = thisLayer->param(#paramArg, shape, init); \
  } \
} while(0);

// Helper macro to turn a layer C++ variable name into a string and to add the layer as a named sublayer to the parent layer
#define registerLayer(layerArg) \
do { \
  ABORT_IF(!layerArg, "Layer {} of type {} is not initialized", #layerArg, utils::cxxTypeName(layerArg)); \
  namedLayers_.emplace_back(#layerArg, layerArg); \
  if(!layerArg->registered()) { \
    layerArg->setName(#layerArg); \
    layerArg->setFirstParent(this); \
  } \
} while(0);

// Helper macro that adds the layer as a named sublayer to the parent layer and uses the given name. Different from above as 
// the C++ variable name itself is not used a name string. 
#define registerLayerWithName(layerArg, name) \
do { \
  ABORT_IF(!layerArg, "Layer {} of type {} with name {} is not initialized", #layerArg, utils::cxxTypeName(layerArg), name); \
  namedLayers_.emplace_back(name, layerArg); \
  if(!layerArg->registered()) { \
    layerArg->setName(name); \
    layerArg->setFirstParent(this); \
  } \
} while(0);

class Layer;

using NamedParameter = std::pair<std::string, Expr>;

template <class LayerType = Layer>
using NamedLayer = std::pair<std::string, Ptr<LayerType>>;

// Base class for all layers. Sub layers should inherit from this class and one or multiple of the interfaces (e.g. IUnaryLayer)
class Layer : public IClassName, public IClearable, public std::enable_shared_from_this<Layer> {
public:
  enum class Mode : int { eval, train };

private:
  Weak<ExpressionGraph> graph_;

  // Using naked pointer as a weak reference. Cannot use shared_ptr or weak_ptr 
  // as registration happens in constructor of parent layer and shared_from_this() 
  // cannot be used before parent layer constructor exits.
  Layer* firstParent_{nullptr};
  std::string name_;

  mutable Mode mode_{Mode::train}; // eval or train ?

protected:
  std::vector<NamedParameter> namedParameters_; // vector of all named parameters belonging to this specific layer (not recurisve)
  std::vector<NamedLayer<Layer>> namedLayers_;  // vector of all named sublayers for this specific layer (not recursive)

  // Create a layer parameter with a full name composed of the path to this layer and localName
  Expr param(const std::string& localName, const Shape& shape, const Ptr<inits::NodeInitializer>& init) {
    std::string fullName = fmt::format("{}->{}", path(), localName);
    auto parameter = graph()->param(fullName, shape, init);
    namedParameters_.emplace_back(localName, parameter);
    return parameter;
  }

public:
  Layer(Ptr<ExpressionGraph> graph)
    : graph_(graph) {}

  virtual ~Layer() = default;

  Ptr<ExpressionGraph> graph() { 
    auto graph = graph_.lock();
    ABORT_IF(!graph, "graph in layer {} expired?", path());
    return graph;
  }

  const Ptr<ExpressionGraph> graph() const { 
    auto graph = graph_.lock();
    ABORT_IF(!graph, "graph in layer {} expired?", path());
    return graph;
  }

#if 1
  // @TODO: this should be removed, currently hack to init graph.
  void setGraph(Ptr<ExpressionGraph> graph) {
    graph_ = graph;
    for(auto& lr: namedLayers())
      lr.second->setGraph(graph);
  }
#endif

  // Dynamic cast to requested layer type. Will return nullptr if not possible
  template <class LayerType>
  Ptr<LayerType> as() {
    return std::dynamic_pointer_cast<LayerType>(shared_from_this());
  }

  // Dynamic cast to requested layer type. Will return nullptr if not possible
  template <class LayerType>
  Ptr<LayerType> as() const {
    return const_cast<Layer*>(this)->as<LayerType>();
  }

  // Dynamic cast to requested layer type. Will abort if the cast is not possible.
  template <class LayerType>
  Ptr<LayerType> cast() {
    auto layerCast = as<LayerType>();
    ABORT_IF(!layerCast, "Layer {} cannot be cast to requested type {}", 
             className(),
             utils::cxxTypeName<LayerType>());
    return layerCast;
  }

  template <class LayerType>
  Ptr<LayerType> cast() const {
    return const_cast<Layer*>(this)->cast<LayerType>();
  }
  
  // Return all named parameters for this specific layer (not descending into sub-layers)
  std::vector<NamedParameter>& namedParameters() { return namedParameters_; }
  const std::vector<NamedParameter>& namedParameters() const { return namedParameters_; }

  // Return all named layers for this specific layer (not descending into sub-layers)
  std::vector<NamedLayer<Layer>>& namedLayers() { return namedLayers_; }
  const std::vector<NamedLayer<Layer>>& namedLayers() const { return namedLayers_; }

  // Return all named sub-layers for this layer and its sub-layers (descending recursively into sub-layers).
  // Can be used with layer type e.g. allNamedLayers<Linear>() to return only sub-layers of this type. 
  // Returned layers will then have the given type and do not need to be cast anymore.
  template <class LayerType = Layer>
  std::vector<NamedLayer<LayerType>> allNamedLayers() {
    std::vector<NamedLayer<LayerType>> layers;
    for(auto& namedLayer : namedLayers()) {
      auto castLayer = namedLayer.second->as<LayerType>();
      if(castLayer)
        layers.emplace_back(namedLayer.first, castLayer);
      
      auto subLayers = namedLayer.second->allNamedLayers<LayerType>();
      layers.insert(layers.end(), subLayers.begin(), subLayers.end());
    }
    return layers;
  }

  template <class LayerType = Layer>
  std::vector<NamedLayer<LayerType>> allNamedLayers() const {
    return const_cast<Layer*>(this)->allNamedLayers<LayerType>();
  }

  // Returns all sub-layers (only the layers, not the names) for this layer and its sub-layers (descending 
  // recursively into sub-layers). Can be used with layer type e.g. allLayers<Linear>() to return only 
  // sub-layers of this type. Returned layers will then have the given type and do not need to be cast anymore.
  template <class LayerType = Layer>
  std::vector<Ptr<LayerType>> allLayers() {
    std::vector<Ptr<LayerType>> layers;
    for(auto namedLayer : allNamedLayers<LayerType>())
      layers.push_back(namedLayer.second);
    return layers;
  }

  template <class LayerType = Layer>
  std::vector<Ptr<LayerType>> allLayers() const {
    return const_cast<Layer*>(this)->allLayers<LayerType>();
  }

  // Used by parent layers to set the name of a sub-layer.
  // @TODO: make this private and only allow friend access from layers before merging with master. 
  // Currently misused for top layer that has no parent layer that can set its name. 
  void setName(const std::string& name) { name_ = name; }

  const std::string& name() const { return name_; }

  // This sets the first parent of a sublayer (the layer a sublayer was first registered with).
  // This is required to generate the correct path/name for layer parameters at saving time. 
  void setFirstParent(Layer* parent) { 
    ABORT_IF(firstParent_ != nullptr, "Parent layer has already been set");
    ABORT_IF(parent == this, "Parent layer has to be different from child");
    firstParent_ = parent; 
  }

  // The parent layer of a sublayer is the first layer the sublayer has been registered with.
  // Subsequent calls to setFirstParent will abort if the parent is already set.
  bool registered() const {
    return firstParent_ != nullptr;
  }

  std::string path() const {
    std::vector<std::string> path;
    if(firstParent_)
      path.push_back(firstParent_->path());
    path.push_back(name_);
    return marian::utils::join(path, "->");
  }

  std::string layerInfo(bool includeChildren=false) const {
    std::stringstream ss;
    std::function<void(const Layer*, int)> recurse;
    recurse = [&](const Layer* layer, int level) {
      auto indent = utils::join(std::vector<std::string>(level, "  "), "");
      ss << indent << layer->name() << " : " << layer->className() << std::endl;
      for(auto& pr: layer->namedParameters())
        ss << indent << "  " << pr.first << " : " << pr.second->shape() << std::endl;
      if(includeChildren)
        for(auto& lr: layer->namedLayers())
          recurse(lr.second.get(), level + 1);
    };
    recurse(this, 0);
    return ss.str();
  }

  // Return Mode::eval or Mode::train. This is used to determine if training only layer-internal actions 
  // like dropout should be run. This will not affect graph-internal gradient propagation unless somehow
  // specified in a layer.  
  Mode getMode() const {
  #if 1
    if(graph()->isInference()) {
      return Mode::eval;
    } else {
      return Mode::train;
    }
  #else
    return mode_;
  #endif
  }

  // Set mode to Mode::eval for this layer and all sub-layers. This will disable dropout and similar actions.
  void setEvalMode() {
    mode_ = Mode::eval;
    for(auto& lr: namedLayers())
      lr.second->setEvalMode();
  }

  // Set mode to Mode::train for this layer and all sub-layers. This will enable dropout and similar actions.
  void setTrainMode() {
    mode_ = Mode::train;
    for(auto& lr: namedLayers())
      lr.second->setTrainMode();
  }

  virtual void clear() override {
    for(auto& lr : namedLayers())
      lr.second->clear();
  }
};

class LayerWithOptions : public Layer {
protected:
  Ptr<Options> options_;

public:
  LayerWithOptions(Ptr<ExpressionGraph> graph, Ptr<Options> options)
    : Layer(graph), options_(options) {}

  virtual ~LayerWithOptions() = default;

  template <typename T>
  T opt(const std::string key) const {
    return options_->get<T>(key);
  }

  template <typename T>
  T opt(const std::string key, const T& defaultValue) const {
    return options_->get<T>(key, defaultValue);
  }
};

/**
 * Wrapper to be used exclusively inside LayerList or other similar containers. This is allows to use the apply(...) functions
 * of a layer without having to cast to specific type (this is done internally based on the number of arguments). Inspired by
 * boost::any_type which allows to construct containers that hold various types. 
 * This should allow to use any layer and iterfaces will be added here as required.
 */
class AnyLayer final : public IUnaryLayer, 
                       public IBinaryLayer,
                       public ITernaryLayer,
                       public IQuaternaryLayer,
                       public INaryLayer,
                       public IClearable {
private:
  Ptr<Layer> layer_;

protected:
  // private/protected constructor, should only be created within listed classes with friendship
  AnyLayer(const Ptr<Layer>& layer)
    : layer_(layer) {}
  
  friend class LayerList;

public:
  // Dynamic cast to requested layer type. Will return nullptr if not possible
  template <class LayerType>
  Ptr<LayerType> as() const {
    return std::dynamic_pointer_cast<LayerType>(layer_);
  }

  // Dynamic cast to requested layer type. Will abort if the cast is not possible.
  template <class LayerType>
  Ptr<LayerType> cast() const {
    auto layerCast = as<LayerType>();
    ABORT_IF(!layerCast, "Layer {} cannot be cast to requested type {}", 
             layer_->className(),
             utils::cxxTypeName<LayerType>());
    return layerCast;
  }

  Expr apply(Expr input) const override {
    return cast<IUnaryLayer>()->apply(input);
  }

  Expr apply(Expr input1, Expr input2) const override {
    return cast<IBinaryLayer>()->apply(input1, input2);
  }

  Expr apply(Expr input1, Expr input2, Expr input3) const override {
    return cast<ITernaryLayer>()->apply(input1, input2, input3);
  }

  Expr apply(Expr input1, Expr input2, Expr input3, Expr input4) const override {
    return cast<IQuaternaryLayer>()->apply(input1, input2, input3, input4);
  }

  Expr apply(const std::vector<Expr>& inputs) const override {
    return cast<INaryLayer>()->apply(inputs);
  }

  virtual void clear() override {
    cast<IClearable>()->clear();
  }
};

/** 
 * Holds sublayers in a list and performs correct registration of sublayers. Sublayers are indexed
 * and can be accessed like array elements, including iteration.
 * `LayerList` -- in contrast to `Sequential` -- does not provide `apply` functions. 
 * You have to define the execution order and information flow in code.
 * 
 * See TransformerEncoder for an example where we hold the transformer layer stack in a LayerList,
 * but define a custom apply function (due to masks being external information and shared between layers).
 */
class LayerList : public Layer {
protected:
  std::vector<Ptr<AnyLayer>> layers_;

  template <class Last>
  void recursiveAppend(Last last) {
    append(last);
  }
  
  template <class First, class ...Rest>
  void recursiveAppend(First first, Rest ...rest) {
    append(first);
    recursiveAppend(rest...);
  }

public:
  LayerList(Ptr<ExpressionGraph> graph)
  : Layer(graph) {}

  template <class ...Layers>
  LayerList(Ptr<ExpressionGraph> graph, Layers ...layers)
  : Layer(graph) {
    recursiveAppend(layers...);
  }

  virtual ~LayerList() = default;

  /** 
   * This inserts an already existing sublayer from this or a different container which will result in 
   * parameter sharing if there are parameters.
  ```
  auto layers = New<LayerList>(graph);
  layers->append(New<Linear>(graph, 100)); // <- creates a new sublayer and registers it.
  layers->append(layers->at(0));           // <- no new sublayer created or registered; reference the first one.
  ```
  */
  void append(const Ptr<AnyLayer>& layer) {
    layers_.push_back(layer);
  }

  void append(const Ptr<Layer>& layer) {
    std::string name = fmt::format("at({})->as<{}>()", layers_.size(), layer->className());
    registerLayerWithName(layer, name);
    layers_.emplace_back(new AnyLayer(layer)); // not using New<...> because of missing friendship
  }

  /** 
   * Retrieve sublayer at index i
   */
  Ptr<AnyLayer> at(size_t i) const {
    return layers_[i];
  }

  auto begin() -> decltype(layers_.begin()) const {
    return layers_.begin();
  }

  auto end() -> decltype(layers_.end()) const {
    return layers_.end();
  }

  size_t size() const { return layers_.size(); }

  virtual void clear() override {
    for(auto& layer : layers_)
      layer->clear();
  }
};

/** 
 * `Sequential` is a list of layers similar to `LayerList`, but does provide a set of `apply` functions.
 * These function assume that the first element in the container can be a unary, binary, ternary
 * or n-ary layer, but all subsequent layers have to be unary layers as they will consume the single
 * output of their preceding layer. Non-unary layers will fail to execute during runtime if they are 
 * not the very first layer.
 * 
 * `Sequential` can be used to implement typical feed forward networks:
 * 
 ```
  using namespace marian::nn;

  auto seq = New<Sequential>(graph, 
    New<Linear>(graph, 100),
    New<ReLU>(graph),
    New<Dropout>(graph, 0.1f),
    New<Linear>(graph, 100),
    New<ReLU>(graph),
    New<LayerNorm>(graph)
  );

  Expr output = seq->apply(input);
 ```
 * For other application patterns use `LayerList` and implement them yourself by traversing the layers.
 */
class Sequential : public LayerList, 
                   public IUnaryLayer,
                   public IBinaryLayer,
                   public ITernaryLayer,
                   public IQuaternaryLayer,
                   public INaryLayer {
public:
  Sequential(Ptr<ExpressionGraph> graph)
  : LayerList(graph) {}

  template <class ...Layers>
  Sequential(Ptr<ExpressionGraph> graph, Layers ...layers)
  : LayerList(graph, layers...) {}

  virtual ~Sequential() = default;

  Expr apply(Expr input) const override {
    ABORT_IF(layers_.empty(), "Applying empty Sequential layer?");
    return applyTail(layers_[0]->apply(input));
  }

  Expr apply(Expr input1, Expr input2) const override {
    ABORT_IF(layers_.empty(), "Applying empty Sequential layer?");
    return applyTail(layers_[0]->apply(input1, input2));
  }

  Expr apply(Expr input1, Expr input2, Expr input3) const override {
    ABORT_IF(layers_.empty(), "Applying empty Sequential layer?");
    return applyTail(layers_[0]->apply(input1, input2, input3));
  }

  Expr apply(Expr input1, Expr input2, Expr input3, Expr input4) const override {
    ABORT_IF(layers_.empty(), "Applying empty Sequential layer?");
    return applyTail(layers_[0]->apply(input1, input2, input3, input4));
  }

  Expr apply(const std::vector<Expr>& inputs) const override {
    ABORT_IF(layers_.empty(), "Applying empty Sequential layer?");
    return applyTail(layers_[0]->apply(inputs));
  }

private:
  // apply remaining layers after first layer has been applied.
  Expr applyTail(Expr input) const {
    Expr output = input;
    for(int i = 1; i < layers_.size(); ++i)
      output = layers_[i]->apply(output);
    return output;
  } 

};

} // namespace nn
} // namespace marian
