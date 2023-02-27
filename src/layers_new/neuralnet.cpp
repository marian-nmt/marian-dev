#include "layers_new/neuralnet.h"

namespace marian {
namespace nn {

// Factory for activation function layers from name as string.
Ptr<Activation> activationLayerByName(Ptr<ExpressionGraph> graph, const std::string& actName) {
  // @TODO: lowercase actName first?
  if(actName == "relu")
    return New<ReLU>(graph);
  else if(actName == "gelu")
    return New<GELU>(graph);
  else if(actName == "tanh")
    return New<Tanh>(graph);
  else if(actName == "sigmoid")
    return New<Sigmoid>(graph);
  else if(actName == "swish")
    return New<Swish>(graph);
  else
    ABORT("Unknown activation function: {}", actName);
}

}
}
