#pragma once

#include "common/options.h"
#include "functional/functional.h"
#include "graph/expression_graph.h"
#include "tensors/backend.h"
#include "tensors/tensor.h"
#include "tensors/tensor_allocator.h"
#include "tensors/tensor_operators.h"

namespace marian {

/* Quantize all the parameters in a model graph
 * This class handles the required error-feedback mechanism internally.
 * Example:
 *   auto mq = New<ModelQuantizer>(options_);
 *   mq->quantize(graph_);
 *
 * parameters in graph_ will be quantized every time quantize is called.
 * the internal error-residual is also updated each quantize call,
 * therefore, use the same ModelQuantizer object to quantize the same graph.
 */
class ModelQuantizer {
public:
  ModelQuantizer(Ptr<Options> options)
      : bits_{options->get<size_t>("quantize-bits")},
        optSteps_{options->get<size_t>("quantize-optimization-steps")},
        quantBias_{options->get<bool>("quantize-bias")},
        logQuant_{options->get<bool>("quantize-log")} {}

  void quantize(Ptr<ExpressionGraph> graph);

protected:
  void quantizeImpl(Tensor t);

  size_t bits_;
  size_t optSteps_;
  bool 	quantBias_;
  bool logQuant_;
  bool firstError_;

  std::vector<Ptr<TensorAllocator>> allocators_;

  Tensor errorResidual_;
  
  // temporary Tensor for storing q to calculate optimal S
  Tensor delta_;

  // single element Tensor for Reduce swap variable
  Tensor tempVar_;
};
}  // namespace marian
