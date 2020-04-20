#pragma once

#include "common/options.h"
#include "graph/expression_graph.h"
#include "tensors/backend.h"
#include "tensors/tensor.h"
#include "tensors/tensor_operators.h"
#include "tensors/tensor_allocator.h"
#include "functional/functional.h"

namespace marian {
void compressImpl(Tensor t, int bit, float base, float clipRange, int kMeanStep = 0);

class Compresser {
public:
  Compresser(Ptr<Options> options) 
    : bit_{options->get<int>("compress-bit")},
      base_{options->get<float>("compress-base")},
      clip_{options->get<float>("compress-clip")},
      interval_{options->get<int>("compress-interval")},
      isMax_{options->get<bool>("compress-max-scale")},
      kMeans_{options->get<int>("compress-k-means")},
      skipBias_{options->get<bool>("compress-skip-bias")} {
    }
      
  void compress(Ptr<ExpressionGraph> graph) {
    // reserve tensor for error feedback mechanism
    if (!error) {
      LOG(info, " EXPERIMENTAL: Applying Log-{} based compress model to {}-bit every {} steps", base_, bit_, interval_);
      LOG(info, " K-means scale adjustment steps: {}", kMeans_);

      int elements = (int) graph->params()->vals()->size();
      errorAlloc = New<TensorAllocator>(graph->getBackend());
      errorAlloc->reserveExact(graph->params()->vals()->memory()->size());
      errorAlloc->allocate(error, {1, elements});
    }

    // apply eror feedback mechanism
    using namespace functional;
    Element(_1 += _2, graph->params()->vals(), error);
    error->copyFrom(graph->params()->vals());

    // compress every interval
    int skip_size = 0;
    if (++step % interval_ == 0)
      for(auto p : *graph->params()){
        // skip biases
        if (skipBias_ && p->val()->shape()[0] == 1) {
          if (step == 1 && p->val()->getDeviceId().no == 0) LOG(info, "skipping {}", p->name());
          skip_size += p->val()->size();
          continue;
        }

        //if (p->name() == "Wemb") {
	//  if (step == 1 && p->val()->getDeviceId().no == 0) LOG(info, "compress to 1 bit {}", p->name());
	//  compressImpl(p->val(), 1, base_, clip_, kMeans_);
	//}
        //else
          compressImpl(p->val(), bit_, base_, clip_, kMeans_);
      }
    if (step == 1 && graph->params()->vals()->getDeviceId().no == 0) LOG(info, "skipping total of {}", skip_size);

    // get new error
    Element(_1 -= _2, error, graph->params()->vals());
  
  }


protected:
  int step{0};
  Tensor error;
  Ptr<TensorAllocator> errorAlloc;
  int bit_;
  float base_;
  float clip_;
  int interval_;
  bool isMax_;
  int kMeans_;
  bool skipBias_;
};
}
