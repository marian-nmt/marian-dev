#include "training/graph_group_async.h"
#include "training/graph_group_async_drop.h"

#include "functional/functional.h"
#include "kernels/tensor_operators.h"
#include "training/dropper.h"
#include "training/sparse_tensor.h"

namespace marian {

Tensor AsyncGraphGroupDrop::newTensor(int size, int device) {
  Tensor t;
  Ptr<TensorAllocator> allocator_ = New<TensorAllocator>(device);
  allocator_->reserveExact(size * sizeof(float));
  allocator_->allocate(t, {1, size});
  allocators.push_back(allocator_);

  return t;
}

void AsyncGraphGroupDrop::init(Ptr<data::Batch> batch) {
  AsyncGraphGroup::init(batch);
  // extra inits for gradient dropping
  if(drop_first) {
    int totalSize = graphs_[0]->params()->vals()->size();
    int sparseCap = totalSize * 1.2 * (1.0 - 0.99);
    int shardSize = ceil(totalSize / devices_.size());

    for(int i = 0; i < devices_.size(); i++)
      paramsLocal_.push_back(std::vector<Tensor>());

    for(int i = 0; i < devices_.size(); i++) {
      // warm-up counter
      fetchStep_.push_back(0);
      pushStep_.push_back(0);

      int device = devices_[i];
      // temporary tensor to compute parameter delta before fetching
      paramsDelta_.push_back(newTensor(shardSize, device));

      // tensors to store local params history
      for(int h_id = 0; h_id < devices_.size(); h_id++) {
        Tensor tmp = newTensor(params_[i]->size(), device);
        tmp->copyFrom(params_[i]);
        paramsLocal_[h_id].push_back(tmp);
      }

      // individual Gradient dropper per-device
      pushDropper_.push_back(GradientDrop(new GradientDropBase()));

      // N-dropper for fetch
      std::vector<GradientDrop> tmpDropper;
      for(int i = 0; i < devices_.size(); i++)
        tmpDropper.push_back(GradientDrop(new GradientDropBase()));
      fetchDropper.push_back(tmpDropper);

      // sparsetensor to store sparsified gradients per-device
      pushSparseGradient_.push_back(
          SparseTensor(new SparseTensorBase(sparseCap, device)));

      pushShardedSparseGradient_.push_back(
          SparseTensor(new SparseTensorBase(sparseCap, device)));
      fetchSparseGradient_.push_back(SparseTensor(
          new SparseTensorBase(sparseCap / devices_.size(), device)));

      std::vector<SparseTensor> tmp;
      for(int i = 0; i < devices_.size(); i++)
        tmp.push_back(SparseTensor(
            new SparseTensorBase(sparseCap / devices_.size(), device)));
      fetchShardedSparseGradient_.push_back(tmp);
    }

    drop_first = false;
  }
}

void AsyncGraphGroupDrop::computeDelta(
    Tensor delta,
    Tensor param,
    Tensor local) {
  using namespace functional;
  Element(_1 = _2 - _3, delta, param, local);
}

}
