#include "training/graph_group_async.h"
#include "training/graph_group_async_drop.h"

#include "training/dropper.h"
#include "training/sparse_tensor.h"

namespace marian {

void AsyncGraphGroupDrop::fetchParams(Tensor oldParams,
                                      const std::vector<Tensor>& params,
                                      int device_id) {
  // @TODO read guard on parameters
  int pos = 0;

  std::vector<std::thread> threads;
  for(int i = 0; i < devices_.size(); i++) {
    threads.emplace_back(std::thread(
        [&](int idx, int pos) {
          // individual mutex per-shard
          std::lock_guard<std::mutex> guard(shardSync_[idx]);

          // normal fetch
          if(fetchStep_[device_id] < FETCH_WARMUP
             || &params == &paramsAvg_) {  // Do not use sparse fetch when
                                           // fetching from paramsAvg
            oldParams->subtensor(pos, params[idx]->size())
                ->copyFrom(params[idx]);
            paramsLocal_[device_id][idx]->copyFrom(params[idx]);
            return;
          }

          // sparse fetch
          // get delta : params latest version - current param (locally)
          computeDelta(paramsDelta_[idx], params[idx],
                       paramsLocal_[device_id][idx]);

          // update current local param
          paramsLocal_[device_id][idx]->copyFrom(params[idx]);

          // get sparse delta
          fetchDropper[device_id][idx]->dropGraph(
              paramsDelta_[idx], fetchSparseGradient_[idx], droping_rate);

          // move sparse delta
          fetchShardedSparseGradient_[device_id][idx]->copyFrom(
              fetchSparseGradient_[idx]);

          fetchShardedSparseGradient_[device_id][idx]->scatterAdd(
              oldParams->subtensor(pos, params[idx]->size()));
        },
        i,
        pos));

    pos += shardSize_;
  }
  for(auto&& t : threads) {
    t.join();
  }
  fetchStep_[device_id]++;
}

void AsyncGraphGroupDrop::pushGradients(Tensor newGrads,
                                        size_t batch_words,
                                        int device_id) {
  if(pushStep_[device_id]++ < PUSH_WARMUP) {
    AsyncGraphGroup::pushGradients(newGrads, batch_words, device_id);
    return;
  }

  // get the sparse gradient
  pushDropper_[device_id]->dropGraph(
      newGrads, pushSparseGradient_[device_id], droping_rate);

  SparseTensor newSparseGrads = pushSparseGradient_[device_id];
  // add instead of copy?
  std::vector<std::thread> threads;
  int pos = 0;
  for(int idx = 0; idx < devices_.size(); idx++) {
    threads.emplace_back(std::thread(
        [=](int idx, int pos) {
          // individual mutex per-shard
          std::lock_guard<std::mutex> guard(shardSync_[idx]);

          // split to shard
          SparseTensor subGrad
              = newSparseGrads->subtensor(pos, grads_[idx]->size(), idx);

          // send the sharded sparse tensor
          pushShardedSparseGradient_[idx]->copyFrom(subGrad);

          // convert back to dense, store it in grads_[idx]
          pushShardedSparseGradient_[idx]->toDense(grads_[idx], -pos);

          if(scaleLearningRate_) {
            shardOpt_[idx]->update(
                params_[idx], grads_[idx], batch_words / avgBatchWords_);
          } else {
            shardOpt_[idx]->update(params_[idx], grads_[idx]);
          }

          if(movingAvg_)
            AsyncGraphGroup::updateMovingAverage(
                paramsAvg_[idx], params_[idx], scheduler_->numberOfBatches());

        },
        idx,
        pos));

    pos += shardSize_;
  }
  for(auto&& t : threads)
    t.join();
}

}
