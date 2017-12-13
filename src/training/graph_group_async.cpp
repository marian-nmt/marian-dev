#include "training/graph_group_async.h"

namespace marian {

void AsyncGraphGroup::fetchParams(Tensor oldParams,
                                  const std::vector<Tensor>& params,
                                  int device_id) {
  // @TODO read guard on parameters
  int pos = 0;

  std::vector<std::thread> threads;
  for(int idx = 0; idx < devices_.size(); idx++) {
    threads.emplace_back(std::thread(
        [&](int idx, int pos) {
          // individual mutex per-shard
          std::lock_guard<std::mutex> guard(shardSync_[idx]);
          oldParams->subtensor(pos, params[idx]->size())->copyFrom(params[idx]);
        },
        idx,
        pos));

    pos += shardSize_;
  }
  for(auto&& t : threads) {
    t.join();
  }
}

void AsyncGraphGroup::pushGradients(Tensor newGrads,
                                    size_t batch_words,
                                    int device_id) {
  // add instead of copy?
  std::vector<std::thread> threads;
  int pos = 0;
  for(int idx = 0; idx < devices_.size(); idx++) {
    threads.emplace_back(std::thread(
        [&](int idx, int pos) {
          // individual mutex per-shard
          std::lock_guard<std::mutex> guard(shardSync_[idx]);
          grads_[idx]->copyFrom(newGrads->subtensor(pos, grads_[idx]->size()));

          if(scaleLearningRate_) {
            shardOpt_[idx]->update(
                params_[idx], grads_[idx], batch_words / avgBatchWords_);
          } else {
            shardOpt_[idx]->update(params_[idx], grads_[idx]);
          }

          if(movingAvg_)
            updateMovingAverage(
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
