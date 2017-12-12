#include "training/graph_group_sync.h"

namespace marian {

void SyncGraphGroup::fetchParams(Tensor oldParams,
                                 const std::vector<Tensor>& params) {
  // @TODO read guard on parameters
  int pos = 0;
  std::vector<std::thread> threads;
  for(int idx = 0; idx < devices_.size(); idx++) {
    threads.emplace_back(std::thread(
        [=](int idx, int pos) {
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

}
