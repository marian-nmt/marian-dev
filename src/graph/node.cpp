#include "graph/node.h"
#include "graph/auto_tuner.h"
#include "graph/expression_graph.h"
#include "tensors/backend.h"

namespace marian {

#if USE_LAYER_TIMER
std::unordered_map<std::string, size_t> Node::executedCnt_ = {};
std::unordered_map<std::string, double> Node::cumulTime_ = {};
#endif

size_t Node::allocate() {
  size_t elements = 0;
  if(!val_) {
    graph()->allocateForward(this);
    elements = val_->shape().elements();
  }
  return elements;
}

void Node::free() {
  if(graph()) {
    if(val_) {
      graph()->free(val_);
      val_ = nullptr;
    }
    if(adj_) {
      graph()->free(adj_); 
      adj_ = nullptr;
    }
  }
}

/**
 * Initialization for backward step of top node
 * in computation graph. Allocates memory and sets gradient
 * to 1 (df/df == 1).
 */
void Node::init_dependent() {
  if(!adj_) {
    graph()->allocateBackward(this);
    adj_->set(1.f);
  }
}

/**
 * Initialization for backward step of any non-top node
 * in computation graph. Allocates memory and sets gradient
 * to 0 for further accumulation of gradients from all
 * parents.
 */
void Node::set_zero_adjoint() {
  if(!adj_) {
    graph()->allocateBackward(this);
    adj_->set(0.f);
  }
}

float Node::scalar() {
  return val_->scalar();
}

Ptr<Backend> Node::getBackend() {
  return graph()->getBackend();
}

void Node::forward() {
  if(recorder_)
    recorder_->start(recorderHash_);

#if USE_LAYER_TIMER
  std::string label = this->label();
  size_t cnt = 0;

  if(executedCnt_.find(label) == executedCnt_.end()) {
    executedCnt_[label] = 1;
  } else {
    cnt = ++executedCnt_[label];
  }

  if(cnt > warmup_) {
    auto start = std::chrono::high_resolution_clock::now();

    runForward(forwardOps());

    double duration = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - start).count();
    double dur = duration + cumulTime_[label];
    cumulTime_[label] = dur;

    if((cnt % warmup_) == 0) {
      LOG(info, "[benchmark] node: {}, count: {}, cumulative time: {}", label, cnt - warmup_, dur / 1e3);
    }
  } else {
    runForward(forwardOps());
  }
#else // USE_LAYER_TIMER
  runForward(forwardOps());
#endif // USE_LAYER_TIMER

  if(recorder_)
    recorder_->stop(recorderHash_, recorderStop_);
}

void Node::backward() {
  if(recorder_)
    recorder_->start(recorderHash_);

  runBackward(backwardOps());

  if(recorder_ && recorderStop_)
    recorder_->stop(recorderHash_, recorderStop_);
}

void Node::record(Ptr<AutoTunerRecorder> recorder,
                  size_t recorderHash,
                  bool stop) {
  recorder_ = recorder;
  recorderHash_ = recorderHash;
  recorderStop_ = stop;
}
}  // namespace marian
