#pragma once

#include <vector>

#include "common/definitions.h"

namespace marian {

class TrainingState;

class TrainingObserver {
public:
  virtual void actAfterEpoch(TrainingState& state) = 0;
  virtual void actBeforeBatches(TrainingState& state, size_t batchWords=0) {}
  virtual void actAfterBatches(TrainingState& state) {}
  virtual void actAfterStalled(TrainingState& state) {}
};

class TrainingState {
public:
  int epochs{1};
  int batches{0};
  int stalled{0};
  int maxStalled{0};
  int warmupStart{0};
  float eta;
  float multiplyFactor_; // For scaling LR according to batch size.
                                      // @TODO make sure it works for pushGradients because they spawn subthreads
  float factor{1.f};
  bool reset{false};

  TrainingState(float learnRate) : eta(learnRate) {}

  void registerObserver(Ptr<TrainingObserver> observer) {
    observers_.push_back(observer);
  }

  void newEpoch() {
    ++epochs;
    for(auto observer : observers_)
      observer->actAfterEpoch(*this);
  }

  void preBatch(size_t batchWords) {
    for(auto observer : observers_)
      observer->actBeforeBatches(*this, batchWords); //Nothing To do here so far
  }

  void newBatch() {
    ++batches;
    for(auto observer : observers_)
      observer->actAfterBatches(*this);
  }

  void newStalled(int num) {
    stalled = num;
    if(num > maxStalled)
      ++maxStalled;
    for(auto observer : observers_)
      observer->actAfterStalled(*this);
  }

private:
  std::vector<Ptr<TrainingObserver>> observers_;
};
}
