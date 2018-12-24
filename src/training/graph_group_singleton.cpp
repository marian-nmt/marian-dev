#include "training/graph_group_singleton.h"

namespace marian {

void SingletonGraph::setScheduler(Ptr<Scheduler> scheduler) {
  scheduler_ = scheduler;
  // optimizer has to be registered last to see changes of learning rate
  scheduler_->registerTrainingObserver(scheduler_);
  scheduler_->registerTrainingObserver(opt_);
}

void SingletonGraph::execute(Ptr<data::Batch> batch) {
  auto costNode = builder_->build(graph_, batch);

  ABORT_IF(costScale_, "Cost-scaling not implemented");

  if(costScaleFactor_ != 1.f)
    costNode = costNode * costScaleFactor_;

  graph_->forward();
  float cost = costNode->scalar() / costScaleFactor_;
  graph_->backward();

  if(costScale_) {
    bool hasNan = false, hasInf = false;
    IsNan(graph_->params()->grads(), graph_->allocator(), hasNan, hasInf);
    if(hasNan || hasInf) {
      costScaleFactor_ /= costScaleMultiplier_;
      LOG(warn, "Seen NaN/Inf in gradient, skipping update, reducing cost-scaling factor to {}", costScaleFactor_);
      noNanSeen_ = 0;
      return;
    }
    noNanSeen_++;
  }

  // Get batch stats
  opt_->update(graph_->params()->vals(),
               graph_->params()->grads(),
               OptimizerBase::mbSizeNotProvided,
               costScaleFactor_);

  if(scheduler_) {
    scheduler_->update(cost, batch);

    if(scheduler_->validating()) {
      if(mvAvg_) {
        ABORT("Not implemented");
         //graphAvg_->reuseWorkspace(graph_);
         //scheduler_->validate({graphAvg_});
      } else {
        scheduler_->validate({graph_});
      }
    }

    if(scheduler_->saving())
      this->save();
  }

  if(costScale_ && noNanSeen_ > 0 && noNanSeen_ % costScaleFreq_ == 0) {
    costScaleFactor_ *= costScaleMultiplier_;
    LOG(info, "No NaN/Inf seen for {} updates. Increasing cost-scaling factor to {}", noNanSeen_, costScaleFactor_);
  }
}

}  // namespace marian

