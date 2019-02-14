#include "training/graph_group_singleton.h"

namespace marian {

void SingletonGraph::setScheduler(Ptr<Scheduler> scheduler) {
  scheduler_ = scheduler;
  // optimizer has to be registered last to see changes of learning rate
  scheduler_->registerTrainingObserver(scheduler_);
  scheduler_->registerTrainingObserver(opt_);
}

void SingletonGraph::execute(Ptr<data::Batch> batch) {

  auto loss = builder_->build(graph_, batch);
  if(costScaleFactor_ != 1.f) {
    // for fp16 training, it's ok to go out of scope, we do not use the scaled version for anything
    auto costNode = loss->loss() * costScaleFactor_;
  }

  graph_->forward();
  graph_->backward();

  bool noNanOrInf = true;
  if(costScale_) {
    // Are there NaNs in the gradient?
    bool hasNan = false, hasInf = false;
    IsNan(graph_->params()->grads(), graph_->allocator(), hasNan, hasInf);
    noNanOrInf = !(hasNan || hasInf);

    if(!noNanOrInf) // there was a NaN, decrease cost-scaling
      GraphGroup::decreaseCostScaleFactor();
  }

  if(noNanOrInf) // skip update if NaN was seen @TODO: repeat instead with smaller factor?
    opt_->update(graph_->params()->vals(),
                 graph_->params()->grads(),
                 OptimizerBase::mbSizeNotProvided,
                 costScaleFactor_);

  if(scheduler_) {
    scheduler_->update(*loss, batch);

    if(scheduler_->validating()) {
      swapWithSmoothed({graph_}, {opt_});
      scheduler_->validate({graph_});
      swapWithOriginal({graph_}, {opt_});
    }

    if(scheduler_->saving())
      this->save();
  }

  if(noNanOrInf)
    GraphGroup::increaseCostScaleFactor();
}

}  // namespace marian

