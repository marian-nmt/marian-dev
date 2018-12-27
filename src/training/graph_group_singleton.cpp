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
  if(costScaleFactor_ != 1.f)
    costNode = costNode * costScaleFactor_;

  // @TODO: missing delay, or use only --sync-sgd
  graph_->forward();
  float cost = costNode->scalar() / costScaleFactor_;
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
    scheduler_->update(cost, batch);

    if(scheduler_->validating()) {
      auto tempGraph = graphFromOptimizer(graph_, {opt_});
      scheduler_->validate({tempGraph});
    }

    if(scheduler_->saving())
      this->save();
  }

  if(noNanOrInf)
    GraphGroup::increaseCostScaleFactor();
}

}  // namespace marian

