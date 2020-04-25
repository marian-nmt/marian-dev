#include "translation_worker.h"
#include "translator/scorers.h"
namespace marian {
namespace server {

void TranslationWorker::init_() {
  graph_ = New<ExpressionGraph>(true); // always optimize
  graph_->setDevice(device_);
  graph_->getBackend()->setClip(options_->get<float>("clip-gemm"));
  graph_->reserveWorkspaceMB(options_->get<size_t>("workspace"));
  scorers_ = createScorers(options_);
  for (auto s: scorers_) {
    // Why aren't these steps part of createScorers?
    // i.e., createScorers(options_, graph_, shortlistGenerator_) [UG]
    s->init(graph_);
    if (slgen_) s->setShortlistGenerator(slgen_);
  }
  graph_->forward();
  // Is there a particular reason that graph_->forward() happens after
  // initialization of the scorers? It would improve code readability
  // to do this before scorer initialization. Logical flow: first
  // set up graph, then set up scorers. [UG]
}

TranslationWorker::
TranslationWorker(DeviceId const device,
                  std::vector<Ptr<Vocab const>> vocabs,
                  Ptr<data::ShortlistGenerator const> slgen,
                  Ptr<data::QueuedInput> job_queue,
                  std::function<void (Ptr<History const>)> callback,
                  Ptr<Options> options)
  : device_(device), job_queue_(job_queue), callback_(callback),
    options_(options), vocabs_(vocabs), slgen_(slgen)
{ }

void TranslationWorker::stop() {
  keep_going_ = false;
}

void TranslationWorker::join() {
  thread_->join();
  thread_.reset();
}

}
}
