// -*- mode: c++; indent-tabs-mode: nil; tab-width: 2 -*-
#pragma once

#include <functional>
#include <string>
#include <vector>
#include <map>

#include "data/batch_generator.h"
#include "data/corpus.h"
#include "data/shortlist.h"
#include "data/text_input.h"

#include "3rd_party/threadpool.h"
#include "translator/history.h"
#include "translator/output_collector.h"
#include "translator/output_printer.h"

#include "models/model_task.h"
#include "translator/scorers.h"

#include "translation_worker.h"
#include "common/logging.h"
#include "queue.h"
#include "queued_input.h"

extern Logger logger;

namespace marian {
namespace server {

template<class Search>
class TranslationService;

template<class Search>
class TranslationWorker
{
private:
  DeviceId device_;
  data::QueuedInput& job_queue_;
  std::function<void (Ptr<History const>)> callback_;
  Ptr<Options> options_;
  Ptr<ExpressionGraph> graph_;
  std::vector<Ptr<Vocab const>> vocabs_;
  std::vector<Ptr<Scorer>> scorers_;
  Ptr<data::ShortlistGenerator const> slgen_;

  void init_() {
    bool optimize = options_->get<bool>("optimize");
    graph_ = New<ExpressionGraph>(true, optimize);
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
public:
  TranslationWorker(DeviceId const device,
                    std::vector<Ptr<Vocab const>>& vocabs,
                    Ptr<data::ShortlistGenerator const> slgen,
                    data::QueuedInput& job_queue,
                    std::function<void (Ptr<History const>)> callback,
                    Options options)
    : device_(device), vocabs_(vocabs), slgen_(slgen),
      job_queue_(job_queue), callback_(callback), options_(options)
  { }

  void run()
  {
    init_();
    while(true) {
      data::BatchGenerator<data::QueuedInput> bgen(job_queue_, options_);
      bgen.prepare(false);
      size_t i = 0;
      auto eos_id = vocabs_.back()->getEosId();
      auto unk_id = vocabs_.back()->getUnkId();
      auto trgVocab = vocabs_.back();
      for (auto b: bgen) {
        auto search = New<Search>(options_, scorers_, eos_id, unk_id);
        auto histories = search->search(graph_, b);
        for (auto h: histories)
          callback_(h);
      }
    }
  }


//   void run(JobQueue& Q)
//   {
//     std::chrono::milliseconds timeout(100);
//     // timeout: max wait on Q for more input, after that, translate
//     // partial batch instead of waiting for more input to fill it up
//     // to do: make this configurable

//     JobQueue::STATUS_CODE status;
//     Ptr<TranslationJob> job;
//     // single line of input to be processed, plus a promise for the result

//     while (true) {
//       status = Q.pop(job);
//       while (status == JobQueue::SUCCESS && batch_.size() < batch_.capacity()){
//         batch_.push_back(job);
//         status = Q.pop(job, timeout);
//       }
//       if (batch_.size()) process_batch();
//       if (status == JobQueue::CLOSED)
//         break;
//     }
//   } // end of function run
// };

// template<class Search>
// class TranslationService
// {
//   Ptr<Config> options_;
//   Ptr<Vocab> trgVocab_;
//   std::vector<Ptr<Vocab> > srcVocabs_;
//   std::vector<DeviceId> devices_;
//   friend TranslationWorker<Search>;
// public:
//   TranslationService(Ptr<config> options)
//     : options_(options)
//   {
//     options->set("inference", true);
//     auto vpaths = options_->get<std::vector<std::string>>("vocabs");
//     auto vdims  = options_->get<std::vector<int>>("dim-vocabs");

//     auto topt = New<Options>();
//     topt->merge(options_);

//     assert(vpaths.size());
//     srcVocabs.resize(vpaths.size()-1);
//     for(size_t i = 0; i < srcVocabs.size(); ++i) {
//       (srcVocabs_[i] = New<Vocab>(topt, i))->load(vpaths[i],vdims[i]);
//     }
//     (trgVocab_ = New<Vocab>(topt, vpaths.size() - 1))->load(vpaths.back());

//     // to do: add workers

};
} // end of namespace marian::server
} // end of namespace marian
