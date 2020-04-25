// -*- mode: c++; indent-tabs-mode: nil; tab-width: 2 -*-
#pragma once

#include <functional>
#include <string>
#include <vector>
#include <map>

#include "common/logging.h"
#include "data/batch_generator.h"
#include "data/corpus.h"
#include "data/shortlist.h"
// #include "data/text_input.h"

#include "translator/history.h"
// #include "translator/output_collector.h"
// #include "translator/output_printer.h"

// #include "models/model_task.h"
#include "translator/scorers.h"
#include <thread>
#include "data/shortlist.h"
#include "queued_input.h"

extern Logger logger;

namespace marian {
namespace server {

class TranslationWorker
{
private:
  DeviceId device_;
  std::unique_ptr<std::thread> thread_;
  // we use a pointer to the worker thread so that we have an easy way of
  // ensuring the worker is run exactly once; we don't want two workers
  // on the same device.

  Ptr<data::QueuedInput> job_queue_;
  std::function<void (Ptr<History const>)> callback_;
  Ptr<Options> options_;
  Ptr<ExpressionGraph> graph_;
  std::vector<Ptr<Vocab const>> vocabs_;
  std::vector<Ptr<Scorer>> scorers_;
  Ptr<data::ShortlistGenerator const> slgen_;
  bool keep_going_{true};

  void init_();

  template<typename Search>
  void run_() {
    init_();
    LOG(info,"Worker {} is ready.", std::string(device_));
    keep_going_ = true;
    while(keep_going_) { // will be set to false by stop()
      data::BatchGenerator<data::QueuedInput> bgen(job_queue_, options_);
      bgen.prepare();
      auto trgVocab = vocabs_.back();
      for (auto b: bgen) {
        auto search = New<Search>(options_, scorers_, trgVocab);
        auto histories = search->search(graph_, b);
        for (auto h: histories)
          callback_(h);
      }
    }
  }

public:
  TranslationWorker(DeviceId const device,
                    std::vector<Ptr<Vocab const>> vocabs,
                    Ptr<data::ShortlistGenerator const> slgen,
                    Ptr<data::QueuedInput> job_queue,
                    std::function<void (Ptr<History const>)> callback,
                    Ptr<Options> options);

  template<typename Search>
  void start() {
    ABORT_IF(thread_ != NULL, "Don't call start on a running worker!");
    thread_.reset(new std::thread([this]{ this->run_<Search>(); }));
  }

  void stop();
  void join();
}; // end of class TranslationWorker

} // end of namespace marian::server
} // end of namespace marian
