// -*- mode: c++; indent-tabs-mode: nil; tab-width: 2 -*-
#pragma once

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
#include <map>

#include <string>
#include "translation_worker.h"
#include <functional>
#include <mutex>

extern Logger logger;

namespace marian {
namespace server {
// This should actually go into vocab.*
// Also it should be merged with the loadOrCreate code in corpus_base.cpp
// and refactored as a separate function (that then goes into vocab.*).
std::vector<Ptr<Vocab const> >
loadVocabularies(Ptr<Options> options) {
  auto vfiles = options->get<std::vector<std::string>>("vocabs");
  // with the current setup, we need at least two vocabs: src and trg
  ABORT_IF(vfiles.size() < 2, "Insufficient number of vocabularies.");
  std::vector<Ptr<Vocab const> > vocabs(vfiles.size());
  std::unordered_map<std::string,Ptr<Vocab>> vmap;
  for (size_t i = 0; i < vocabs.size(); ++i) {
    auto m = vmap.emplace(std::make_pair(vfiles[i],Ptr<Vocab>()));
    if (m.second) { // new: load the vocab
      m.first->second = New<Vocab>(options, i);
      m.first->second->load(vfiles[i]);
    }
    vocabs[i] = m.first->second;
  }
  return vocabs;
}

// class PrintTranslationsImmediately {
//   // very simple response handler that prints a response immediately
//   // to stdout
//   Ptr<Options> options_;
//   bool reverse_;
//   PrintTranslationsImmediately(Ptr<Options> options)
//     : options_(options)
//     , reverse_(options->get<bool>("right-left"))
//   { }

// public:
//   void operator()(History const&) {

//   }
// };

template<class Search>
class TranslationService {

  // Note to callback n00bs: see this:
  // https://oopscenities.net/2012/02/24/c11-stdfunction-and-stdbind/
  typedef std::function<void (uint64_t ejid, Ptr<History const> h)> ResponseHandler;
  typedef TranslationWorker<Search> Worker;

  // bits and pieces for translating
  Ptr<Options> options_;
  std::vector<Ptr<Vocab const>> vocabs_;
  std::vector<Ptr<Worker>> workers_;
  data::QueuedInput jq_;
  Ptr<data::ShortlistGenerator const> slgen_;

  // bits and pieces for callbacks
  std::mutex lock_; // for management of pending callbacks
  std::unordered_map<uint64_t, std::pair<uint64_t, ResponseHandler>> callback_map_;

  void callback_(Ptr<History const> h) {
    // This function is called by the workers once translations are available.
    // It routes histories back to the original client together with the external
    // job ids.

    ResponseHandler callback;
    uint64_t ejid;
    { // scope for lock-guard
      std::lock_guard<std::mutex> lock;
      auto m = callback_map_.find(h->GetLineNum());
      if (m == callback_map_.end()) return; // job was cancelled
      callback = m->second.second;
      ejid = m->second.first;
      callback_map_.erase(m);
    }
    callback(ejid,h);
  }

public:
  TranslationService(Ptr<Options> options)
    : options_(options) {
  }


  void start() {
    vocabs_ = loadVocabularies(options_);

    if(options_->hasAndNotEmpty("shortlist")) {
      Ptr<data::ShortlistGenerator const> slgen;
      slgen_ = New<data::LexicalShortlistGenerator>                   \
        (options_, vocabs_.front(), vocabs_.back(), /*srcIdx=*/ 0, /*trgIdx=*/ 1,
         /*shared (vocab) = */ vocabs_.front() == vocabs_.back());
    }

    auto devices = Config::getDevices(options_);
    for (auto d: devices) {
      workers_.push_back(New<Worker>(d, vocabs_, slgen_, jq_,
                                     callback_, options_));
    }
  }

  uint64_t push(uint64_t ejid, std::string const& input, ResponseHandler cb) {
    // push new job, return internal job id
    std::lock_guard<std::mutex> lock(lock_);
    uint64_t jid = jq_.push({input});
    if (jid) {
      callback_[jid] = std::make_pair(ejid, cb);
    }
    return jid;
  }
};

}} // end of namespace marian::server
