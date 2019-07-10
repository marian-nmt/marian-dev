// -*- mode: c++; indent-tabs-mode: nil; tab-width: 2 -*-
#pragma once

// @TODO: - priority handling of translation requests (for faster premium service)


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
#include <thread>

extern Logger logger;

namespace marian {

std::vector<std::pair<float,std::string>>
topTranslations(Ptr<History const> h, bool const R2L,
                Vocab const& V, size_t const n = 1) {
  auto nbest_histories = h->NBest(n,true);
  std::vector<std::pair<float,std::string>> nbest;
  for (auto& hyp: nbest_histories) {
    auto& snt = std::get<0>(hyp);
    if (R2L) std::reverse(snt.begin(), snt.end());
    nbest.push_back(std::make_pair(std::get<2>(hyp), V.decode(snt)));
  }
  return nbest;
}

namespace server {
// This should actually go into vocab.*
// Also it should be merged with the loadOrCreate code in corpus_base.cpp
// and refactored as a separate function (that then goes into vocab.*).
std::vector<Ptr<Vocab const> >
loadVocabularies(Ptr<Options> options) {
  // @TODO: parallelize vocab loading for faster startup
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

template<class Search>
class TranslationService {

  // Note to callback n00bs: see this:
  // https://oopscenities.net/2012/02/24/c11-stdfunction-and-stdbind/
  typedef std::function<void (uint64_t ejid, Ptr<History const> h)>
  ResponseHandler;

  typedef TranslationWorker<Search>
  Worker;

  // bits and pieces for translating
  Ptr<Options> options_;
  std::vector<Ptr<Vocab const>> vocabs_;
  std::vector<Ptr<Worker>> workers_;
  Ptr<data::QueuedInput> jq_;
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
      std::lock_guard<std::mutex> lock(lock_);
      auto m = callback_map_.find(h->GetLineNum());
      if (m == callback_map_.end()) return; // job was cancelled
      callback = m->second.second;
      ejid = m->second.first;
      callback_map_.erase(m);
    }
    callback(ejid,h);
  }

  bool keep_going_{true};

public:
  TranslationService(Ptr<Options> options)
    : options_(options) {
  }

  ~TranslationService() {
    stop();
  }

  void stop() {
    for (auto& w: workers_) w->stop();
    for (auto& w: workers_) w->join();
  }

  void start() {
    keep_going_ = true;
    vocabs_ = loadVocabularies(options_);

    if(options_->hasAndNotEmpty("shortlist")) {
      Ptr<data::ShortlistGenerator const> slgen;
      slgen_ = New<data::LexicalShortlistGenerator>                   \
        (options_, vocabs_.front(), vocabs_.back(),
         /*srcIdx=*/ 0, /*trgIdx=*/ 1,
         /*shared (vocab) = */ vocabs_.front() == vocabs_.back());
    }

    jq_.reset(new data::QueuedInput(vocabs_,options_));
    auto devices = Config::getDevices(options_);
    for (auto d: devices) {
      // wrap callback in a lambda function because it's a non static
      // member function:
      auto cb = [=](Ptr<History const> h) { this->callback_(h); };
      workers_.push_back(New<Worker>(d, vocabs_, slgen_, jq_, cb, options_));
      workers_.back()->start();
    }
  }

  uint64_t push(uint64_t ejid, std::string const& input, ResponseHandler cb) {
    // push new job, return internal job id
    std::lock_guard<std::mutex> lock(lock_);
    uint64_t jid = jq_->push({input});
    if (jid) {
      callback_map_[jid] = std::make_pair(ejid, cb);
    }
    LOG(info, "{} jobs queued up.", jq_->size());
    return jid;
  }

  Ptr<Vocab const> vocab(int i) const {
    if (i < 0) i += vocabs_.size();
    return vocabs_.at(i);
  }

  bool isRight2LeftDecoder() const {
    return options_->get<bool>("right-left");
  }

  std::string
  translate(std::string const& srcText) {
    // @TODO: add priority for QoS differentiation [UG]
    std::vector<std::future<std::string>> ftrans;
    std::istringstream buf(srcText);
    std::string line;
    for (size_t linectr = 0; getline(buf,line); ++linectr) {
      auto prom = New<std::promise<std::string>>();
      ftrans.push_back(prom->get_future());
      // check for empty lines
      if (line.find_first_not_of(" ") == std::string::npos) {
        prom->set_value("");
      }
      else {
        bool R2L = this->isRight2LeftDecoder();
        Ptr<Vocab const> V = vocabs_.back();
        auto cb = [prom,R2L,V](uint64_t jid, Ptr<History const> h) {
          auto response = topTranslations(h, R2L,*V,1)[0].second;
          LOG(info, "Translation of job {} is {}", jid, response);
          prom->set_value(response);
        };
        this->push(linectr,line,cb);
      }
    }
    std::ostringstream obuf;
    for (auto& t: ftrans) {
      obuf << t.get() << std::endl;
    }
    std::string translation = obuf.str();
    if (srcText.size() && srcText.back() != '\n')
      translation.pop_back();
    return translation;
  }
};

}} // end of namespace marian::server
