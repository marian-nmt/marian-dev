// -*- mode: c++; indent-tabs-mode: nil; tab-width: 2 -*-
#pragma once

// @TODO: - priority handling of translation requests (for faster premium service)

#include "3rd_party/ssplit-cpp/src/ssplit/ssplit.h"
#include "3rd_party/threadpool.h"
#include "common/logging.h"

#include "data/batch_generator.h"
#include "data/corpus.h"
#include "data/shortlist.h"
#include "data/text_input.h"
#include "models/model_task.h"
#include "queue.h"
#include "queued_input.h"
#include <map>
#include <ctime>

#include <string>
#include "translation_worker.h"

#include "translation_job.h"
#include "translation_worker.h"
#include "translation_worker.h"
#include "translator/beam_search.h"
#include "translator/history.h"
#include "translator/output_collector.h"
#include "translator/output_printer.h"
#include "translator/scorers.h"
#include <ctime>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>

#ifdef __CUDA_ARCH__
#include <cuda.h>
#endif

extern Logger logger;

namespace marian {
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

template<class Search> class PlainTextTranslation;
template<class Search=BeamSearch> class NodeTranslation;

template<class Search>
class TranslationService {
public:
  typedef std::function<void (uint64_t ejid, Ptr<History const> h)>
  ResponseHandler;
  typedef ug::ssplit::SentenceStream::splitmode splitmode;
  typedef Search SearchType;
private:
  // Note to callback n00bs: see this:
  // https://oopscenities.net/2012/02/24/c11-stdfunction-and-stdbind/

  typedef TranslationWorker<Search>
  Worker;

  // bits and pieces for translating
  Ptr<Options> options_;
  std::vector<Ptr<Vocab const>> vocabs_;
  std::vector<Ptr<Worker>> workers_;
  Ptr<data::QueuedInput> jq_;
  Ptr<data::ShortlistGenerator const> slgen_;
  ug::ssplit::SentenceSplitter ssplit_;

  // bits and pieces for callbacks
  std::mutex lock_; // for management of pending callbacks
  typedef std::pair<Ptr<Job>, std::promise<Ptr<Job const>>> JobEntry;
  std::unordered_map<uint64_t, JobEntry> scheduled_jobs_;

  bool keep_going_{true};

  void callback_(Ptr<History const> h) {
    // This function is called by the workers once translations are available.

    JobEntry entry;
    { // remove the job / promise pair from the pool of scheduled jobs
      std::lock_guard<std::mutex> lock(lock_);
      auto m = scheduled_jobs_.find(h->GetLineNum());
      if (m == scheduled_jobs_.end()) return; // job was cancelled (not yet implemented)
      entry = std::move(m->second);
      scheduled_jobs_.erase(m);
    }

    // extract translations from history and fulfil the promise
    entry.first->finish(h, isRight2LeftDecoder(), *vocabs_.back());
    entry.first->callback(entry.first);
    entry.second.set_value(entry.first);
  }

  // When run in a docker container with nvidia docker, CUDA may or may
  // not be available, depending on how the Docker container is run, i.e.,
  // mapping host devices into the container via --gpus or not.
  // chooseDevice() checks if CUDA is available and automatically switches
  // to CPU mode if not. Without this check, Marian will crash unless
  // --cpu-threads is set explicitly.
  void chooseDevice_(Ptr<Options> options){
#ifdef __CUDA_ARCH__
    if (options->get<int>("cpu_threads",0) > 0)
      return; // nothing to worry about, user wants to use CPU anyway
    int ngpus;
    cudaError_t err = cudaGetDeviceCount(&ngpus);
    if (err != cudaSuccess){
      size_t nproc = std::thread::hardware_concurrency();
      size_t num_workers = options->get<size_t>("max_workers",nproc);
      LOG(warn, "NO GPU available, using CPU instead. "
          "Setting --cpu-threads to {}", num_workers);
      options->set("cpu_threads",num_workers);
    }
    if (options->get<int>("cpu_threads",0)){
      // use mini-batch size 1 if running on CPU
      options->set<int>("mini_batch",1);
    }
#endif
  }

public:
  TranslationService(Ptr<Options> options)
    : options_(options)
    , ssplit_(options_->get<std::string>("ssplit-prefix_file","")) {
    chooseDevice_(options);
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

  std::pair<uint64_t, std::future<Ptr<Job const>>>
  push(uint64_t ejid, std::string const& input, size_t const nbest=1,
       size_t const priority=0,
       std::function<void (Ptr<Job> j)> callback
       =[=](Ptr<Job> j){return;}) {
    auto starttime = std::clock();
    auto job = New<Job>(ejid, input, nbest, priority);
    if (input.empty()){//nothing to do
      std::promise<Ptr<Job const>> prom;
      prom.set_value(job);
      return std::make_pair(job->unique_id,prom.get_future());
    }
    else if (!jq_->push(job)) {
      job->error = New<Error>("Could not push to Queue.");
      std::promise<Ptr<Job const>> prom;
      prom.set_value(job);
      return std::make_pair(job->unique_id,prom.get_future());
    }
    job->callback = callback;
    JobEntry* entry;
    {
      std::lock_guard<std::mutex> lock(lock_);
      entry = &scheduled_jobs_[job->unique_id];
    }
    entry->first = job;
    // LOG(info, "Pushed job No {}; {} jobs queued up.",
    //     job->unique_id, jq_->size());
    auto pushtime = float(std::clock()-starttime)/CLOCKS_PER_SEC;
    LOG(debug,"[service] Pushing job took {}ms", 1000.* pushtime);
    return std::make_pair(job->unique_id, entry->second.get_future());
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
    std::vector<std::future<Ptr<Job const>>> ftrans;
    std::istringstream buf(srcText);
    std::string line;

    auto starttime = clock();
    for (size_t linectr = 0; getline(buf,line); ++linectr) {
      ftrans.push_back(push(linectr,line).second);
    }
    auto pushtime = (clock()-starttime)*1000./CLOCKS_PER_SEC;
    LOG(debug, "[service] Pushing translation job took {} msec.", pushtime);
    std::ostringstream obuf;
    for (auto& t: ftrans) {
      Ptr<Job const> j = t.get();
      LOG(debug, "[service] Translated job {} in {:.2f}/{:.2f} seconds:\n{}\n{}",
          j->unique_id, j->translationTime(), j->totalTime(), j->input[0], j->translation);
      obuf << j->translation << std::endl;
    }
    std::string ret = buf.str();
    if (ret.size() && !ends_with_eol_char_ && ret.back()=='\n')
      ret.pop_back();
    return ret;
  }
};

template<class Search>
class NodeTranslation {

  // Map from strings to sentence splitting modes.
  // paragraph and wrapped_text use linguistic sentence splitting.
  using splitmode=ug::ssplit::SentenceStream::splitmode;

  std::vector<NodeTranslation<Search>> children_;
  rapidjson::Value* node_;
  // std::vector<std::future<Ptr<Job const>>> delayed_;
  Ptr<PlainTextTranslation<Search>> translation_;
  bool ends_with_eol_char_{false};
  splitmode smode_;

  splitmode
  determine_splitmode_(rapidjson::Value* n,
                       NodeTranslation const* parent,
                       const std::string& options_field){
    if (n->IsObject()){
      auto x = n->FindMember(options_field.c_str());
      if (x != n->MemberEnd() && x->value.IsObject()){
        auto y = x->value.FindMember("input-format");
        if (y != x->value.MemberEnd() && y->value.IsString()){
          std::string m = y->value.GetString();
          if (m == "sentence")
            return (smode_ = splitmode::one_sentence_per_line);
          if (m == "paragraph")
            return (smode_ = splitmode::one_paragraph_per_line);
          if (m != "wrapped_text"){
            LOG(warn,"Ignoring unknown text input format specification: {}.",m);
          }
        }
      }
    }
    smode_ = parent ? parent->smode_ : splitmode::wrapped_text;
    return smode_;
  }

  void setOptions(rapidjson::Value* n,
                  NodeTranslation const* parent,
                  const std::string& options_field){
    determine_splitmode_(n, parent, options_field);
  }

 public:
  NodeTranslation(rapidjson::Value* n,
                  TranslationService<Search>& service,
                  std::string payload_field="text",
                  std::string options_field="options",
                  NodeTranslation* parent=NULL)
    : node_(n) {
    if (n == NULL) return; // nothing to do
    if (n->IsString()) {
      std::istringstream buf(n->GetString());
      std::string line;
      for (size_t linectr = 0; getline(buf, line); ++linectr) {
        // LOG(info,"Input: {}",line);
        auto foo = std::move(service.push(linectr,line));
        delayed_.push_back(std::move(foo.second));
        LOG(debug, "[service] Scheduled job No. {}: {}", foo.first, line);
      }
    }
    else if (n->IsString()) {
      translation_.reset(new PlainTextTranslation<Search>(n->GetString(),
                                                          service, smode_));
    }
    else if (n->IsArray()) {
      for (auto c = n->Begin(); c != n->End(); ++c){
        auto x = NodeTranslation(c, service, payload_field, options_field, this);
        children_.push_back(std::move(x));
      }
    }
  }

  void finish(rapidjson::Document::AllocatorType& alloc) {
    for (auto& c: children_) c.finish(alloc);
    if (delayed_.size()) {
      std::ostringstream buf;
      for (auto& f: delayed_) {
        Ptr<Job const> j = f.get();
        buf << j->translation << std::endl;
        LOG(debug, "[service] Translated in {:.2f}/{:.2f} seconds:\n{}\n{}",
            j->translationTime(), j->totalTime(), j->input[0], j->translation);
      }
      std::string translation = buf.str();
      if (!ends_with_eol_char_)
        translation.pop_back();
      if (node_) {
        ABORT_IF(!node_->IsString(), "Node is not a string!");
        // @TODO: We should thrown an exception here instead of aborting
        node_->SetString(translation.c_str(), translation.size(), alloc);
      }
    }
  }
};

std::string serialize(rapidjson::Document const& D) {
  // @TODO: this should be in a different namespace, maybe rapidjson
  rapidjson::StringBuffer buffer;
  buffer.Clear();
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  D.Accept(writer);
  return std::string(buffer.GetString(), buffer.GetSize());
}

void dump(rapidjson::Value& v, std::ostream& out) {
  if (v.IsString()) { out << v.GetString() << std::endl; }
  else if (v.IsArray()) { for (auto& c: v.GetArray()) dump(c,out); }
}

}} // end of namespace marian::server
