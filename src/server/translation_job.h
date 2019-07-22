// -*- mode: c++; indent-tabs-mode: nil; tab-width: 2 -*-
#pragma once
#include "translator/history.h"
#include "data/vocab.h"
#include <sys/time.h>
#include <thread>

namespace marian {
namespace server {

class Error {
  std::string errmsg_;
public:
  Error(std::string const& msg) : errmsg_(msg) {}
  std::string const& str() { return errmsg_; }
};



class Job {
  static std::atomic_ullong job_ctr_;
public:
  typedef std::pair<struct timeval, struct timezone> timestamp;
  typedef std::pair<float, std::string> nbestlist_item;
  uint64_t const unique_id; // internal job id
  uint64_t external_id{0}; // Client's job id
  int         priority{0}; // Job priority; currently not used
  timestamp    created; // time item entered the queue
  timestamp    started; // time item left the queue
  timestamp   finished; // time item was translated and postprocessed

  std::vector<std::string> const input;
  size_t const nbestlist_size{1};
  std::string translation;
  std::vector<nbestlist_item> nbest;
  Ptr<History const> history;

  Ptr<Error> error;
  std::function<void (Ptr<Job>)> callback;

  Job(uint64_t ejid, std::string const text,
      size_t const num_nbest=0, size_t const pri=0)
    : unique_id(++job_ctr_), external_id(ejid), priority(pri), input({text}),
      nbestlist_size(num_nbest)
  {
    gettimeofday(&created.first, &created.second);
  }

  void dequeued() {
    gettimeofday(&started.first, &started.second);
  }

  void
  finish(Ptr<History const> h, bool const R2L, Vocab const& V)
  {
    history = h;
    auto nbest_histories = h->NBest(nbestlist_size,true);
    for (auto& hyp: nbest_histories) {
      auto& snt = std::get<0>(hyp);
      if (R2L) std::reverse(snt.begin(), snt.end());
      nbest.push_back(std::make_pair(std::get<2>(hyp), V.decode(snt)));
    }
    if (nbest.size())
      translation = nbest[0].second;
    gettimeofday(&finished.first, &finished.second);
  }

};
}}
