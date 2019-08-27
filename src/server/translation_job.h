// -*- mode: c++; indent-tabs-mode: nil; tab-width: 2 -*-
#pragma once
#include "translator/history.h"
#include "data/vocab.h"
#include <sys/time.h>
#include <thread>
#include "3rd_party/rapidjson/include/rapidjson/document.h"
#include "3rd_party/rapidjson/include/rapidjson/writer.h"
#include "3rd_party/rapidjson/include/rapidjson/stringbuffer.h"
#include "3rd_party/rapidjson/include/rapidjson/allocators.h"

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

  // the following auxiliary function is adapted from
  // https://www.gnu.org/software/libc/manual/html_node/Elapsed-Time.html
  int timeval_subtract_ (struct timeval& result,
                         struct timeval const& x,
                         struct timeval y) const {
    /* Perform the carry for the later subtraction by updating y. */
    if (x.tv_usec < y.tv_usec) {
      int nsec = (y.tv_usec - x.tv_usec) / 1000000 + 1;
      y.tv_usec -= 1000000 * nsec;
      y.tv_sec += nsec;
    }
    if (x.tv_usec - y.tv_usec > 1000000) {
      int nsec = (x.tv_usec - y.tv_usec) / 1000000;
      y.tv_usec += 1000000 * nsec;
      y.tv_sec -= nsec;
    }

    /* Compute the time remaining to wait.
       tv_usec is certainly positive. */
    result.tv_sec = x.tv_sec - y.tv_sec;
    result.tv_usec = x.tv_usec - y.tv_usec;

    /* Return 1 if result is negative. */
    return x.tv_sec < y.tv_sec;
  }

public:
  typedef std::pair<struct timeval, struct timezone> timestamp;
  typedef std::pair<float, std::string> nbestlist_item;
  uint64_t const unique_id; // internal job id
  uint64_t external_id{0}; // Client's job id
  int         priority{0}; // Job priority; currently not used
  timestamp    created; // time item was created
  timestamp     queued; // time item entered the queue
  timestamp    started; // time item left the queue
  timestamp   finished; // time item was translated and postprocessed
  std::vector<std::string> const input;
  size_t const nbestlist_size{1};
  std::string translation;
  std::vector<nbestlist_item> nbest;
  Ptr<History const> history;

  Ptr<Error> error;
  std::function<void (Ptr<Job>)> callback;
  rapidjson::Document request; // RapidJson Document representing the json request

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
    // auto starttime = clock();
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
    // LOG(debug,"Finishing Job took {} sec.", float(clock()-starttime)/CLOCKS_PER_SEC);
  }

  float
  totalTime() const {
    struct timeval t;
    timeval_subtract_(t, finished.first, created.first);
    return t.tv_sec + t.tv_usec/1000000.;
  }

  float
  timeBeforeQueue() const {
    struct timeval t;
    timeval_subtract_(t, queued.first, created.first);
    return t.tv_sec + t.tv_usec/1000000.;
  }

  float
  timeInQueue() const {
    struct timeval t;
    timeval_subtract_(t, started.first, queued.first);
    return t.tv_sec + t.tv_usec/1000000.;
  }

  float
  translationTime() const {
    struct timeval t;
    timeval_subtract_(t, finished.first, started.first);
    return t.tv_sec + t.tv_usec/1000000.;
  }

};
}}
