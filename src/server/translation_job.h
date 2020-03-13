// -*- mode: c++; indent-tabs-mode: nil; tab-width: 2 -*-
#pragma once
#include <sys/time.h>
#include <thread>
#include "data/vocab.h"
#include "translator/history.h"
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


// the following auxiliary function is adapted from
// https://www.gnu.org/software/libc/manual/html_node/Elapsed-Time.html
int timeval_subtract_ (struct timeval& result,
                       const struct timeval& x,
                       struct timeval y) {
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

class Job {
  static std::atomic_ullong job_ctr_;


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
  const std::vector<std::string> input;
  const size_t nbestlist_size{1};
  std::string translation;
  std::vector<nbestlist_item> nbest;
  Ptr<const History> history;

  Ptr<Error> error;
  std::function<void (Ptr<Job>)> callback;
  rapidjson::Document request; // RapidJson Document representing the json request

  Job(uint64_t ejid, const std::string text,
      const size_t num_nbest=0, const size_t pri=0);

  void dequeued(); // record start time
  void finish(Ptr<const History> h, const bool R2L, const Vocab& V);

  // functions for keeping track of workflow
  float totalTime() const;
  float timeBeforeQueue() const;
  float timeInQueue() const;
  float translationTime() const;

};
}}
