// -*- mode: c++; indent-tabs-mode: nil; tab-width: 2 -*-
#include "translation_job.h"
namespace marian {
namespace server {
std::atomic_ullong Job::job_ctr_{0};

Job::Job(uint64_t ejid, const std::string text,
         const size_t num_nbest, const size_t pri)
  : unique_id(++job_ctr_),
    external_id(ejid),
    priority(pri),
    input({text}),
    nbestlist_size(num_nbest) {
  gettimeofday(&created.first, &created.second);
}

void Job::dequeued() {
  gettimeofday(&started.first, &started.second);
}

void Job::finish(Ptr<const History> h, const bool R2L, const Vocab& V) {
  history = h;
  auto nbest_histories = h->nBest(nbestlist_size,true);
  for (auto& hyp: nbest_histories) {
    auto& snt = std::get<0>(hyp);
    if (R2L) std::reverse(snt.begin(), snt.end());
    nbest.push_back(std::make_pair(std::get<2>(hyp), V.decode(snt)));
  }
  if (nbest.size())
    translation = nbest[0].second;
  gettimeofday(&finished.first, &finished.second);
}

float Job::totalTime() const {
  struct timeval t;
  timeval_subtract_(t, finished.first, created.first);
  return t.tv_sec + t.tv_usec/1000000.;
}

float Job::timeBeforeQueue() const {
  struct timeval t;
  timeval_subtract_(t, queued.first, created.first);
  return t.tv_sec + t.tv_usec/1000000.;
}

float Job::timeInQueue() const {
  struct timeval t;
  timeval_subtract_(t, started.first, queued.first);
  return t.tv_sec + t.tv_usec/1000000.;
}

float Job::translationTime() const {
  struct timeval t;
  timeval_subtract_(t, finished.first, started.first);
  return t.tv_sec + t.tv_usec/1000000.;
}

}} // end of namespace marian::server
