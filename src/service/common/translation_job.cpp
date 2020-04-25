// -*- mode: c++; indent-tabs-mode: nil; tab-width: 2 -*-
#include "translation_job.h"
namespace marian {
namespace server {
std::atomic_ullong Job::job_ctr_{0};

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

Job::Job(uint64_t ejid, const std::string text,
         const TranslationOptions& topts, const size_t pri)
  : unique_id(++job_ctr_),
    external_id(ejid),
    priority(pri),
    input({text}),
    nbestlist_size(topts.nbest) {
  gettimeofday(&created.first, &created.second);
}

void Job::dequeued() {
  gettimeofday(&started.first, &started.second);
}

void Job::finish(Ptr<const History> h, const bool R2L, const Vocab& V) {
  history = h;
  nbest = h->nBest(nbestlist_size,true);
  for (auto& hyp: nbest) {
    auto& snt = std::get<0>(hyp);
    if (R2L) std::reverse(snt.begin(), snt.end());
  }
  if (nbest.size()) {
    translation = V.decode(std::get<0>(nbest[0]));
  }
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
