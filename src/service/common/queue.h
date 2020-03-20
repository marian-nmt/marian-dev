// -*- mode: c++; indent-tabs-mode: nil; tab-width: 2 -*-
// Generic thread-safe, locking queue with timeout for push and pop.

#pragma once

#include<chrono>
#include<condition_variable>
#include<deque>
#include<iostream>
#include<mutex>
#include<vector>

#include"common/logging.h"

namespace marian {
namespace server {

template<typename item_t>
class Queue {
  size_t capacity_; // limit capacity; Queue blocks on push when full
  std::deque<item_t> queue_; // the actual queue
  std::mutex mutex_; // mutex for controlling access
  std::condition_variable ready_;     // signal that items are available
  std::condition_variable have_room_; // signal that items can be pushed

  // The status flags below control queue behavior.
  // They are set by close() and cancel().
  // Note: queue_active_ == false implies queue_open_ == false
  bool queue_open_;   // queue allows pushing of new items
  bool queue_active_; // queue is open or still has items in it

public:
  enum STATUS_CODE { SUCCESS=0, EMPTY=1, FULL=2, CLOSED=4, ERROR=8 };
  typedef typename std::chrono::duration<double> timeout_t;
  Queue(size_t capacity=0);
  STATUS_CODE pop(item_t& item, timeout_t timeout = timeout_t(0));
  STATUS_CODE push(item_t item, timeout_t timeout = timeout_t(0),
                   Logger* logger=NULL);
  void close();  // prohibit push, allow pop until empty
  void cancel(); // prohibit push, drop everything in the queue
  bool ready() const; // ready to accept more?
  size_t size() const { return queue_.size(); }
};

template<typename item_t>
Queue<item_t>::
Queue(size_t capacity)
  : capacity_(capacity), queue_open_(true), queue_active_(true)
{ }

template<typename item_t>
bool
Queue<item_t>::
ready() const { // ready to accept more items?
  return queue_open_ && (capacity_ == 0 or queue_.size() < capacity_);
}


template<typename item_t>
typename Queue<item_t>::STATUS_CODE
Queue<item_t>::
push(item_t item, timeout_t timeout, Logger* logger) {
  std::unique_lock<std::mutex> lock(mutex_);

  // If the queue has limited capacity and is full, wait for for a free slot
  while (queue_open_ and capacity_ and queue_.size() == capacity_) {
    if (logger)
      (*logger)->trace("Waiting for slot for item {}", item);
    if (timeout.count() > 0)
      have_room_.wait_for(lock, timeout);
    else
      have_room_.wait(lock); // NOT recommended, you may deadlock
  }

  if (logger)
    (*logger)->debug("Got slot for item {}", item);
  if (!queue_open_) {
    if (logger) (*logger)->debug("Queue is closed");
    return CLOSED; // queue is closed for new business
  }

  queue_.push_back(std::move(item));
  if (logger)
    (*logger)->debug("Queue now has {} item{}.", queue_.size(),
                  queue_.size() ==  1 ? "" : "s");
  lock.unlock();
  ready_.notify_one();
  return SUCCESS;
}

template<typename item_t>
typename Queue<item_t>::STATUS_CODE
Queue<item_t>::
pop(item_t& item, timeout_t timeout) {
  std::unique_lock<std::mutex> lock(mutex_);
  if(queue_.empty()) {
    ready_.wait_for(lock, timeout);
  }
  if (!queue_active_) { // someone shut down the queue
    return CLOSED;
  }

  if (queue_.empty()) { // timed out, queue is empty
    return queue_open_ ? EMPTY : CLOSED;
  }

  item = std::move(queue_.front());
  queue_.pop_front();
  lock.unlock();
  have_room_.notify_one();
  return SUCCESS;
}

template<typename item_t>
void
Queue<item_t>::
close() {
  std::lock_guard<std::mutex> lock(mutex_);
  queue_open_ = false;
}

template<typename item_t>
void
Queue<item_t>::
cancel() {
  std::lock_guard<std::mutex> lock(mutex_);
  queue_open_ = false;
  queue_active_ = false;
  queue_.clear();
}

}// end of namespace marian::server
}// end of namespace marian
