#pragma once

#include <cassert>
#include <iostream>

// Proper shared pointer with reference counting, not thread-safe.
// Maybe to be used later instead of std::shared_ptr where appropriate

class Counter {
  size_t references_{0};

public:
  size_t use_count() { return references_; };

  size_t increase() {
    return ++references_;
  }

  size_t decrease() {
    return --references_;
  }
};

template<class T>
class CountingPtr {
private:
  typedef CountingPtr this_type;

public:
  typedef T element_type;

  CountingPtr() : ptr_(0), counter_(new Counter()) {};

  CountingPtr(T* p)
  : ptr_(p), counter_(new Counter()) {
      if(ptr_ != 0)
        counter_->increase();
  }

  template<class Y>
  CountingPtr(const CountingPtr<Y>& rhs)
  : ptr_(rhs.get()), counter_(rhs.getCounter()) {
    if(ptr_ != 0)
        counter_->increase();
  }

  CountingPtr(const CountingPtr& rhs)
  : ptr_(rhs.ptr_), counter_(rhs.counter_) {
    if(ptr_ != 0)
      counter_->increase();
  }

  ~CountingPtr() {
    if(ptr_ != 0) {
      if(counter_->decrease() == 0) {
        delete ptr_;
        delete counter_;
        
        ptr_ = 0;
        counter_ = 0;
      }
    }
  }

  CountingPtr(CountingPtr&& rhs)
  : ptr_(rhs.ptr_), counter_(rhs.counter_) {
    rhs.ptr_ = 0;
    rhs.counter_ = 0;
  }

  CountingPtr& operator=(CountingPtr&& rhs) {
    this_type(static_cast<CountingPtr&&>(rhs)).swap(*this);
    return *this;
  }

  CountingPtr& operator=(const CountingPtr& rhs) {
    this_type(rhs).swap(*this);
    return *this;
  }

  template<class Y>
  CountingPtr& operator=(const CountingPtr<Y>& rhs) {
    this_type(rhs).swap(*this);
    return *this;
  }

  void reset() {
    this_type().swap(*this);
    delete counter_;
    counter_ = new Counter();
  }

  void reset(T* rhs) {
    this_type(rhs).swap(*this);
  }

  T* get() const {
    return ptr_;
  }

  Counter* getCounter() const {
    return counter_;
  }

  size_t use_count() const {
    return counter_->use_count();
  }

  T* detach() {
    T* ret = ptr_;
    ptr_ = 0;
    return ret;
  }

  T& operator*() const {
    assert(ptr_ != 0);
    return *ptr_;
  }

  T* operator->() const {
    assert(ptr_ != 0);
    return ptr_;
  }

  explicit operator bool() const {
    return ptr_ != 0;
  }

  bool operator!() const {
    return ptr_ == 0;
  }

  void swap(CountingPtr& rhs) {
    T* tmp = ptr_;
    ptr_ = rhs.ptr_;
    rhs.ptr_ = tmp;

    Counter* ctmp = counter_;
    counter_ = rhs.counter_;
    rhs.counter_ = ctmp;
  }

private:
  T* ptr_;
  Counter* counter_;
};

template<class T, class U>
bool operator==(const CountingPtr<T>& a, const CountingPtr<U>& b) {
  return a.get() == b.get();
}

template<class T, class U>
bool operator!=(const CountingPtr<T>& a, const CountingPtr<U>& b) {
  return a.get() != b.get();
}

template<class T>
bool operator==(const CountingPtr<T>& a, T* b) {
  return a.get() == b;
}

template<class T>
bool operator!=(const CountingPtr<T>& a, T* b) {
  return a.get() != b;
}

template<class T>
bool operator==(const CountingPtr<T>& a, std::nullptr_t) {
  return a.get() == 0;
}

template<class T>
bool operator!=(const CountingPtr<T>& a, std::nullptr_t) {
  return a.get() != 0;
}

template<class T>
bool operator==(T* a, const CountingPtr<T>& b) {
  return b.get();
}

template<class T>
bool operator!=(T* a, const CountingPtr<T>& b) {
  return b.get();
}

template<class T, class U>
bool operator<(const CountingPtr<T>& a, const CountingPtr<U>& b) {
  return std::less<T*>()(a.get(), b.get());
}

template<class T>
void swap(CountingPtr<T> & a, CountingPtr<T> & b) {
  a.swap(b);
}

template<class E, class T, class Y>
std::basic_ostream<E, T>& operator<<(std::basic_ostream<E, T>& os, const CountingPtr<Y>& p) {
  os << p.get();
  return os;
}

namespace std {
  template<class T>
  T* get_pointer(const CountingPtr<T>& p) {
    return p.get();
  }

  template<class T, class U>
  CountingPtr<T> static_pointer_cast(const CountingPtr<U>& p) {
    return static_cast<T*>(p.get());
  }

  template<class T, class U>
  CountingPtr<T> const_pointer_cast(const CountingPtr<U>& p) {
    return const_cast<T*>(p.get());
  }

  template<class T, class U>
  CountingPtr<T> dynamic_pointer_cast(const CountingPtr<U>& p) {
    return dynamic_cast<T*>(p.get());
  }

  template <class T> struct hash<CountingPtr<T>> {
    size_t operator()(const CountingPtr<T>& x) const {
      std::hash<size_t> hasher;
      return hasher((size_t)x.get());
    }
  };
}

// To be tested
// template <template <typename> class PtrType, class T>
// class WeakPtr {
// private:
//   const PtrType<T>& ptr_;

// public:
//   WeakPtr(const PtrType<T>& ptr) : ptr_(ptr) {}

//   PtrType<T> lock() {
//     return ptr_;
//   }

// };
