#pragma once

#include <cassert>
#include <iostream>

// Smart pointer class for small objects with reference counting but no thread-safety.
// Inspired by boost::intrusive_ptr<T>.

// Compared to std::shared_ptr this is small and cheap to construct and destroy.
// Does not hold the counter, the pointed to class `T` needs to add
// ENABLE_STICKY_PTR(T) into the body of the class (private section). This adds
// the reference counters and count manipulation functions to the class.


#define ENABLE_STICKY_PTR(type)           \
  size_t references_{0};                  \
                                          \
  friend void stickyPtrAddRef(type* x) {  \
    if(x != 0)                            \
      ++x->references_;                   \
  }                                       \
                                          \
  friend void stickyPtrRelease(type* x) { \
    if(x != 0 && --x->references_ == 0)   \
      delete x;                           \
  }                                       \


template<class T>
class StickyPtr {
private:
  typedef StickyPtr this_type;

public:
  typedef T element_type;

  StickyPtr() : ptr_(0) {};

  StickyPtr(T* p)
  : ptr_(p) {
      if(ptr_ != 0)
        stickyPtrAddRef(ptr_);
  }

  template<class Y>
  StickyPtr(const StickyPtr<Y>& rhs)
  : ptr_(rhs.get()) {
    if(ptr_ != 0)
      stickyPtrAddRef(ptr_);
  }

  StickyPtr(const StickyPtr& rhs)
  : ptr_(rhs.ptr_) {
    if(ptr_ != 0)
      stickyPtrAddRef(ptr_);
  }

  ~StickyPtr() {
    if(ptr_ != 0)
      stickyPtrRelease(ptr_);
  }

  StickyPtr(StickyPtr&& rhs)
  : ptr_(rhs.ptr_) {
    rhs.ptr_ = 0;
  }

  StickyPtr& operator=(StickyPtr&& rhs) {
    this_type(static_cast<StickyPtr&&>(rhs)).swap(*this);
    return *this;
  }

  StickyPtr& operator=(const StickyPtr& rhs) {
    this_type(rhs).swap(*this);
    return *this;
  }

  template<class Y>
  StickyPtr& operator=(const StickyPtr<Y>& rhs) {
    this_type(rhs).swap(*this);
    return *this;
  }

  void reset() {
    this_type().swap(*this);
  }

  void reset(T* rhs) {
    this_type(rhs).swap(*this);
  }

  T* get() const {
    return ptr_;
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

  void swap(StickyPtr& rhs) {
    T* tmp = ptr_;
    ptr_ = rhs.ptr_;
    rhs.ptr_ = tmp;
  }

private:
  T* ptr_;
};

template<class T, class U>
bool operator==(const StickyPtr<T>& a, const StickyPtr<U>& b) {
  return a.get() == b.get();
}

template<class T, class U>
bool operator!=(const StickyPtr<T>& a, const StickyPtr<U>& b) {
  return a.get() != b.get();
}

template<class T>
bool operator==(const StickyPtr<T>& a, T* b) {
  return a.get() == b;
}

template<class T>
bool operator!=(const StickyPtr<T>& a, T* b) {
  return a.get() != b;
}

template<class T>
bool operator==(const StickyPtr<T>& a, std::nullptr_t) {
  return a.get() == 0;
}

template<class T>
bool operator!=(const StickyPtr<T>& a, std::nullptr_t) {
  return a.get() != 0;
}

template<class T>
bool operator==(T* a, const StickyPtr<T>& b) {
  return b.get();
}

template<class T>
bool operator!=(T* a, const StickyPtr<T>& b) {
  return b.get();
}

template<class T, class U>
bool operator<(const StickyPtr<T>& a, const StickyPtr<U>& b) {
  return std::less<T*>()(a.get(), b.get());
}

template<class T>
void swap(StickyPtr<T> & a, StickyPtr<T> & b) {
  a.swap(b);
}

template<class E, class T, class Y>
std::basic_ostream<E, T>& operator<<(std::basic_ostream<E, T>& os, const StickyPtr<Y>& p) {
  os << p.get();
  return os;
}

namespace std {
  template<class T>
  T* get_pointer(const StickyPtr<T>& p) {
    return p.get();
  }

  template<class T, class U>
  StickyPtr<T> static_pointer_cast(const StickyPtr<U>& p) {
    return static_cast<T*>(p.get());
  }

  template<class T, class U>
  StickyPtr<T> const_pointer_cast(const StickyPtr<U>& p) {
    return const_cast<T*>(p.get());
  }

  template<class T, class U>
  StickyPtr<T> dynamic_pointer_cast(const StickyPtr<U>& p) {
    return dynamic_cast<T*>(p.get());
  }

  template <class T> struct hash<StickyPtr<T>> {
    size_t operator()(const StickyPtr<T>& x) const {
      std::hash<size_t> hasher;
      return hasher((size_t)x.get());
    }
  };
}


