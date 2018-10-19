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
class IckyPtr {
private:
    typedef IckyPtr this_type;

public:
    typedef T element_type;

    IckyPtr() : ptr_(0), counter_(new Counter()) {};

    IckyPtr(T* p)
    : ptr_(p), counter_(new Counter()) {
       if(ptr_ != 0)
         counter_->increase();
    }

    template<class Y>
    IckyPtr(const IckyPtr<Y>& rhs)
    : ptr_(rhs.get()), counter_(rhs.getCounter()) {
      if(ptr_ != 0)
         counter_->increase();
    }

    IckyPtr(const IckyPtr& rhs)
    : ptr_(rhs.ptr_), counter_(rhs.counter_) {
      if(ptr_ != 0)
        counter_->increase();
    }

    ~IckyPtr() {
      if(ptr_ != 0) {
        if(counter_->decrease() == 0) {
          delete ptr_;
          delete counter_;
        }
      }
    }

    IckyPtr(IckyPtr&& rhs)
    : ptr_(rhs.ptr_), counter_(rhs.counter_) {
      rhs.ptr_ = 0;
      rhs.counter_ = 0;
    }

    IckyPtr& operator=(IckyPtr&& rhs) {
      this_type(static_cast<IckyPtr&&>(rhs)).swap(*this);
      return *this;
    }

    IckyPtr& operator=(const IckyPtr& rhs) {
      this_type(rhs).swap(*this);
      return *this;
    }

    template<class Y>
    IckyPtr& operator=(const IckyPtr<Y>& rhs) {
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

    void swap(IckyPtr& rhs) {
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
bool operator==(const IckyPtr<T>& a, const IckyPtr<U>& b) {
  return a.get() == b.get();
}

template<class T, class U>
bool operator!=(const IckyPtr<T>& a, const IckyPtr<U>& b) {
  return a.get() != b.get();
}

template<class T>
bool operator==(const IckyPtr<T>& a, T* b) {
  return a.get() == b;
}

template<class T>
bool operator!=(const IckyPtr<T>& a, T* b) {
  return a.get() != b;
}

template<class T>
bool operator==(const IckyPtr<T>& a, std::nullptr_t) {
  return a.get() == 0;
}

template<class T>
bool operator!=(const IckyPtr<T>& a, std::nullptr_t) {
  return a.get() != 0;
}

template<class T>
bool operator==(T* a, const IckyPtr<T>& b) {
  return b.get();
}

template<class T>
bool operator!=(T* a, const IckyPtr<T>& b) {
  return b.get();
}

template<class T, class U>
bool operator<(const IckyPtr<T>& a, const IckyPtr<U>& b) {
  return std::less<T*>()(a.get(), b.get());
}

template<class T>
void swap(IckyPtr<T> & a, IckyPtr<T> & b) {
  a.swap(b);
}

template<class E, class T, class Y>
std::basic_ostream<E, T>& operator<<(std::basic_ostream<E, T>& os, const IckyPtr<Y>& p) {
  os << p.get();
  return os;
}

namespace std {
  template<class T>
  T* get_pointer(const IckyPtr<T>& p) {
    return p.get();
  }

  template<class T, class U>
  IckyPtr<T> static_pointer_cast(const IckyPtr<U>& p) {
    return static_cast<T*>(p.get());
  }

  template<class T, class U>
  IckyPtr<T> const_pointer_cast(const IckyPtr<U>& p) {
    return const_cast<T*>(p.get());
  }

  template<class T, class U>
  IckyPtr<T> dynamic_pointer_cast(const IckyPtr<U>& p) {
    return dynamic_cast<T*>(p.get());
  }

  template <class T> struct hash<IckyPtr<T>> {
    size_t operator()(const IckyPtr<T>& x) const {
      std::hash<size_t> hasher;
      return hasher((size_t)x.get());
    }
  };
}

