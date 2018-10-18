#pragma once

#include <cassert>
#include <iostream>

// Smart pointer class for small objects with reference counting but no thread-safety.
// Inspired by boost::intrusive_ptr<T>.

// Compared to std::shared_ptr this is small and cheap to construct and destroy. 
// Does not hold the counter, the pointed to class `T` needs to do that:
// Either by implementing `size_t references_` and befriending `StupidPtr<T>` or by inheriting from
// `EnableStupidPtr<T>` for class of type `T`.
// @TODO check cost for inheriting from `EnableStupidPtr<T>`

template<class T> 
class StupidPtr {
private:
    typedef StupidPtr this_type;

    inline void stupidPtrAddRef(T* x){
      ++x->references_;
    }

    inline void stupidPtrRelease(T* x) {
      if(--x->references_ == 0) 
        delete x;
    }

public:
    typedef T element_type;

    StupidPtr() : ptr_(0) {};

    StupidPtr(T* p, bool add_ref = true) 
    : ptr_(p) {
       if(ptr_ != 0 && add_ref) 
         stupidPtrAddRef(ptr_);
    }

    template<class Y> 
    StupidPtr(const StupidPtr<Y>& rhs)
    : ptr_(rhs.get()) {
      if(ptr_ != 0)
        stupidPtrAddRef(ptr_);
    }

    StupidPtr(const StupidPtr& rhs)
    : ptr_(rhs.ptr_) {
      if(ptr_ != 0)
        stupidPtrAddRef(ptr_);
    }

    ~StupidPtr() {
      if(ptr_ != 0) 
        stupidPtrRelease(ptr_);
    }

    StupidPtr(StupidPtr&& rhs) 
    : ptr_(rhs.ptr_) {
      rhs.ptr_ = 0;
    }

    StupidPtr& operator=(StupidPtr&& rhs) {
      this_type(static_cast<StupidPtr&&>(rhs)).swap(*this);
      return *this;
    }

    StupidPtr& operator=(const StupidPtr& rhs) {
      this_type(rhs).swap(*this);
      return *this;
    }

    template<class Y> 
    StupidPtr& operator=(const StupidPtr<Y>& rhs) {
      this_type(rhs).swap(*this);
      return *this;
    }

    void reset() {
      this_type().swap(*this);
    }

    void reset(T* rhs) {
      this_type(rhs).swap(*this);
    }

    void reset(T* rhs, bool add_ref) {
      this_type(rhs, add_ref).swap(*this);
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

    void swap(StupidPtr& rhs) {
      T* tmp = ptr_;
      ptr_ = rhs.ptr_;
      rhs.ptr_ = tmp;
    }

private:
  T* ptr_;
};

template<class T, class U>
bool operator==(const StupidPtr<T>& a, const StupidPtr<U>& b) { 
  return a.get() == b.get();
}

template<class T, class U>
bool operator!=(const StupidPtr<T>& a, const StupidPtr<U>& b) {
  return a.get() != b.get();
}

template<class T>
bool operator==(const StupidPtr<T>& a, T* b) {
  return a.get() == b;
}

template<class T>
bool operator!=(const StupidPtr<T>& a, T* b) {
  return a.get() != b;
}

template<class T>
bool operator==(const StupidPtr<T>& a, std::nullptr_t) {
  return a.get() == 0;
}

template<class T>
bool operator!=(const StupidPtr<T>& a, std::nullptr_t) {
  return a.get() != 0;
}

template<class T>
bool operator==(T* a, const StupidPtr<T>& b) {
  return b.get();
}

template<class T>
bool operator!=(T* a, const StupidPtr<T>& b) {
  return b.get();
}

template<class T, class U>
bool operator<(const StupidPtr<T>& a, const StupidPtr<U>& b) {
  return std::less<T*>()(a.get(), b.get());
}

template<class T> 
void swap(StupidPtr<T> & a, StupidPtr<T> & b) {
  a.swap(b);
}

template<class T> 
T* get_pointer(const StupidPtr<T>& p) {
  return p.get();
}

template<class T, class U>
StupidPtr<T> static_pointer_cast(const StupidPtr<U>& p) {
  return static_cast<T*>(p.get());
}

template<class T, class U>
StupidPtr<T> const_pointer_cast(const StupidPtr<U>& p) {
  return const_cast<T*>(p.get());
}

template<class T, class U>
StupidPtr<T> dynamic_pointer_cast(const StupidPtr<U>& p) {
   return dynamic_cast<T*>(p.get());
}

template<class E, class T, class Y>
std::basic_ostream<E, T>& operator<<(std::basic_ostream<E, T>& os, const StupidPtr<Y>& p) {
  os << p.get();
  return os;
}

template <class T>
class EnableStupidPtr {
private:
  size_t references_{0};
  friend StupidPtr<T>;

  StupidPtr<T> stupidFromThis() {
    return (T*)this;
  }
};
