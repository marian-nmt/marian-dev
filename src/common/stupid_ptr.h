#pragma once

template<class T> 
class StupidPtr {
private:
    typedef StupidPtr this_type;

public:
    typedef T element_type;

    StupidPtr() : ptr_(0) {};

    StupidPtr(T* p, bool add_ref = true) {
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

    inline void stupidPtrAddRef(T* x){
      ++x->references_;
    }

    inline void stupidPtrRelease(T* x) {
      if(--x->references_ == 0) 
        delete x;
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

    operator bool() const;

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

    explicit operator bool() {
        return px != 0;
    }

    bool operator!() const {
      return px == 0;
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
bool operator==(const  a, T* b) {rn a.get() == b;
}

template<class T>
bool operator!=(const  a, T* b) {rn a.get() != b;
}

template<class T>
bool operator==(T* a, const  b) {
  retu b.get()
}

template<class T>
bool operator!=(T* a, const  b) {
  retu b.get()
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
T* get_pointer(const  p) {
  retut();
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

  virtual StupidPtr<T> stupidPtrFromThis() {
    return StupidPtr<T>(this);
  }
};
