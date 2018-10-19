#pragma once

#include "common/definitions.h"
#include "common/shape.h"
#include "common/types.h"
#include "tensors/backend.h"
#include "tensors/memory_piece.h"
#ifdef CUDA_FOUND
#include "tensors/gpu/algorithm.h"
#endif

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

namespace marian {

class TensorBase {
private:
  MemoryPiece::PtrType memory_;
  Shape shape_;
  Type type_{Type::float32};
  Ptr<Backend> backend_;

  ENABLE_INTRUSIVE_PTR(TensorBase)

  TensorBase(MemoryPiece::PtrType memory,
             Shape shape,
             Type type,
             Ptr<Backend> backend)
      : memory_(memory), shape_(shape), type_(type), backend_(backend) {}

  TensorBase(MemoryPiece::PtrType memory, Shape shape, Ptr<Backend> backend)
      : memory_(memory),
        shape_(shape),
        type_(Type::float32),
        backend_(backend) {}

public:
  // Use this whenever pointing to MemoryPiece
  typedef IPtr<TensorBase> PtrType;

  // Use this whenever creating a pointer to MemoryPiece
  template <class ...Args>
  static PtrType New(Args&& ...args) {
    return PtrType(new TensorBase(std::forward<Args>(args)...));
  }

  ~TensorBase() {}

  virtual void reset(MemoryPiece::PtrType memory) { memory_ = memory; }

  virtual MemoryPiece::PtrType memory() { return memory_; }

  virtual Type type() { return type_; }

  virtual Shape& shape() { return shape_; }

  virtual float* data() { return memory_->data<float>(); }

  template <typename T>
  T* data() {
    return memory_->data<T>();
  }

  virtual size_t size() { return shape_.elements(); }

  virtual float scalar() {
    matchOrAbort<float>(type_);
    ABORT_IF(size() != 1, "Tensor is not a scalar");
    return get(0);
  }

  template <typename T>
  T scalar() {
    matchOrAbort<T>(type_);
    ABORT_IF(size() != 1, "Tensor is not a scalar");
    return get<T>(0);
  }

  Ptr<Backend> getBackend() { return backend_; }
  DeviceId getDeviceId() { return backend_->getDeviceId(); }

  Tensor subtensor(size_t offset, size_t size) {
    auto mem = MemoryPiece::New(memory_->data() + sizeOf(type_) * offset, sizeOf(type_) * size);
    return TensorBase::New(mem, Shape{1, (int)size}, backend_);
  }

  float get(size_t i) {
    matchOrAbort<float>(type_);
    float temp = 0; // (initialize to keep compiler happy)
    if(backend_->getDeviceId().type == DeviceType::cpu) {
      std::copy(data() + i, data() + i + 1, &temp);
    }
#ifdef CUDA_FOUND
    else {
      gpu::copy(backend_, data() + i, data() + i + 1, &temp);
    }
#endif
    return temp;
  }

  template <typename T>
  T get(size_t i) {
    matchOrAbort<T>(type_);

    T temp;
    if(backend_->getDeviceId().type == DeviceType::cpu) {
      std::copy(data<T>() + i, data<T>() + i + 1, &temp);
    }
#ifdef CUDA_FOUND
    else {
      gpu::copy(backend_, data<T>() + i, data<T>() + i + 1, &temp);
    }
#endif
    return temp;
  }

  template <typename T>
  void set(size_t i, T value) {
    matchOrAbort<T>(type_);

    if(backend_->getDeviceId().type == DeviceType::cpu) {
      std::copy(&value, &value + 1, data<T>() + i);
    }
#ifdef CUDA_FOUND
    else {
      gpu::copy(backend_, &value, &value + 1, data<T>() + i);
    }
#endif
  }

  template <typename T>
  void get(std::vector<T>& v) {
    matchOrAbort<T>(type_);

    v.resize(size());
    if(backend_->getDeviceId().type == DeviceType::cpu) {
      std::copy(data<T>(), data<T>() + size(), v.data());
    }
#ifdef CUDA_FOUND
    else {
      gpu::copy(backend_, data<T>(), data<T>() + size(), v.data());
    }
#endif
  }

  template <typename T>
  void set(const T* begin, const T* end) {
    matchOrAbort<T>(type_);

    if(backend_->getDeviceId().type == DeviceType::cpu) {
      std::copy(begin, end, data<T>());
    }
#ifdef CUDA_FOUND
    else {
      gpu::copy(backend_, begin, end, data<T>());
    }
#endif
  }

  template <typename T>
  void set(const std::vector<T>& v) {
    set(v.data(), v.data() + v.size());
  }

  // For single values enable conversion to other numeric formats if possible
  template <typename T>
  void set(T value) {
    if(!matchType<T>(type_)) {
      switch(type_) {
        case Type::float32: set<float   >((float   )value); break;
        case Type::float64: set<double  >((double  )value); break;
        case Type::int8:    set<int8_t  >((int8_t  )value); break;
        case Type::int16:   set<int16_t >((int16_t )value); break;
        case Type::int32:   set<int32_t >((int32_t )value); break;
        case Type::int64:   set<int64_t >((int64_t )value); break;
        case Type::uint8:   set<uint8_t >((uint8_t )value); break;
        case Type::uint16:  set<uint16_t>((uint16_t)value); break;
        case Type::uint32:  set<uint32_t>((uint32_t)value); break;
        case Type::uint64:  set<uint64_t>((uint64_t)value); break;
        default:
          ABORT(
              "Requested type ({}) cannot be converted to underlying type ({})",
              request<float>(),
              type_);
      }
    } else {
      if(backend_->getDeviceId().type == DeviceType::cpu) {
        std::fill(data<T>(), data<T>() + size(), value);
      }
  #ifdef CUDA_FOUND
      else {
        gpu::fill(backend_, data<T>(), data<T>() + size(), value);
      }
  #endif
    }
  }

  void setSparse(const std::vector<size_t>& k, const std::vector<float>& v) {
    ABORT_IF(!matchType<float>(type_),
             "Requested type ({}) and underlying type ({}) do not match",
             request<float>(),
             type_);

    if(backend_->getDeviceId().type == DeviceType::cpu) {
      for(size_t i = 0; i < k.size(); ++i)
        data()[k[i]] = v[i];
    }
#ifdef CUDA_FOUND
    else {
      gpu::setSparse(backend_, k, v, data());
    }
#endif
  }

  void copyFrom(Tensor in) {
    // @TODO: solve this later
    ABORT_IF(!matchType<float>(type_),
             "Requested type ({}) and underlying type ({}) do not match",
             request<float>(),
             type_);

    if(in->getBackend()->getDeviceId().type == DeviceType::cpu
       && backend_->getDeviceId().type == DeviceType::cpu) {
      std::copy(in->data(), in->data() + in->size(), data());
    }
#ifdef CUDA_FOUND
    else {
      gpu::copy(backend_, in->data(), in->data() + in->size(), data());
    }
#endif
  }

  template <typename T>
  std::string debug(int precision = 8, int dispCols = 5) {
    matchOrAbort<T>(type_);

    std::stringstream strm;
    assert(shape_.size());
    strm << shape_;
    strm << " type=" << type_;
    strm << " device=" << backend_->getDeviceId();
    strm << " ptr=" << (size_t)memory_->data();
    strm << " bytes=" << memory_->size();
    strm << std::endl;

    // values
    size_t totSize = shape_.elements();
    std::vector<T> values(totSize);
    get(values);

    int colWidth  = precision + 4;
    
    if(isFloat(type_))
      strm << std::fixed << std::setprecision(precision) << std::setfill(' ');
    else
      strm << std::fixed << std::setprecision(0) << std::setfill(' ');

    for(int i = 0; i < values.size(); ++i) {
      std::vector<int> dims;
      shape().dims(i, dims);

      bool disp = true;
      for(int j = 0; j < dims.size(); ++j)
        disp = disp && (dims[j] < dispCols || dims[j] >= shape()[j] - dispCols);

      if(disp) {
        if(dims.back() == 0) {
          bool par = true;
          std::vector<std::string> p;
          for(int j = (int)dims.size() - 1; j >= 0; --j) {
            if(dims[j] != 0)
              par = false;

            p.push_back(par ? "[" : " ");
          }
          for(auto it = p.rbegin(); it != p.rend(); ++it)
            strm << *it;
          strm << " ";
        }

        strm << std::setw(colWidth);
        if(isFloat(type_)) {
          strm << (double)values[i];
        } else if(isSignedInt(type_)) {
          strm << (int64_t)values[i];
        } else {
          strm << (uint64_t)values[i];
        }
        strm << " ";

        if(dims.back() + 1 == shape().back()) {
          for(int j = (int)dims.size() - 1; j >= 0; --j) {
            if(dims[j] + 1 != shape()[j])
              break;
            strm << "]";
          }
          strm << std::endl;
        }

        bool prev = true;
        for(int j = (int)dims.size() - 1; j >= 0; --j) {
          if(j < (int)dims.size() - 1)
            prev = prev && dims[j + 1] + 1 == shape()[j + 1];
          if(prev && dims[j] + 1 == dispCols && shape()[j] > 2 * dispCols) {
            if(j < (int)dims.size() - 1)
              for(int k = 0; k <= j; ++k)
                strm << " ";
            strm << "... ";
            if(j < (int)dims.size() - 1)
              strm << std::endl;
            break;
          }
        }
      }
    }
    strm << std::endl;
    return strm.str();
  }

  std::string debug(int precision = 8, int dispCols = 5) {
    switch(type_) {
      case Type::int8:    return debug<int8_t  >(precision, dispCols);
      case Type::int16:   return debug<int16_t >(precision, dispCols);
      case Type::int32:   return debug<int32_t >(precision, dispCols);
      case Type::int64:   return debug<int64_t >(precision, dispCols);

      case Type::uint8:   return debug<uint8_t >(precision, dispCols);
      case Type::uint16:  return debug<uint16_t>(precision, dispCols);
      case Type::uint32:  return debug<uint32_t>(precision, dispCols);
      case Type::uint64:  return debug<uint64_t>(precision, dispCols);

      case Type::float32: return debug<float   >(precision, dispCols);
      case Type::float64: return debug<double  >(precision, dispCols);

      default: ABORT("Unknown type {}", type_);
    }
  }
};

typedef TensorBase::PtrType Tensor;
}  // namespace marian
