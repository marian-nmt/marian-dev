#include "tensors/tensor.h"
#include "tensors/tensor_operators.h"

namespace marian {

template <typename T>
std::string TensorBase::debug(int precision, int dispCols) {
  // values
  size_t totSize = shape_.elements();
  std::vector<T> values(totSize);

  get(values);

  std::stringstream strm;
  assert(shape_.size());
  strm << shape_;
  strm << " type=" << type_;
  strm << " device=" << backend_->getDeviceId();
  strm << " ptr=" << (size_t)memory_->data();
  strm << " bytes=" << memory_->size();
  strm << std::endl;

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

template std::string TensorBase::debug<float>(int, int);
template std::string TensorBase::debug<float16>(int, int);
template std::string TensorBase::debug<double>(int, int);

template std::string TensorBase::debug<uint8_t>(int, int);
template std::string TensorBase::debug<uint16_t>(int, int);
template std::string TensorBase::debug<uint32_t>(int, int);
template std::string TensorBase::debug<uint64_t>(int, int);

template std::string TensorBase::debug<int8_t>(int, int);
template std::string TensorBase::debug<int16_t>(int, int);
template std::string TensorBase::debug<int32_t>(int, int);
template std::string TensorBase::debug<int64_t>(int, int);

const io::Item TensorBase::toItem(const std::string& name) {
  std::vector<char> bytes(memory_->size());
  copy(backend_,
        memory_->data<char>(),
        memory_->data<char>() + memory_->size(),
        bytes.data());

  io::Item item;
  item.name = name;
  item.shape = shape_;

  // Model files are saved as tensors of float. Other floating point
  // types will be converted first. 
  if(type_ == Type::float32) {
    item.type = type_;
    // Use the actual memory as this will be aligned and padded.
    // When memory mapping this is required. Shape keeps track of
    // tensor size. Saving to *.npz will cut to size.
    item.bytes.swap(bytes);
  } else if(type_ == Type::float16) {
    // converting to float
    item.type = Type::float32;
    item.bytes.resize(size() * sizeOf(item.type));

    const float16* beg16 = (const float16*)bytes.data();
    const float16* end16 = beg16 + size();
    float* beg32 = (float*)item.bytes.data();

    // This performs a conversion due to different pointer
    std::copy(beg16, end16, beg32); 
  } else {
    ABORT("Other types are currently not supported for saving");
  }

  return item;
}

}  // namespace marian

