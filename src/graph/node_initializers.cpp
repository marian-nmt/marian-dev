#include "graph/node_initializers.h"
#include "layers/word2vec_reader.h"
#include "tensors/tensor_operators.h"

#include <stdint.h>
#include <algorithm>
#include <iterator>
#include <random>

namespace marian {

namespace inits {

void LambdaInitConvert::operator()(Tensor tensor) {
      if(tensor->type() != intermediateType_) {
        auto allocator = graph_->allocator();
        auto memory = allocator->alloc(tensor->size(), intermediateType_);
        auto temp = TensorBase::New(memory,
                                    tensor->shape(),
                                    intermediateType_,
                                    tensor->getBackend());
        lambda_(temp);
        CopyCast(tensor, temp); // Casting from temp to tensor
        allocator->free(memory);
      }
      else {
        lambda_(tensor);
      }
}

Ptr<NodeInitializer> zeros() {
  return fromValue(0.0f);
}

Ptr<NodeInitializer> ones() {
  return fromValue(1.0f);
}

Ptr<NodeInitializer> fromValue(float v) {
  return New<LambdaInit>([v](Tensor t){ t->set(v); });
}

// diagonal matrix with value val along diagonal
Ptr<NodeInitializer> eye(float val) {

  auto eyeLambda = [val](Tensor t) {
    ABORT_IF(t->shape().size() != 2 || t->shape()[-1] != t->shape()[-2],
              "eye(val) is defined only for quadratic tensors, shape is {}",
              t->shape());

    // @TODO: implement efficient version on the GPU
    std::vector<float> vec(t->size(), 0);
    for(int i = 0; i < t->shape()[-1]; ++i)
      vec[i * t->shape()[0] + i] = val;
    t->set(vec);
  };

  return New<LambdaInitConvert>(eyeLambda);
}

Ptr<NodeInitializer> uniform(float a, float b) {
  return New<LambdaInitConvert>([a, b](Tensor t) {
    t->getBackend()->getRandomGenerator()->uniform(t, a, b);
  });
}

Ptr<NodeInitializer> normal(float mean, float stddev) {
  return New<LambdaInitConvert>([mean, stddev](Tensor t) {
    t->getBackend()->getRandomGenerator()->normal(t, mean, stddev);
  });
}

Ptr<NodeInitializer> glorotUniform(bool fanIn, bool fanOut) {
  return New<LambdaInitConvert>([fanIn, fanOut](Tensor t) {
    float scale = sqrtf(6.0f / (t->shape()[-2] + t->shape()[-1]));
    if(fanIn && !fanOut)
      scale = sqrtf(3.0f / t->shape()[-2]);
    if(!fanIn && fanOut)
      scale = sqrtf(3.0f / t->shape()[-1]);

    t->getBackend()->getRandomGenerator()->uniform(t, -scale, scale);
  });
}

Ptr<NodeInitializer> glorotNormal(bool fanIn, bool fanOut) {
  return New<LambdaInitConvert>([fanIn, fanOut](Tensor t) {
    float scale = sqrtf(2.0f / (t->shape()[-2] + t->shape()[-1]));
    if(fanIn && !fanOut)
      scale = sqrtf(1.0f / t->shape()[-2]);
    if(!fanIn && fanOut)
      scale = sqrtf(1.0f / t->shape()[-1]);

    t->getBackend()->getRandomGenerator()->normal(t, 0.f, scale);
  });
}

Ptr<NodeInitializer> bernoulli(float prob, float scale) {
  return New<LambdaInitConvert>([prob, scale](Tensor t) { Bernoulli(t, prob, scale); });
}

Ptr<NodeInitializer> dropout(float dropProb) {
  return New<LambdaInitConvert>([dropProb](Tensor t) { Dropout(t, dropProb); });
}

// gumbel noise:
// -log(-log(uniform(0.f + eps, 1.f - eps)));
Ptr<NodeInitializer> gumbel() {
  return New<LambdaInitConvert>([](Tensor t) {
    using namespace functional;
    float eps = 1e-05f; // @TODO: make eps a parameter? Seems to influence amplitude quite heavily
    auto rng = t->getBackend()->getRandomGenerator();

    rng->uniform(t, 0.f + eps, 1.f - eps);
    Element(_1 = -log(-log(_1)), t);
  });
}

template <typename T>
Ptr<NodeInitializer> fromVector(const std::vector<T>& v) {
  auto vPtr = New<std::vector<T>>(v.begin(), v.end());
  return New<LambdaInitConvert>([vPtr](Tensor t) {
    t->set(vPtr->data(), vPtr->data() + vPtr->size());
  }, typeId<T>());
}

template Ptr<NodeInitializer> fromVector<float16>(const std::vector<float16>& v);
template Ptr<NodeInitializer> fromVector<float>(const std::vector<float>& v);
template Ptr<NodeInitializer> fromVector<IndexType>(const std::vector<IndexType>& v);

Ptr<NodeInitializer> fromSparseVector(std::pair<std::vector<size_t>, std::vector<float>>& v) {
  return New<LambdaInit>([v](Tensor t) {
    t->set(1e-6);
    t->setSparse(v.first, v.second);
  });
}

// move this somewhere else
Ptr<NodeInitializer> fromWord2vec(const std::string& file,
                              int dimVoc,
                              int dimEmb,
                              bool normalize /*= false*/) {
  return New<LambdaInit>([file, dimVoc, dimEmb, normalize](Tensor t) {
    auto embs = Word2VecReader().read(file, dimVoc, dimEmb);
    if(normalize) {
      float norm = 0;
      for(auto e : embs)
        norm += e * e;
      norm = std::sqrt(norm);
      if(norm != 0)
        for(auto& e : embs)
          e = e / norm;
    }
    t->set(embs);
  });
}

Ptr<NodeInitializer> fromItem(const io::Item& item) {
  if(item.mapped) {
    return New<LambdaInit>([item](Tensor tensor) {
      // @TODO: implement other types, for now croak loudly.
      ABORT_IF(tensor->getBackend()->getDeviceId().type != DeviceType::cpu,
               "Memory mapping only works for CPU tensors");
      ABORT_IF(tensor->type() != item.type,
               "Tensor type ({}) and type for mapping ({}) do not match",
               tensor->type(),
               item.type);
      ABORT_IF(tensor->size() != item.size() / sizeOf(item.type),
               "Tensor size ({}) and mapped size ({}) do not match",
               tensor->size(),
               item.size() / sizeOf(item.type));
      auto mp = MemoryPiece::New((uint8_t*)item.ptr, tensor->size() * sizeOf(item.type));
      tensor->reset(mp);
    });
  } else {
    return New<LambdaInitConvert>(
      [item](Tensor tensor) { tensor->set(item); },
      item.type);
  }
}

Ptr<NodeInitializer> fromTensor(Tensor externalTensor) {
  return New<LambdaInitConvert>([externalTensor](Tensor t) {
    t->copyFrom(externalTensor);
  }, externalTensor->type());
}

Ptr<NodeInitializer> dummy() {
  return New<LambdaInit>([](Tensor /*t*/) { });
}

}  // namespace inits

}  // namespace marian
