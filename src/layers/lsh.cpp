#include "common/timer.h"
#include "common/utils.h"
#include "layers/lsh.h"
#include "tensors/tensor_operators.h"
#include "tensors/cpu/cpu_info.h"
#include "tensors/cpu/mjdgemm/mjdgemm.h"

#if _MSC_VER
#include "3rd_party/faiss/Index.h"
#endif

#include "3rd_party/faiss/utils/hamming.h"

#if BLAS_FOUND
#include "3rd_party/faiss/VectorTransform.h"
#endif

namespace marian {
namespace lsh {

int bytesPerVector(int nBits) {
  return (nBits + 7) / 8;
}

void fillRandomRotationMatrix(Tensor output, Ptr<Allocator> allocator) {
#if BLAS_FOUND
  int nRows = output->shape()[-2];
  int nBits = output->shape()[-1];

  // @TODO re-implement using Marian code so it uses the correct random generator etc.
  faiss::RandomRotationMatrix rrot(nRows, nBits);
  // Then we do not need to use this seed at all
  rrot.init(5); // currently set to 5 following the default from FAISS, this could be any number really.

  // The faiss random rotation matrix is column major, hence we create a temporary tensor,
  // copy the rotation matrix into it and transpose to output.
  Shape tempShape = {nBits, nRows};
  auto memory = allocator->alloc(requiredBytes(tempShape, output->type()));
  auto temp = TensorBase::New(memory,
                              tempShape,
                              output->type(),
                              output->getBackend());
  temp->set(rrot.A);
  TransposeND(output, temp, {0, 1, 3, 2});
  allocator->free(memory);
#else
  output; allocator;
  ABORT("LSH with rotation matrix requires Marian to be compiled with a BLAS library");
#endif
}

void encode(Tensor output, Tensor input) {
  int nBits = input->shape()[-1]; // number of bits is equal last dimension of float matrix
  int nRows = input->shape().elements() / nBits;
  if (input->getDeviceId().type == DeviceType::cpu) {
    faiss::fvecs2bitvecs(input->data<float>(), output->data<uint8_t>(), (size_t)nBits, (size_t)nRows);
  }
  else {
#ifdef CUDA_FOUND
    marian::gpu::Float2Bit(output, input);
#endif
}
}

void encodeWithRotation(Tensor output, Tensor input, Tensor rotation, Ptr<Allocator> allocator) {
  int nBits = input->shape()[-1]; // number of bits is equal last dimension of float matrix unless we rotate
  int nRows = input->shape().elements() / nBits;

  Tensor tempInput = input;
  MemoryPiece::PtrType memory;
  if(rotation) {
    int nBitsRot = rotation->shape()[-1];
    Shape tempShape = {nRows, nBitsRot};
    memory = allocator->alloc(requiredBytes(tempShape, input->type()));
    tempInput = TensorBase::New(memory, tempShape, input->type(), input->getBackend());
    if(rotation->type() == Type::packed8avx512) {
      using namespace marian::cpu::mjdgemm;
      gemmInt8Packed(input->data(), rotation->data<int8_t>(), nullptr, tempInput->data(), nRows, nBitsRot, nBits);
    } else {
      Prod(tempInput, input, rotation, false, false, 0.f, 1.f);
    }
  }
  encode(output, tempInput);

  if(memory)
    allocator->free(memory);
};

Expr encode(Expr input, Expr rotation) {
  auto encodeFwd = [](Expr out, const std::vector<Expr>& inputs) {
    if(inputs.size() == 1) {
      encode(out->val(), inputs[0]->val());
    } else if(inputs.size() == 2) {
      encodeWithRotation(out->val(), inputs[0]->val(), inputs[1]->val(), out->graph()->allocator());
    } else {
      ABORT("Too many inputs to encode??");
    }
  };

  // Use the address of the first lambda function as an immutable hash. Making it static and const makes sure
  // that this hash value will not change. Next pass the hash into the lambda functor were it will be used
  // to identify this unique operation. Marian's ExpressionGraph can automatically memoize and identify nodes
  // that operate only on immutable nodes (parameters) and have the same hash. This way we make sure that the
  // codes node won't actually get recomputed throughout ExpressionGraph lifetime. `codes` will be reused
  // and the body of the lambda will not be called again. This does however build one index per graph.
  static const size_t encodeHash = (size_t)&encodeFwd;

  Shape encodedShape = input->shape();

  int nBits = rotation ? rotation->shape()[-1] : input->shape()[-1];
  encodedShape.set(-1, bytesPerVector(nBits));
  std::vector<Expr> inputs = {input};
  if(rotation)
    inputs.push_back(rotation);
  return lambda(inputs, encodedShape, Type::uint8, encodeFwd, encodeHash);
}

Expr rotator(Expr weights, int inDim, int nBits) {
  auto rotator = [](Expr out, const std::vector<Expr>& inputs) {
    inputs;
    fillRandomRotationMatrix(out->val(), out->graph()->allocator());
  };

  static const size_t rotatorHash = (size_t)&rotator;
  return lambda({weights}, {inDim, nBits}, Type::float32, rotator, rotatorHash);
}

Expr searchEncoded(Expr encodedQuery, Expr encodedWeights, int dimK, int firstNRows, bool noSort/*= false*/) {
  ABORT_IF(encodedQuery->shape()[-1] != encodedWeights->shape()[-1],
           "Query and index bit vectors need to be of same size ({} != {})", encodedQuery->shape()[-1], encodedWeights->shape()[-1]);

  int currBeamSize = encodedQuery->shape()[0];
  int batchSize    = encodedQuery->shape()[2];

  auto search = [=](Expr out, const std::vector<Expr>& inputs) {
    Expr encodedQuery   = inputs[0];
    Expr encodedWeights = inputs[1];

    if (encodedQuery->val()->getDeviceId().type == DeviceType::cpu) {
      int bytesPerVector = encodedWeights->shape()[-1];
      int wRows = encodedWeights->shape().elements() / bytesPerVector;

      // we use this with Factored Segmenter to skip the factor embeddings at the end
      if(firstNRows != 0)
        wRows = firstNRows;

      ABORT_IF(dimK > wRows, "k is larger than number of candidate values?"); // @TODO: use min(k, wRows) silently?

#if _MSC_VER // unfortunately MSVC is horrible at loop unrolling, so we fall back to the old code (hrmph!) @TODO: figure this out one day
      int qRows = encodedQuery->shape().elements() / bytesPerVector;

      uint8_t* qCodes = encodedQuery->val()->data<uint8_t>();
      uint8_t* wCodes = encodedWeights->val()->data<uint8_t>();

      // use actual faiss code for performing the hamming search.
      std::vector<int> distances(qRows * dimK);
      std::vector<faiss::Index::idx_t> ids(qRows * dimK);
      faiss::int_maxheap_array_t res = {(size_t)qRows, (size_t)dimK, ids.data(), distances.data()};
      faiss::hammings_knn_hc(&res, qCodes, wCodes, (size_t)wRows, (size_t)bytesPerVector, 0);

      // Copy int64_t indices to Marian index type and sort by increasing index value per hypothesis.
      // The sorting is required as we later do a binary search on those values for reverse look-up.
      uint32_t* outData = out->val()->data<uint32_t>();

      int numHypos = out->shape().elements() / dimK;
      for (size_t hypoIdx = 0; hypoIdx < numHypos; ++hypoIdx) {
        size_t startIdx = dimK * hypoIdx;
        size_t endIdx = startIdx + dimK;
        for(size_t i = startIdx; i < endIdx; ++i)
          outData[i] = (uint32_t)ids[i];
        if(!noSort)
          std::sort(outData + startIdx, outData + endIdx);
      }
#else // this is using the new code for search, other parts of the code, like conversion are fine.
      IndexType* outData = out->val()->data<IndexType>();
      auto gather = [outData, dimK](IndexType rowId, IndexType k, IndexType kthColId, DistType /*dist*/) {
        outData[rowId * dimK + k] = kthColId;
      };

      Parameters params;
      params.k              = dimK;
      params.queryRows      = encodedQuery->val()->data<uint8_t>();
      params.numQueryRows   = encodedQuery->shape().elements() / bytesPerVector;
      params.codeRows       = encodedWeights->val()->data<uint8_t>();
      params.numCodeRows    = wRows;
      params.bytesPerVector = bytesPerVector;

      hammingTopK(params, gather);
#endif
    }
    else {
#ifdef CUDA_FOUND
      Ptr<Backend> backend = out->val()->getBackend();

      const size_t CHUNK = 128;
      const size_t MBYTE = 1024 * 1024;
      const size_t GROW = CHUNK * MBYTE;
      Ptr<Allocator> alloc = marian::New<marian::Allocator>(backend->getDeviceId(), 0, GROW);

      auto memory = alloc->alloc(requiredBytes(out->shape(), marian::Type::uint32));

      // not required for calculations. Useful for debugging
      Tensor outCounts = nullptr; //marian::TensorBase::New(memory, out->shape(), marian::Type::uint32, backend);

      uint16_t numHash = (uint16_t) encodedWeights->shape()[-1] * 8;

      marian::gpu::HammmingAndSort(out->val(), outCounts,
                  encodedWeights->val(), encodedQuery->val(),
                  dimK, 0, numHash,
                  alloc, backend);
#endif
    }
  };

  Shape kShape({currBeamSize, batchSize, dimK});
  return lambda({encodedQuery, encodedWeights}, kShape, Type::uint32, search);
}

Expr search(Expr query, Expr weights, int k, int nBits, int firstNRows, bool abortIfDynamic) {
  int dim = weights->shape()[-1];

  Expr rotMat = nullptr;
  if(dim != nBits) {
    rotMat = weights->graph()->get("lsh_output_rotation");
    if(rotMat) {
      LOG_ONCE(info, "Reusing parameter LSH rotation matrix {} with shape {}", rotMat->name(), rotMat->shape());
    } else {
      ABORT_IF(abortIfDynamic, "Dynamic creation of LSH rotation matrix prohibited");
      LOG_ONCE(info, "Creating ad-hoc rotation matrix with shape {}", Shape({dim, nBits}));
      rotMat = rotator(weights, dim, nBits);
    }
  }

  Expr encodedWeights = weights->graph()->get("lsh_output_codes");
  if(encodedWeights) {
    LOG_ONCE(info, "Reusing parameter LSH code matrix {} with shape {}", encodedWeights->name(), encodedWeights->shape());
  } else {
    ABORT_IF(abortIfDynamic, "Dynamic creation of LSH code matrix prohibited");
    LOG_ONCE(info, "Creating ad-hoc code matrix with shape {}", Shape({weights->shape()[-2], lsh::bytesPerVector(nBits)}));
    encodedWeights = encode(weights, rotMat);
  }

  return searchEncoded(encode(query, rotMat), encodedWeights, k, firstNRows);
}

class RandomRotation : public inits::NodeInitializer {
public:
  void apply(Tensor tensor) override {
    auto sharedAllocator = allocator_.lock();
    ABORT_IF(!sharedAllocator, "Allocator in RandomRotation has not been set or expired");
    fillRandomRotationMatrix(tensor, sharedAllocator);
  }
};

Ptr<inits::NodeInitializer> randomRotation() {
  return New<RandomRotation>();
}

void addDummyParameters(Ptr<ExpressionGraph> graph, ParamConvInfo paramInfo) {
  auto weights = graph->get(paramInfo.name);
  int nBitsRot = paramInfo.nBits;

  ABORT_IF(!weights, "Trying to encode non-existing weights matrix {}??", paramInfo.name);

  int nBits = weights->shape()[-1];
  if(paramInfo.transpose)
    nBits = weights->shape()[-2];

  int nRows = weights->shape().elements() / nBits;

  Expr rotation;
  if(nBits != nBitsRot) {
    LOG(info, "Adding LSH rotation parameter {} with shape {}", paramInfo.rotationName, Shape({nBits, nBitsRot}));
    rotation = graph->param(paramInfo.rotationName, {nBits, nBitsRot}, inits::dummy(), Type::float32);
    nBits = nBitsRot;
  }

  int bytesPerVector = lsh::bytesPerVector(nBits);
  LOG(info, "Adding LSH encoded weights {} with shape {}", paramInfo.codesName, Shape({nRows, bytesPerVector}));
  auto codes = graph->param(paramInfo.codesName, {nRows, bytesPerVector}, inits::dummy(), Type::uint8);
}

void overwriteDummyParameters(Ptr<ExpressionGraph> graph, ParamConvInfo paramInfo) {
  Expr weights  = graph->get(paramInfo.name);
  Expr codes    = graph->get(paramInfo.codesName);
  Expr rotation = graph->get(paramInfo.rotationName);

  ABORT_IF(!weights, "Trying to encode non-existing weights matrix {}??", paramInfo.name);
  ABORT_IF(!codes, "Trying to overwrite non-existing LSH parameters lsh_output_codes??");

  if(paramInfo.transpose) {
    weights = transpose(weights);
    graph->forward();
  }

  if(rotation) {
    fillRandomRotationMatrix(rotation->val(), weights->graph()->allocator());
    encodeWithRotation(codes->val(), weights->val(), rotation->val(), weights->graph()->allocator());
  } else {
    encode(codes->val(), weights->val());
  }
}

}
}