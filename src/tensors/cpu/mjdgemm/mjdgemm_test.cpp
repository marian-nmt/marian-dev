#include <iostream>

#include "common/io.h"
#include "common/types.h"

#include "3rd_party/fbgemm/include/fbgemm/QuantUtilsAvx2.h"
#include "3rd_party/fbgemm/include/fbgemm/QuantUtils.h"
#include "3rd_party/fbgemm/include/fbgemm/Fbgemm.h"
#include "packed_gemm.h"

#include "mjd_gemm.h"

using namespace fbgemm;
using namespace marian;

// Memory blocking factors (parameters) for packing into AVX2 int8
static const fbgemm::BlockingFactors Packed8Avx2BlockingFactors = {
    PackingTraits<int8_t, int32_t, inst_set_t::avx2>::MR,
    PackingTraits<int8_t, int32_t, inst_set_t::avx2>::NR,
    PackingTraits<int8_t, int32_t, inst_set_t::avx2>::NR_MIN,
    PackingTraits<int8_t, int32_t, inst_set_t::avx2>::ROW_INTERLEAVE,
    PackingTraits<int8_t, int32_t, inst_set_t::avx2>::MCB,
    PackingTraits<int8_t, int32_t, inst_set_t::avx2>::KCB,
    PackingTraits<int8_t, int32_t, inst_set_t::avx2>::NCB
};

// Memory blocking factors (parameters) for packing into AVX512 int8
static const fbgemm::BlockingFactors Packed8Avx512BlockingFactors = {
    PackingTraits<int8_t, int32_t, inst_set_t::avx512>::MR,
    PackingTraits<int8_t, int32_t, inst_set_t::avx512>::NR,
    PackingTraits<int8_t, int32_t, inst_set_t::avx512>::NR_MIN,
    PackingTraits<int8_t, int32_t, inst_set_t::avx512>::ROW_INTERLEAVE,
    PackingTraits<int8_t, int32_t, inst_set_t::avx512>::MCB,
    PackingTraits<int8_t, int32_t, inst_set_t::avx512>::KCB,
    PackingTraits<int8_t, int32_t, inst_set_t::avx512>::NCB
};

// This function returns the correct blocking factors structure for given packing type.
inline const fbgemm::BlockingFactors* getBlockingFactors(marian::Type packType) {
  if(packType == Type::packed8avx2) {
    return &Packed8Avx2BlockingFactors;
  } else if(packType == Type::packed8avx512) {
    return &Packed8Avx512BlockingFactors;
  } else {
    ABORT("Only avx2 and avx512 instruction sets are supported for int8. {}", packType);
  }
}

void unpack(int8_t* in, int8_t* out) {
  int32_t k = 512;
  int32_t n = 512;

  using namespace fbgemm;
  const fbgemm::BlockingFactors* params = getBlockingFactors(Type::packed8avx512);

  // packed matrix size of B
  // int packSizeB = PackMatrix<PackBMatrix<int8_t>, int8_t>::packedBufferSize(k, n);

  PackBMatrix<int8_t> repackedB(matrix_op_t::NoTranspose, k, n, in, n, 1, params);

  std::vector<int8_t> unpacked(k * n);
  repackedB.unpack(out, params);
}

void fbgemmPacked8Gemm(float* C,
                       const float* A,
                       const int8_t* B,
                       const size_t m,
                       const size_t n,
                       const size_t k,
                       const int transA,
                       const int transB) {
  Type packType = Type::packed8avx512;
  const fbgemm::BlockingFactors* params = getBlockingFactors(packType);

  // Check if the packed format matches with the available AVX instruction set in the machine
  const bool avx512Support = fbgemmHasAvx512Support();
  if((packType == Type::packed8avx2 && avx512Support)
     || (packType == Type::packed8avx512 && !avx512Support)) {
    ABORT("FBGEMM doesn't allow to use {} packing order on {} CPUs",
          packType == Type::packed8avx2 ? "AVX2" : "AVX512",
          avx512Support ? "AVX512" : "AVX2");
  }

  // compute range to quantize A (activations) - (min/max quantization)
  float minA = std::numeric_limits<float>::max(), maxA = std::numeric_limits<float>::lowest();

  int elemA = m * k;
  float* dataA = const_cast<float*>(A);
  // AVX based find min/max
  FindMinMax(dataA, &minA, &maxA, elemA);

  // MJD: so we only compute one quant scale and zero point for the A matrix? But multiple for the paramter B?
  float quantScaleA = (maxA - minA) / 255;
  float quantZeropointA = (255 - maxA / quantScaleA);

  // To avoid any repeated memory allocation and deallocation, make the scratch buffer variables static thread_local
  // In a multi-threaded situation, heap access lock for the memory allocation/free could
  // makes all the threads are blocked by each other. (heap contention)
  const size_t sizeBufA = params->KCB * params->MCB;
  static thread_local std::vector<uint8_t> packedBufA;
  if (packedBufA.size() < sizeBufA)
	  packedBufA.resize(sizeBufA);
  const size_t sizeRowOffsetBufA = PackAWithQuantRowOffset<uint8_t>::rowOffsetBufferSize();
  static thread_local std::vector<int32_t> rowOffsetBufA;
  if (rowOffsetBufA.size() < sizeRowOffsetBufA)
	  rowOffsetBufA.resize(sizeRowOffsetBufA);

  // print sizeRowOffsetBufA
  std::cout << "MCB: " << params->MCB << std::endl;
  std::cout << "sizeRowOffsetBufA: " << sizeRowOffsetBufA << std::endl;

  PackAWithQuantRowOffset<uint8_t> packA(
      transA ? matrix_op_t::Transpose : matrix_op_t::NoTranspose,
      (int32_t)(transA ? k : m),
      (int32_t)(transA ? m : k),
      dataA,
      (int32_t)(transA ? m : k),
      // buffer for packed matrix, pass a pre-allocated memory to avoid additional allocation/deallocation inside fbgemm
      packedBufA.data(),
      quantScaleA,
      quantZeropointA,
      1, /*groups*/
      rowOffsetBufA.data(),
      params);

  // packed matrix size of B
  int packSizeB = PackMatrix<PackBMatrix<int8_t>, int8_t>::packedBufferSize((int32_t)k, (int32_t)n);

  // retrieve B matrix
  int8_t* dataB = const_cast<int8_t*>(B);

  // quantization parameters for B, access them from the end of the packed buffer
  // there is n of quantScaleB, quantZeropointB and colOffsetsB, each
  const float* quantScaleB       = (const float*)   (dataB + packSizeB);
  const int32_t* quantZeropointB = (const int32_t*) (dataB + packSizeB + n * sizeof(float));
  const int32_t* colOffsetsB     = (const int32_t*) (dataB + packSizeB + n * sizeof(float) + n * sizeof(int32_t));

  DoNothing<float, float> doNothingObj{};
  ReQuantizeForFloat<false, QuantizationGranularity::OUT_CHANNEL> outputProcObj(
      doNothingObj,
      quantScaleA,
      quantScaleB,
      quantZeropointA,
      quantZeropointB,
      packA.getRowOffsetBuffer(),
      colOffsetsB,
      nullptr,
      (std::uint32_t) n);

  PackBMatrix<int8_t> repackedB(transB ? matrix_op_t::Transpose : matrix_op_t::NoTranspose, (int32_t) k, (int32_t) n, dataB, (int32_t) (transB ? k : n), 1, params);

  // gemm computation
  fbgemmPacked(packA, repackedB, C, (int32_t*)C, (int32_t) n, outputProcObj, 0, 1, params);
}


int main(int argc, char** argv) {
  // get model path from command line
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model-path>" << std::endl;
    return 1;
  }
  const std::string modelpath = argv[1];
  marian::io::ModelWeights modelWeights(modelpath);
  auto& items = modelWeights.items();

  // find item with name "encoder_l1_self_Wq"
  std::string paramName = "encoder_l1_ffn_W1";
  auto it = std::find_if(items.begin(), items.end(), [paramName](const marian::io::Item& item) {
    return item.name == paramName;
  });

  if (it == items.end()) {
    std::cerr << "Parameter " << paramName << " not found in model" << std::endl;
    return 1;
  }

  // print paramter information
  const marian::io::Item& w = *it;
  std::cout << "Parameter name: " << w.name << std::endl;
  std::cout << "Parameter shape: " << w.shape << std::endl;
  std::cout << "Parameter type: " << w.type << std::endl;
  std::cout << std::endl;

  // find parameter with name "encoder_l1_self_bq"
  paramName = "encoder_l1_self_bq";
  it = std::find_if(items.begin(), items.end(), [paramName](const marian::io::Item& item) {
    return item.name == paramName;
  });

  if (it == items.end()) {
    std::cerr << "Parameter " << paramName << " not found in model" << std::endl;
    return 1;
  }

  // print parameter information
  const marian::io::Item& b = *it;
  std::cout << "Parameter name: " << b.name << std::endl;
  std::cout << "Parameter shape: " << b.shape << std::endl;
  std::cout << "Parameter type: " << b.type << std::endl;
  std::cout << std::endl;

  // create float32 buffer with M x K elements
  const int M = 4;
  const int K = w.shape[-2];
  const int N = w.shape[-1];

  std::vector<float> A(M * K);
  std::vector<float> C(M * N, 0.0f);
  std::vector<float> Ctest(M * N, 0.0f);

  // initialize A with random values between -1 and 1 and set seed
  srand(1234);
  std::generate(A.begin(), A.end(), []() {
    return static_cast<float>(rand()) / RAND_MAX * 2 - 1;
  });

  // print first 10 elements of A
  std::cout << "First 10 elements of A:" << std::endl;
  for (int i = 0; i < 10; ++i) {
    std::cout << A[i] << " ";
  }
  std::cout << std::endl << std::endl;

  // first 10 elements of W
  std::cout << "First 10 elements of W:" << std::endl;
  for (int i = 0; i < 10; ++i) {
    std::cout << (int)reinterpret_cast<const int8_t*>(w.data())[i] << " ";
  }
  std::cout << std::endl << std::endl;

  // first 10 elements of b
  std::cout << "First 10 elements of b:" << std::endl;
  for (int i = 0; i < 10; ++i) {
    std::cout << reinterpret_cast<const float*>(b.data())[i] << " ";
  }
  std::cout << std::endl << std::endl;

  // perform matrix multiplication
  fbgemmPacked8Gemm(C.data(), A.data(), reinterpret_cast<const int8_t*>(w.data()), M, N, K, 0, 0);

  // print first 10 elements of C
  std::cout << "First 10 elements of C:" << std::endl;
  for (int i = 0; i < 10; ++i) {
    std::cout << C[i] << " ";
  }
  std::cout << std::endl << std::endl;

  using namespace marian::cpu::mjdgemm;
  gemmInt8Packed(A.data(), reinterpret_cast<const int8_t*>(w.data()), Ctest.data(), M, N, K);

  // print first 10 elements of Ctest
  std::cout << "First 10 elements of Ctest:" << std::endl;
  for (int i = 0; i < 10; ++i) {
    std::cout << Ctest[i] << " ";
  }
  std::cout << std::endl << std::endl;

  // compare C and Ctest
  bool correct = true;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      if (C[i * N + j] != Ctest[i * N + j]) {
        correct = false;
        std::cout << "Mismatch at (" << i << ", " << j << "): "
              << "C=" << C[i * N + j]
              << " Ctest=" << Ctest[i * N + j] << std::endl;
      }
    }
  }

  if (correct) {
    std::cout << "C and Ctest match!" << std::endl;
  } else {
    std::cout << "C and Ctest do not match." << std::endl;
  }

  return 0;
}
