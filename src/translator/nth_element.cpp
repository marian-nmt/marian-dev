/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#include "translator/nth_element.h"
#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#ifdef __AVX512F__
#include <immintrin.h>
#endif

namespace marian {

class NthElementCPU {
  std::vector<int> h_res_idx;
  std::vector<float> h_res;
  //size_t lastN_;

public:
  NthElementCPU() {}
  NthElementCPU(const NthElementCPU& copy) = delete;

#ifdef __AVX512F__
  int max_elem(const float * vec, size_t size) {
    float maxVal = vec[0];
    int max_idx = 0;
    div_t setup = div(size, 16);
    int overhang = setup.rem;
    int seq = setup.quot;
    __m512 maxvalVec = _mm512_set1_ps(maxVal);
    for (int i = 0; i < seq*16; i+=16) {
        auto res = _mm512_cmp_ps_mask(maxvalVec, _mm512_load_ps(&vec[i]), _CMP_LT_OS);
        if (res != 0) {
            for (int j = 0; j<16; j++) {
                if (vec[i+j] > maxVal) {
                    maxVal = vec[i+j];
                    max_idx = i+j;
                }
            }
            maxvalVec = _mm512_set1_ps(maxVal);
        }
    }

    for (int i = seq*16; i < seq*16 + overhang; i++) {
        if (maxVal < vec[i]) {
            max_idx = i;
            maxVal = vec[i];
        }
    }
    return max_idx;
}
#else
int max_elem(const float * vec, size_t size) {
    auto elem = std::max_element(vec, vec + size);
    return std::distance(vec, elem);
}
#endif


public:
  void getNBestList(Tensor scores, // [dimBatch, 1, beamSize, dimVocab or dimShortlist]
                    size_t N,
                    std::vector<float>& outPathScores,
                    std::vector<unsigned>& outKeys,
                    const bool isFirst) {
    const auto vocabSize = scores->shape()[-1];
    const auto inputN    = scores->shape()[-2];
    const auto dimBatch  = scores->shape()[-4];
    ABORT_IF(inputN != (isFirst ? 1 : N), "Input tensor has wrong beam dim??"); // @TODO: Remove isFirst argument altogether
    const float* scoresData = scores->data();

    size_t maxSize = N * dimBatch;
    h_res.resize(maxSize);
    h_res_idx.resize(maxSize);
    size_t pos = 0; // iterates through h_res and h_res_idx

    size_t batchOffset = inputN * vocabSize;
    std::vector<int> idxs(batchOffset); // re-used for each batch
    std::iota(idxs.begin(), idxs.end(), 0);

    for(size_t batchIdx = 0; batchIdx < dimBatch; ++batchIdx) {

      std::partial_sort( 
        // sorts the top N (beam size) idxs by score to the front
        idxs.begin(),
        idxs.begin() + N,
        idxs.end(),
        [&](int a, int b) { return scoresData[a] > scoresData[b]; }
      );

      // copy top N idxs and scores to return vectors
      for(size_t i = 0; i < N; ++i) {
        int idx = idxs[i];
        // since idxs is re-used for each batch, add batch offset to each idx to get absolute position
        h_res_idx[pos] = (int) (idx + batchIdx * batchOffset);
        h_res[pos] = scoresData[idx];
        ++pos;
      }

      // advance pointer to next batch's beginning
      scoresData += batchOffset;
    }
    getPairs(/*cumulativeBeamSizes.back(),*/ outKeys, outPathScores);
  }

  void getNBestListBeam1(Tensor scores,  // [dimBatch, 1, beamSize, dimVocab or dimShortlist]
                         std::vector<float>& outPathScores,
                         std::vector<unsigned>& outKeys,
                         const bool isFirst) {
    constexpr size_t N   = 1;  // N best list is always 1
    const auto vocabSize = scores->shape()[-1];
    const auto inputN    = scores->shape()[-2];
    const auto dimBatch  = scores->shape()[-4];
    // ABORT_IF(inputN != (isFirst ? 1 : N), "Input tensor has wrong beam dim??");  // @TODO: Remove isFirst argument altogether
    const float* scoresData = scores->data();

    size_t maxSize = N * dimBatch;
    h_res.resize(maxSize);
    h_res_idx.resize(maxSize);
    size_t pos = 0;  // iterates through h_res and h_res_idx

    size_t batchOffset = inputN * vocabSize;

    for(size_t batchIdx = 0; batchIdx < dimBatch; ++batchIdx) {

        // copy top N idxs and scores to return vectors
        int idx = max_elem(scoresData, batchOffset); // n-th vocabulary item is the vocabulary id
        // since idxs is re-used for each batch, add batch offset to each idx to get absolute position
        h_res_idx[pos] = (int)(idx + batchIdx * batchOffset);
        h_res[pos]     = scoresData[idx];
        ++pos;

      // advance pointer to next batch's beginning
      scoresData += batchOffset;
    }
    getPairs(/*cumulativeBeamSizes.back(),*/ outKeys, outPathScores);
  }

private:
  void getPairs(/*size_t number,*/
                std::vector<unsigned>& outKeys,
                std::vector<float>& outValues) {
    std::copy(h_res_idx.begin(), h_res_idx.end(), std::back_inserter(outKeys));
    std::copy(h_res    .begin(), h_res    .end(), std::back_inserter(outValues));
    //lastN_ = number;
  }

  //void getValueByKey(std::vector<float>& out, float* d_in) {
  //  for(size_t i = 0; i < lastN_; ++i) {
  //    out[i] = d_in[h_res_idx[i]];
  //  }
  //}
};

#ifdef CUDA_FOUND
GetNBestListFn createGetNBestListGPUFn(size_t beamSize, size_t dimBatch, DeviceId deviceId); // in .cu file
#endif

// factory function
// Returns a lambda with the same signature as the getNBestList() function.
GetNBestListFn createGetNBestListFn(size_t beamSize, size_t dimBatch, DeviceId deviceId) {
#ifdef CUDA_FOUND
  if(deviceId.type == DeviceType::gpu)
    return createGetNBestListGPUFn(beamSize, dimBatch, deviceId);
#else
  deviceId; beamSize; dimBatch; // (unused)
#endif
  auto nth = New<NthElementCPU>();
  if (beamSize == 1) {
    return [nth](Tensor logProbs, size_t N, std::vector<float>& outCosts, std::vector<unsigned>& outKeys, const bool isFirst) {
      N; // unused
      return nth->getNBestListBeam1(logProbs, outCosts, outKeys, isFirst);
    };
  } else {
    return [nth](Tensor logProbs, size_t N, std::vector<float>& outCosts, std::vector<unsigned>& outKeys, const bool isFirst) {
      return nth->getNBestList(logProbs, N, outCosts, outKeys, isFirst);
    };
  }
}

}  // namespace marian
