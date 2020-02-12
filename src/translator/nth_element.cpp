/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#include "translator/nth_element.h"
#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <iostream>

namespace marian {

class NthElementCPU {
  std::vector<int> h_res_idx;
  std::vector<float> h_res;
  //size_t lastN_;

public:
  NthElementCPU() {}
  NthElementCPU(const NthElementCPU& copy) = delete;


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

    std::vector<int> idxs(dimBatch * inputN * vocabSize);
    std::iota(idxs.begin(), idxs.end(), 0);

    for(size_t batchIdx = 0; batchIdx < dimBatch; ++batchIdx) {

      std::vector<int>::iterator begin = idxs.begin() + batchIdx * inputN * vocabSize;
      std::vector<int>::iterator middle = begin + (int)N;
      std::vector<int>::iterator end = idxs.begin() + (batchIdx + 1) * inputN * vocabSize;
      std::partial_sort(
          begin, middle, end, [&](int a, int b) { return scoresData[a] > scoresData[b]; });

      int pos = batchIdx * (int)N;
      while(begin != middle) {
        int idx = *begin++;
        h_res_idx[pos] = idx;
        h_res[pos] = scoresData[idx];
        ++pos;
      }
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
  return [nth](Tensor logProbs, size_t N, std::vector<float>& outCosts, std::vector<unsigned>& outKeys, const bool isFirst) {
    return nth->getNBestList(logProbs, N, outCosts, outKeys, isFirst);
  };
}

}  // namespace marian