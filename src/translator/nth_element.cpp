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
#include <fstream>
#include <map>
#include <boost/algorithm/string.hpp>

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
                    const bool isFirst,
                    std::vector<std::vector<int>> trieVocabIdxs) {
    
    // vocabMap used for debugging
    /*
    std::map<int, std::string> vocabMap;
    std::string delimiter = ": ";
    std::ifstream input( "/home/patrick/Desktop/marian-dev/examples/trieme_new/model/vocab.deen.yml" );
    int count = 0;
    for( std::string line; getline( input, line ); ) {
      boost::trim_right(line);
      std::string token = line.substr(0, line.find(delimiter));
      // std::cout << token << " is " << count << ", ";
      vocabMap[count] = token;
      ++count;
    } */

    const auto vocabSize = scores->shape()[-1];
    const auto inputN    = scores->shape()[-2];
    const auto dimBatch  = scores->shape()[-4];

    ABORT_IF(inputN != (isFirst ? 1 : N), "Input tensor has wrong beam dim??"); // @TODO: Remove isFirst argument altogether
    const float* scoresData = scores->data();

    h_res.clear();
    h_res_idx.clear();

    size_t batchOffset = inputN * vocabSize;
    for(size_t batchIdx = 0; batchIdx < dimBatch; ++batchIdx) {
      if (trieVocabIdxs[batchIdx].size() > 0) {
        std::vector<int> idxs = trieVocabIdxs[batchIdx]; // idxs for all hyps
        std::partial_sort(
          idxs.begin(),
          idxs.begin() + std::min(N, idxs.size()),
          idxs.end(),
          [&](int a, int b) {return scoresData[a] > scoresData[b]; }
        );
        for(int temp = 0; temp < std::min(N, idxs.size()); ++temp) {
          int idx = idxs[temp];
          h_res_idx.push_back(idx + batchIdx * batchOffset);
          // scores do not need offset because the pointer gets advanced each time
          h_res.push_back(scoresData[idx]);
        }
        scoresData += batchOffset;
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
  return [nth](Tensor logProbs, size_t N, std::vector<float>& outCosts, std::vector<unsigned>& outKeys, const bool isFirst, std::vector<std::vector<int>> trieVocabIdxs) {
    return nth->getNBestList(logProbs, N, outCosts, outKeys, isFirst, trieVocabIdxs);
  };
}

}  // namespace marian