#pragma once

namespace marian {
namespace lsh {

// this namespace contains the unrolled version of the hammingTopK function which is faster than the dynamic version
// The compiler seems to like to know the total number of code rows at compile time.
namespace unrolled {

template <marian::cpu::InstructionSet IS, int warpSize, int NumCodeRows, int BytesPerVector, class Functor>
inline void hammingTopKUnrollWarp(int queryOffset, const Parameters& parameters, const Functor& gather) {
  constexpr int numBits = BytesPerVector * 8;
  static_assert(numBits % 64 == 0, "LSH hash size must be a multiple of 64");

  // counter to keep track of seen hamming distances
  std::array<std::array<DistType, numBits>, warpSize> counter;
  // buffer the distances for query vector warpRowId to all weight weight vectors codeRowId
  std::array<std::array<DistType, NumCodeRows>, warpSize> distBuffer;
  // minimal distances per query
  std::array<DistType, warpSize> minDist;

  constexpr int StepsStatic = BytesPerVector / sizeof(ChunkType);
  ChunkType* codeRow = (ChunkType*)parameters.codeRows;

  for(int warpRowId = 0; warpRowId < warpSize; warpRowId++) {
    std::fill(counter[warpRowId].begin(), counter[warpRowId].end(), 0);
    minDist[warpRowId] = (DistType)numBits;
  }

  for(IndexType codeRowId = 0; codeRowId < (IndexType)NumCodeRows; ++codeRowId, codeRow += StepsStatic) {
    ChunkType* queryRow = (ChunkType*)parameters.queryRows;
    for(IndexType warpRowId = 0; warpRowId < warpSize; warpRowId++, queryRow += StepsStatic) {
      // Compute the bit-wise hamming distance
      const DistType dist = hamming<IS, StepsStatic>(queryRow, codeRow);

      // Record the minimal distance seen for this query vector wrt. all weight vectors
      if(dist < minDist[warpRowId]) {
        minDist[warpRowId] = dist;
      }

      // Record the number of weight vectors that have this distance from the query vector.
      // Note, because there is at most numBits different distances this can be trivially done.
      // Not the case for generic distances like float.
      counter[warpRowId][dist]++;

      // Record the distance for this weight vector
      distBuffer[warpRowId][codeRowId] = dist;
    }
  }
  // warp finished, harvest k top distances

  for(int warpRowId = 0; warpRowId < warpSize; warpRowId++) {
    // Here we search for the distance at which we have seen equal or more than k elements with
    // smaller distances. We start with the minimal distance from above which is its own address
    // to the counter.
    DistType maxDist = minDist[warpRowId];
    size_t cummulativeDistances = 0;

    // Accumulate number of elements until we reach k in growing distance order. Note that
    // counter is indexed by hamming distance - from lowest to highest. Some slots will be 0.
    // The cumulative sum from position a to b tells you how many elements have distances smaller
    // than the distance at b.
    while(cummulativeDistances < parameters.k)
      cummulativeDistances += counter[warpRowId][maxDist++];
    if(cummulativeDistances)
      maxDist--; // fix overcounting

    // Usually, we overshoot by a couple of elements and we need to take care of the distance at which the k-th
    // element sits. This elements has more neighbors at the same distance, but we only care for them
    // as long we have not reached k elements in total.
    // By contrast, we trivially collect all elements below that distance -- these are always safe.

    // This is the number of elements we need to collect at the last distance.
    const DistType maxDistLimit = /*number of elements at maxDist=*/counter[warpRowId][maxDist] - /*overflow=*/((DistType)cummulativeDistances - (DistType)parameters.k);
    IndexType kSeen = 0;
    IndexType kSeenAtKDist = 0;

    for(IndexType codeRowId = 0; kSeen < (IndexType)parameters.k && codeRowId < (IndexType)NumCodeRows; ++codeRowId) {
      DistType dist = distBuffer[warpRowId][codeRowId];
      // - if the current distance is smaller than the maxDist, just consume.
      // - if the distance is equal to maxDist, make sure to only consume maxDistLimit elements at maxDist
      //   and ignore the rest (smaller indices make it in first).
      // - after we finish this loop we have exactly k top values for every query row in original index order.
      int queryRowId = queryOffset + warpRowId;
      if(dist < maxDist) {
        gather(queryRowId, (IndexType)kSeen, codeRowId, dist);
        kSeen++;
      } else if(dist == maxDist && kSeenAtKDist < (DistType)maxDistLimit) {
        gather(queryRowId, (IndexType)kSeen, codeRowId, dist);
        kSeen++;
        kSeenAtKDist++;
      }
    }
  }
}

// Faster top-k search for hamming distance. The idea here is that instead of sorting the elements we find a hamming distances at which it is safe
// to copy the given index. Copying only the indices below that distance is guaranteed to results in no more than k elements. For elements at that
// distance we need to correct for overshooting.
// Once we have that distance we only need to traverse the set of distances. In the end we get exactly k elements per queryRows vector.
template <marian::cpu::InstructionSet IS, int NumCodeRows, int BytesPerVector, class Functor>
inline void hammingTopKUnroll(const Parameters& parameters, const Functor& gather) {
  int warpSize = 4; // starting warpSize of 4 seems optimal
  auto warpParameters = parameters;
  for(int queryOffset = 0; queryOffset < parameters.numQueryRows; queryOffset += warpSize) {
    while(parameters.numQueryRows - queryOffset < warpSize)
      warpSize /= 2;

    warpParameters.queryRows    = parameters.queryRows + queryOffset * BytesPerVector;
    warpParameters.numQueryRows = warpSize;
    switch(warpSize) {
      case 8 : hammingTopKUnrollWarp<IS, 8, NumCodeRows, BytesPerVector>(queryOffset, warpParameters, gather); break;
      case 4 : hammingTopKUnrollWarp<IS, 4, NumCodeRows, BytesPerVector>(queryOffset, warpParameters, gather); break;
      case 2 : hammingTopKUnrollWarp<IS, 2, NumCodeRows, BytesPerVector>(queryOffset, warpParameters, gather); break;
      case 1 : hammingTopKUnrollWarp<IS, 1, NumCodeRows, BytesPerVector>(queryOffset, warpParameters, gather); break;
      default: ABORT("Unhandled warpSize = {}??", warpSize);
    }
  }
}

} // namespace unrolled

// this namespace contains the dynamic version of the hammingTopK function which is slower than the unrolled version
// Here the number of code rows is not known at compile time. Otherwise the code is the same as in the unrolled namespace.
namespace dynamic {

template <marian::cpu::InstructionSet IS, int warpSize, int bytesPerVector, class Functor>
inline void hammingTopKUnrollWarp(int queryOffset, const Parameters& parameters, const Functor& gather) {
  constexpr int numBits = bytesPerVector * 8;
  static_assert(numBits % 64 == 0, "LSH hash size must be a multiple of 64");

  const int numCodeRows = parameters.numCodeRows;

  // we make these static and thread_local to avoid re-allocating them for every warp. In the unrolled version
  // we can use std::array which is stack allocated and does not need to be thread_local.
  // Counter to keep track of seen hamming distances
  static thread_local std::vector<std::vector<DistType>> counter(warpSize, std::vector<DistType>(numBits));
  counter.resize(warpSize);

  // Buffer the distances for query vector warpRowId to all weight vectors codeRowId
  static thread_local std::vector<std::vector<DistType>> distBuffer(warpSize, std::vector<DistType>(numCodeRows));
  distBuffer.resize(warpSize);

  // Minimal distances per query
  static thread_local std::vector<DistType> minDist(warpSize);
  minDist.resize(warpSize);

  constexpr int StepsStatic = bytesPerVector / sizeof(ChunkType);
  ChunkType* codeRow = (ChunkType*)parameters.codeRows;

  for(int warpRowId = 0; warpRowId < warpSize; warpRowId++) {
    std::fill(counter[warpRowId].begin(), counter[warpRowId].end(), 0);
    minDist[warpRowId] = (DistType)numBits;
  }

  for(IndexType codeRowId = 0; codeRowId < (IndexType)numCodeRows; ++codeRowId, codeRow += StepsStatic) {
    ChunkType* queryRow = (ChunkType*)parameters.queryRows;
    for(IndexType warpRowId = 0; warpRowId < warpSize; warpRowId++, queryRow += StepsStatic) {
      // Compute the bit-wise hamming distance
      const DistType dist = hamming<IS, StepsStatic>(queryRow, codeRow);

      // Record the minimal distance seen for this query vector wrt. all weight vectors
      if(dist < minDist[warpRowId]) {
        minDist[warpRowId] = dist;
      }

      // Record the number of weight vectors that have this distance from the query vector.
      // Note, because there is at most numBits different distances this can be trivially done.
      // Not the case for generic distances like float.
      counter[warpRowId][dist]++;

      // Record the distance for this weight vector
      distBuffer[warpRowId][codeRowId] = dist;
    }
  }
  // warp finished, harvest k top distances

  for(int warpRowId = 0; warpRowId < warpSize; warpRowId++) {
    // Here we search for the distance at which we have seen equal or more than k elements with
    // smaller distances. We start with the minimal distance from above which is its own address
    // to the counter.
    DistType maxDist = minDist[warpRowId];
    size_t cummulativeDistances = 0;

    // Accumulate number of elements until we reach k in growing distance order. Note that
    // counter is indexed by hamming distance - from lowest to highest. Some slots will be 0.
    // The cumulative sum from position a to b tells you how many elements have distances smaller
    // than the distance at b.
    while(cummulativeDistances < parameters.k)
      cummulativeDistances += counter[warpRowId][maxDist++];
    if(cummulativeDistances)
      maxDist--; // fix overcounting

    // Usually, we overshoot by a couple of elements and we need to take care of the distance at which the k-th
    // element sits. This elements has more neighbors at the same distance, but we only care for them
    // as long we have not reached k elements in total.
    // By contrast, we trivially collect all elements below that distance -- these are always safe.

    // This is the number of elements we need to collect at the last distance.
    const DistType maxDistLimit = /*number of elements at maxDist=*/counter[warpRowId][maxDist] - /*overflow=*/((DistType)cummulativeDistances - (DistType)parameters.k);
    IndexType kSeen = 0;
    IndexType kSeenAtKDist = 0;

    for(IndexType codeRowId = 0; kSeen < (IndexType)parameters.k && codeRowId < (IndexType)numCodeRows; ++codeRowId) {
      DistType dist = distBuffer[warpRowId][codeRowId];
      // - if the current distance is smaller than the maxDist, just consume.
      // - if the distance is equal to maxDist, make sure to only consume maxDistLimit elements at maxDist
      //   and ignore the rest (smaller indices make it in first).
      // - after we finish this loop we have exactly k top values for every query row in original index order.
      int queryRowId = queryOffset + warpRowId;
      if(dist < maxDist) {
        gather(queryRowId, (IndexType)kSeen, codeRowId, dist);
        kSeen++;
      } else if(dist == maxDist && kSeenAtKDist < (DistType)maxDistLimit) {
        gather(queryRowId, (IndexType)kSeen, codeRowId, dist);
        kSeen++;
        kSeenAtKDist++;
      }
    }
  }
}

// Faster top-k search for hamming distance. The idea here is that instead of sorting the elements we find a hamming distances at which it is safe
// to copy the given index. Copying only the indices below that distance is guaranteed to results in no more than k elements. For elements at that
// distance we need to correct for overshooting.
// Once we have that distance we only need to traverse the set of distances. In the end we get exactly k elements per queryRows vector.
template <marian::cpu::InstructionSet IS, int bytesPerVector, class Functor>
inline void hammingTopKUnroll(const Parameters& parameters, const Functor& gather) {
  int warpSize = 4; // starting warpSize of 4 seems optimal
  auto warpParameters = parameters;
  for(int queryOffset = 0; queryOffset < parameters.numQueryRows; queryOffset += warpSize) {
    while(parameters.numQueryRows - queryOffset < warpSize)
      warpSize /= 2;

    warpParameters.queryRows    = parameters.queryRows + queryOffset * bytesPerVector;
    warpParameters.numQueryRows = warpSize;
    switch(warpSize) {
      case 8 : hammingTopKUnrollWarp<IS, 8, bytesPerVector>(queryOffset, warpParameters, gather); break;
      case 4 : hammingTopKUnrollWarp<IS, 4, bytesPerVector>(queryOffset, warpParameters, gather); break;
      case 2 : hammingTopKUnrollWarp<IS, 2, bytesPerVector>(queryOffset, warpParameters, gather); break;
      case 1 : hammingTopKUnrollWarp<IS, 1, bytesPerVector>(queryOffset, warpParameters, gather); break;
      default: ABORT("Unhandled warpSize = {}??", warpSize);
    }
  }
}

} // namespace dynamic

template <marian::cpu::InstructionSet IS>
inline void hammingTopK(const Parameters& parameters, const GatherFn& gather) {
  if(parameters.numCodeRows == 32000 && parameters.bytesPerVector ==  512 / 8) {
    unrolled::hammingTopKUnroll<IS, 32000,  64>(parameters, gather);
  } else if(parameters.numCodeRows == 32000 && parameters.bytesPerVector == 1024 / 8) {
    unrolled::hammingTopKUnroll<IS, 32000, 128>(parameters, gather);
  } else if(parameters.bytesPerVector ==  512 / 8) {
    dynamic::hammingTopKUnroll<IS, 64>(parameters, gather);
  } else if(parameters.bytesPerVector == 1024 / 8) {
    dynamic::hammingTopKUnroll<IS, 128>(parameters, gather);
  } else {
    ABORT("Unsupported number of bytes per vector {}", parameters.bytesPerVector);
  }
}

} // namespace lsh
} // namespace marian