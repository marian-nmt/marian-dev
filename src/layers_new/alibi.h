#pragma once

#include "models/states.h"
#include "layers_new/attention.h"
#include "layers_new/decoder.h"
#include "layers_new/neuralnet.h"

namespace marian {

// @TODO: this whole set of functions is currently somewhat akward in general, since we need to implement
// old style and new style decoder state for this to work. We decoder with the old decoder framework, but
// use the new style transformer layers. This will eventually be cleaned up.

// Specialized version of DecoderState for model that knows about algorithmic ALIBI position shifts
class AlibiDecoderState : public DecoderState {
private:
  typedef std::tuple<int, int, int> SyncCoord;
  mutable std::vector<SyncCoord> syncPoints_;
  int lastBeam_{1};

public:
  AlibiDecoderState(const rnn::States& states,
                    Logits logProbs,
                    const std::vector<Ptr<EncoderState>>& encStates,
                    Ptr<data::CorpusBatch> batch,
                    bool isBatchMajor = false);

  // override to create derived decoder states
  virtual Ptr<DecoderState> Create(const rnn::States& states,
                                   Logits logProbs,
                                   const std::vector<Ptr<EncoderState>>& encStates,
                                   Ptr<data::CorpusBatch> batch,
                                   bool isBatchMajor = false) const override;

  // expand the decoder state
  virtual Ptr<DecoderState> next(const rnn::States& states,
                                 Logits logProbs) const override;

  // select the hypotheses based on beam search indices
  virtual Ptr<DecoderState> select(
      const std::vector<IndexType>& hypIndices,    // [beamIndex * activeBatchSize + batchIndex]
      const Words& words,
      const std::vector<IndexType>& batchIndices,  // [batchIndex]
      int beamSize) const override;

  // get the alibi shift for the current state based on currently stored sync points computed while decoding
  Expr getAlibiShift(Ptr<ExpressionGraph> graph, bool decoding) const;

  // get the alibi shift based on the batch data - this is used during training or scoring where ground truth is available
  Expr getAlibiShiftFromBatch(Ptr<ExpressionGraph> graph) const;

private:

  // compute the sync points for the current state based on the previous sync points and the last generated words.
  // This happens one step at a time while decoding.
  std::vector<SyncCoord> computeSyncPoints(
      const std::vector<IndexType>& hypIndices,   // [beamIndex * activeBatchSize + batchIndex]
      const Words& words,                         // [beamIndex * activeBatchSize + batchIndex]
      const std::vector<IndexType>& batchIndices, // [batchIndex] of activeBatchSize
      int beamSize
  ) const;
};

// create a new (alibi) decoder state
Ptr<DecoderState> NewDecoderState(Ptr<Options> options,
                                  const rnn::States& states,
                                  Logits logProbs,
                                  const std::vector<Ptr<EncoderState>>& encStates,
                                  Ptr<data::CorpusBatch> batch,
                                  bool isBatchMajor = false);

// convert an old-style decoder state to an (alibi) decoder state
Ptr<nn::DecoderState> convertDecoderState(Ptr<DecoderState> state, 
                                          Ptr<ExpressionGraph> graph, 
                                          bool decoding=false);

// efficient operator for ALIBI log mask with shift and optionally learnable parameters
Expr alibiLogMask(Expr mask, Expr query, Expr shift, Expr slopes, Expr biases, int numHeads, int start);

namespace nn {

class AlibiDecoderStateItem : public DecoderStateItem {
private:
  Expr shift_;

public:
  AlibiDecoderStateItem(Expr state, Expr shift, size_t position) : DecoderStateItem(state, position), shift_(shift) {}
  virtual ~AlibiDecoderStateItem() = default;

  Expr getShift() const {
    return shift_;
  }
};

// Experimental implementation of the ALIBI attention mechanism (via masking) (https://arxiv.org/abs/2108.12409)
class AlibiAttentionMaskProcessor : public AttentionMaskProcessor {
public:
  bool trainable{false};    // if true don't use learnable parameters

  Expr slopes;  // learnable per head ALIBI slopes
  Expr biases;  // learnable per head additive biases
  
  using AttentionMaskProcessor::numHeads;

  AlibiAttentionMaskProcessor(Ptr<ExpressionGraph> graph,
                              Ptr<Options> options)
    : AttentionMaskProcessor(graph, options),
      trainable(options->get<bool>("transformer-alibi-trainable", false))
    {}

  virtual ~AlibiAttentionMaskProcessor() = default;

private:  
// @TODO: eventually to be removed. This computes ALIBI log masks with multiple operators, replaced with more efficient version below.
// For now we keep this for documentation and experimentation puprposes.
// The same functionality is implemented in `alibiLogMask` above via a special operator
#if 0
  const float ALIBI_REFERENCE_HEADS{8.f}; // number of reference heads that ALIBI slopes are computed for

  // Compute the alibi mask for a given query and keys
  Expr alibiMask(Expr query, int dimQuery, int dimKeys, Ptr<DecoderState> state) const {
    int start = 0;
    Expr shift = nullptr;

    int dimBatch = query->shape()[-3];
    int dimBeam  = query->shape()[-4];

    if(state) {
      start = (int)state->getPosition();
      auto alibiState = std::dynamic_pointer_cast<AlibiDecoderStateItem>(state);
      shift = alibiState ? alibiState->getShift() : nullptr; // [dimBeam, dimBatch, dimQuery, 1]
    }
    
    // Create constant tensors of reflecting the query and key positions.
    // When decoding, we start with the decoding state position for the query. The key positions are just the indices for the whole sequence.
    Expr queryPositions = graph()->constant({1, 1, dimQuery, 1}, inits::range((float)start, (float)(start + dimQuery)));  // [1, 1, dimQuery, 1]
    Expr keyPositions   = graph()->constant({1, 1, 1,  dimKeys}, inits::range(0.f, (float)dimKeys));                      // [1, 1, 1, dimKeys]
    
    // Create matrix of distances between positions, rows are distances of current query position vs all key positions.
    // Layout is the same as the attention distance matrix where we compute rowwise softmaxes of similarities between
    // each target word and all the source words
    Expr alibiBiases = keyPositions - queryPositions; // [1, 1, dimQuery, dimKeys]

    // apply the corrective shift if any sync-points are present
    if(shift) {
      alibiBiases = alibiBiases - shift;                                              // [dimBeam, dimBatch, dimQuery, dimKeys]
      alibiBiases = reshape(alibiBiases, {dimBeam * dimBatch, 1, dimQuery, dimKeys}); // [dimBeam * dimBatch, 1, dimQuery, dimKeys]
    }

    Expr alibi = slopes * abs(alibiBiases + biases);  // [(dimBeam * dimBatch)|1, numHeads, dimQuery, dimKeys]
    return alibi;
  };

  // Compute the log mask for a given query and combine with the alibi mask
  Expr logMask(Expr query, Expr mask, Ptr<DecoderState> state) const {
    ABORT_IF(!mask, "mask is expected!!");

    // query: [dimBeam, dimBatch, dimQuery, dimModel] -> dimQuery == dimTrgWords
    int dimBatch = query->shape()[-3];
    int dimBeam  = query->shape()[-4];
    
    int dimQuery = query->shape()[-2];
    int dimKeys  = mask->shape()[-2];
    
    // all this is bascially a copy of the normal attention mask computation, however we need to do some extra reshaping
    // to make the alibi mask and the log mask broadcastable and then combine them via minimum

    // Note, this is not a typical logMask with values 0 (don't mask) and -inf (mask). Rather we use +inf (or a large value) 
    // and -inf and then compbine with the ALIBI mask via minimum. This way, we keep the original ALIBI values where the mask has
    // +inf and have -inf for masking.
     // largest useful value and making sure we do not overflow for fp16
    float maskFactor = std::min(NumericLimits<float>(mask->value_type()).max / 2.f, 99999999.f);
    // convert binary 0/1 mask to -1/1 mask and then muliply with inf, results in -inf/+inf mask.
    auto logMask = (2.f * mask - 1.f) * maskFactor; // [1, dimBatch, dimKeys, 1]
    logMask = reshape(logMask, {dimBatch, 1, 1, dimKeys});       // [dimBatch,                      1,        1, dimKeys]
    

    // make logMask broadcastable when decoding with beam search
    logMask = repeat(logMask, /*repeats=*/dimBeam, /*axis=*/-4); // [dimBeam|1 * dimBatch,          1,        1, dimKeys]
    
    // make logMask and alibiBias broadcastable, then combine
    auto alibiBias = alibiMask(query, dimQuery, dimKeys, state); // [(dimBeam * dimBatch)|1, numHeads, dimQuery, dimKeys]
    logMask = minimum(logMask, alibiBias);                       // [dimBeam|1 * dimBatch,   numHeads, dimQuery, dimKeys]

    // final reshape to match attention operation
    logMask = reshape(logMask, {dimBeam, dimBatch * numHeads, dimQuery, dimKeys}); // [dimBeam|1, dimBatch * numHeads, dimQuery, dimKeys]
    return logMask;
  }
#endif

  // Initialized the head-wise scaling factors from ALIBI (they are constant in the original paper,
  // we are making them optionally learnable here)
  Ptr<inits::NodeInitializer> initSlopes(bool decoder = false) const {
// This is the original implementation of ALIBI slopes for LMs. We find our slopes and biases work better for Seq2seq models
// Keep for now until we find a use, e.g. in LMs
#if 0
    std::vector<float> mVec(numHeads);
    for(size_t i = 0; i < numHeads; ++i) {
      // slopes in the paper go from 1/2^1 to 1/2^8 where 8 is the reference number of heads;
      // if there are more or less heads we scale back to 8 heads and interpolate.
      float exponent = (float)(i + 1) * (ALIBI_REFERENCE_HEADS / (float)numHeads);

      // We multiply slopes with 2 for the symmetric mask to keep total probability mass the 
      // same as in the causal mask (we have two symmetric halves instead of just one causal half)
      mVec[i] = -2.f / std::pow(2.f, exponent);
      if(decoder)
        mVec[i] *= 0.5f;
    }
    
    return inits::fromVector(mVec);
#else
    // Magic numbers, for now don't ask.
    std::vector<float> init;
    if(decoder) {
      return inits::fromValue(-0.1f);
    } else {
      init = { -2.00f, -1.00f, -0.50f, -0.25f, -0.05f, -0.05f, -0.05f, -0.05f };
      init.resize(numHeads, -0.05f);
      return inits::fromVector(init);
    }
#endif
  }

  // Head-wise biases for ALIBI, this does not occur in the paper, ignore the magic numbers
  Ptr<inits::NodeInitializer> initBiases(bool decoder=false) const {
    if(decoder) {
      return inits::fromValue(0.3f);
    } else {
      std::vector<float> init({ 1.00f, -2.00f, 3.00f, -4.00f, 5.00f, -6.00f, 7.00f, -8.00f });
      init.resize(numHeads, 0.f);
      return inits::fromVector(init);
    }
  }

public:
  // Apply the alibi mask to the given query and mask
  virtual Expr apply(Expr query, Expr mask) const override {
    return apply(query, mask, /*state=*/nullptr);
  }

  // Apply the alibi mask to the given query and mask for decoder cross-attention
  virtual Expr apply(Expr query, Expr mask, Ptr<DecoderState> state) const override {
    bool decoder = state != nullptr;

    if(!trainable) {
      const_cast<Expr&>(slopes) = graph()->constant({numHeads, 1, 1}, initSlopes(decoder));
      const_cast<Expr&>(biases) = graph()->constant({numHeads, 1, 1}, initBiases(decoder));
    } else {
      registerParameterLazy(slopes, Shape({numHeads, 1, 1}), initSlopes(decoder));
      registerParameterLazy(biases, Shape({numHeads, 1, 1}), initBiases(decoder));
    }

    Expr shift = nullptr;
    int start = 0;
    
    if(state) {
      start = (int)state->getPosition();
      auto alibiState = std::dynamic_pointer_cast<AlibiDecoderStateItem>(state);
      shift = alibiState ? alibiState->getShift() : nullptr; // [dimBeam, dimBatch, dimQuery, 1]
    }

    auto alibiMask = alibiLogMask(mask, query, slopes, biases, shift, numHeads, start);
    return alibiMask;
  }
};

} // namespace nn
} // namespace marian