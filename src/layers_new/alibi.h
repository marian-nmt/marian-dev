#pragma once

#include "models/states.h"
#include "layers_new/attention.h"
#include "layers_new/decoder.h"
#include "layers_new/neuralnet.h"

namespace marian {

const int ALIBI_REFERENCE_HEADS = 8; // number of heads in the reference model

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
Expr alibiLogMask(Expr mask, Expr query, Expr shift, Expr slopes, Expr biases, int numHeads, int start, bool addCausalMask = false);

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

/**
 * Experimental implementation of the ALIBI attention mechanism (via masking) (https://arxiv.org/abs/2108.12409)
 */
class AlibiAttentionMaskProcessor : public AttentionMaskProcessor {
public:
  bool trainable{false}; // if true don't use learnable parameters

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

  // Initialized the head-wise scaling factors from ALIBI (they are constant in the original paper,
  // we are making them optionally learnable here)
  Ptr<inits::NodeInitializer> initSlopes() const {
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
    std::vector<float> init = { -2.00f, -1.00f, -0.50f, -0.25f, -0.05f, -0.05f, -0.05f, -0.05f };
    init.resize(numHeads, -0.05f);
    return inits::fromVector(init);
#endif
  }

  // Head-wise biases for ALIBI, this does not occur in the paper, ignore the magic numbers
  Ptr<inits::NodeInitializer> initBiases() const {
    std::vector<float> init({ 1.00f, -2.00f, 3.00f, -4.00f, 5.00f, -6.00f, 7.00f, -8.00f });
    init.resize(numHeads, 0.f);
    return inits::fromVector(init);
  }

public:

  // Apply the alibi mask to the given query and mask
  virtual Expr apply(Expr query, Expr mask) const override {
    if(!trainable) {
      const_cast<Expr&>(slopes) = graph()->constant({numHeads, 1, 1}, initSlopes());
      const_cast<Expr&>(biases) = graph()->constant({numHeads, 1, 1}, initBiases());
    } else {
      registerParameterLazy(slopes, Shape({numHeads, 1, 1}), initSlopes());
      registerParameterLazy(biases, Shape({numHeads, 1, 1}), initBiases());
    }

    Expr shift = nullptr;
    int start = 0;

    auto alibiMask = alibiLogMask(mask, query, slopes, biases, shift, numHeads, start);
    return alibiMask;
  }
};

/**
 * Experimental implementation of the ALIBI attention mechanism for decoder layers
 */
class AlibiDecoderAttentionMaskProcessor : public DecoderAttentionMaskProcessor {
public:
  bool trainable{false}; // if true don't use learnable parameters

  Expr slopes;  // learnable per head ALIBI slopes
  Expr biases;  // learnable per head additive biases

  using DecoderAttentionMaskProcessor::numHeads;

  AlibiDecoderAttentionMaskProcessor(Ptr<ExpressionGraph> graph,
                                     Ptr<Options> options,
                                     bool addCausalMask = false)
    : DecoderAttentionMaskProcessor(graph, options, addCausalMask),
      trainable(options->get<bool>("transformer-alibi-trainable", false)) {}

  virtual ~AlibiDecoderAttentionMaskProcessor() = default;

private:
  // Initialized the head-wise scaling factors from ALIBI (they are constant in the original paper,
  // we are making them optionally learnable here)
  Ptr<inits::NodeInitializer> initSlopes() const {
    if(addCausalMask) {
      std::vector<float> mVec(numHeads);
      for(size_t i = 0; i < numHeads; ++i) {
        // slopes in the paper go from 1/2^1 to 1/2^8 where 8 is the reference number of heads;
        // if there are more or less heads we scale back to 8 heads and interpolate.
        float exponent = (float)(i + 1) * (ALIBI_REFERENCE_HEADS / (float)numHeads);
        mVec[i] = -1.f / std::pow(2.f, exponent);
      }
      return inits::fromVector(mVec);
    } else {
      return inits::fromValue(-0.1f); // Magic numbers, for now don't ask.
    }
  }

  // Head-wise biases for ALIBI, this does not occur in the paper, ignore the magic numbers
  Ptr<inits::NodeInitializer> initBiases() const {
    if(addCausalMask) {
      return inits::fromValue(0.0f);
    } else {
      return inits::fromValue(0.3f);
    }
  }

public:
  // Apply the alibi mask to the given query and mask for decoder cross-attention
  virtual Expr apply(Expr query, Expr mask, Ptr<DecoderState> state) const override {
    auto processMask = [this, query, state](Expr mask) {
      if(!trainable) {
        const_cast<Expr&>(slopes) = graph()->constant({numHeads, 1, 1}, initSlopes());
        const_cast<Expr&>(biases) = graph()->constant({numHeads, 1, 1}, initBiases());
      } else {
        registerParameterLazy(slopes, Shape({numHeads, 1, 1}), initSlopes());
        registerParameterLazy(biases, Shape({numHeads, 1, 1}), initBiases());
      }

      Expr shift = nullptr;
      int start = 0;

      if(state) {
        start = (int)state->getPosition();
        auto alibiState = std::dynamic_pointer_cast<AlibiDecoderStateItem>(state);
        shift = alibiState ? alibiState->getShift() : nullptr; // [dimBeam, dimBatch, dimQuery, 1]
      }

      // @TODO: make sure that we never want to have a causal mask here if start > 0 (this should indicate decoding)
      return alibiLogMask(mask, query, slopes, biases, shift, numHeads, start, addCausalMask && start == 0);
    };

    if(mask) {
      // recompute the mask if input mask changes (different memory address), otherwise return cached version
      auto equal = [](Expr a, Expr b) { return a == b; };
      return cachedMask_->apply(mask, processMask, equal);
    } else {
      // @TODO: avoid this mask recreation for every layer
      int dimBatch   = query->shape()[-3];
      int dimKeys    = (int)state->getPosition() + 1;
      mask = graph()->constant({1, dimBatch, dimKeys, 1}, inits::ones());

      // recompute the ALIBI mask if shape changes, but still has to create the above temporary mask first
      auto equal = [](Expr a, Expr b) { return a->shape() == b->shape(); };
      return cachedMask_->apply(mask, processMask, equal);
    }
  }
};

} // namespace nn
} // namespace marian