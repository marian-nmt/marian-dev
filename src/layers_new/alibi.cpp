#include "graph/node_operators_unary.h"
#include "layers_new/alibi.h"

namespace marian {

AlibiDecoderState::AlibiDecoderState(const rnn::States& states,
                                     Logits logProbs,
                                     const std::vector<Ptr<EncoderState>>& encStates,
                                     Ptr<data::CorpusBatch> batch,
                                     bool isBatchMajor)
: DecoderState(states, logProbs, encStates, batch, isBatchMajor) {}

// override to create derived decoder states
Ptr<DecoderState> AlibiDecoderState::Create(const rnn::States& states,
                                            Logits logProbs,
                                            const std::vector<Ptr<EncoderState>>& encStates,
                                            Ptr<data::CorpusBatch> batch,
                                            bool isBatchMajor) const {
  return New<AlibiDecoderState>(states, logProbs, encStates, batch, isBatchMajor);
}

// expand the decoder state
Ptr<DecoderState> AlibiDecoderState::next(const rnn::States& states,
                                          Logits logProbs) const {
  // expand the previous decoder state via the base class expansion
  auto state = std::dynamic_pointer_cast<AlibiDecoderState>(DecoderState::next(states, logProbs));
  // this should always succeed, unless we somehow messed up inheritance
  ABORT_IF(!state, "state is nullptr, i.e. the conversion to AlibiDecoderState failed??");

  // carry over the sync points and last beam size from the previous state
  state->syncPoints_ = syncPoints_;
  state->lastBeam_   = lastBeam_;
  return state;
}

// select the hypotheses based on beam search indices
Ptr<DecoderState> AlibiDecoderState::select(
    const std::vector<IndexType>& hypIndices,    // [beamIndex * activeBatchSize + batchIndex]
    const Words& words,
    const std::vector<IndexType>& batchIndices,  // [batchIndex]
    int beamSize) const {
  // select the hypotheses via the base class selection
  auto state = std::dynamic_pointer_cast<AlibiDecoderState>(DecoderState::select(hypIndices, words, batchIndices, beamSize));
  // this should always succeed, unless we somehow messed up inheritance
  ABORT_IF(!state, "state is nullptr, i.e. the conversion to AlibiDecoderState failed??");
  // compute the new sync points and carry over the current beam size
  // this is the most important part of the algorithm while decoding
  state->syncPoints_ = computeSyncPoints(hypIndices, words, batchIndices, beamSize);
  state->lastBeam_   = beamSize;
  return state;
}

// get the alibi shift for the current state based on currently stored sync points computed while decoding
Expr AlibiDecoderState::getAlibiShift(Ptr<ExpressionGraph> graph, bool decoding) const {
  if(decoding) {
    std::vector<float> shift;
    for(const auto& [trgPos, srcPos, batchIdx] : syncPoints_)
      shift.push_back((float)(srcPos - trgPos));
    
    if(!shift.empty()) {
      int dimBeam  = lastBeam_;
      ABORT_IF(dimBeam == 0, "dimBeam is 0??");
      int dimBatch = (int)shift.size() / dimBeam;
      return graph->constant({dimBeam, dimBatch, 1, 1}, inits::fromVector(shift)); // [dimBeam, dimBatch, dimTrg=1, 1]
    } else {
      return nullptr;
    }
  } else {
    ABORT_IF(getBatch()->sets() != 2, 
             "--transformer-alibi-shift=true currently only works with batch sets=2");
    return getAlibiShiftFromBatch(graph);
  }
}

// get the alibi shift based on the batch data - this is used during training or scoring where ground truth is available
Expr AlibiDecoderState::getAlibiShiftFromBatch(Ptr<ExpressionGraph> graph) const {
  std::vector<float> shift;

  auto targetBatch = getBatch()->back();
  Word trgSyncSym  = targetBatch->vocab()->getSepId();

  auto locateInTrg = [&targetBatch](int batchIdx, int j) {
    return targetBatch->data()[targetBatch->locate(batchIdx, j)];
  };

  auto sourceBatch = getBatch()->front();
  Word srcSyncSym  = sourceBatch->vocab()->getSepId();

  auto locateInSrc = [&sourceBatch](int batchIdx, int j) {
    return sourceBatch->data()[sourceBatch->locate(batchIdx, j)];
  };

  int dimBatch = (int)targetBatch->batchSize();
  int dimSrc   = (int)sourceBatch->batchWidth();
  int dimTrg   = (int)targetBatch->batchWidth();
  
  for(int batchIdx = 0; batchIdx < dimBatch; ++batchIdx) {
    int trgPos = -1, srcPos = -1;
    for(int i = 0; i < dimTrg; ++i) {
      if(i > 0) { // don't check until we are one word ahead to mimic generation order where we look back by one word (i - 1)
        if(locateInTrg(batchIdx, i - 1) == trgSyncSym) {
          trgPos = i - 1; // record that position
          // now we are looking for the corresponding source position, no need to look backwards
          for(int j = srcPos + 1; j < dimSrc; ++j) {
            if(locateInSrc(batchIdx, j) == srcSyncSym) {
              srcPos = j;
              break;
            }
          }
        }
      }

      shift.push_back((float)(srcPos - trgPos));
    }
  }

  if(!shift.empty()) {
    return graph->constant({1, dimBatch, dimTrg, 1}, inits::fromVector(shift)); // [dimBeam=1, dimBatch, dimTrg, 1]
  } else {
    return nullptr;
  }
}

// compute the sync points for the current state based on the previous sync points and the last generated words.
// This happens one step at a time while decoding.
std::vector<AlibiDecoderState::SyncCoord> AlibiDecoderState::computeSyncPoints(
    const std::vector<IndexType>& hypIndices,   // [beamIndex * activeBatchSize + batchIndex]
    const Words& words,                         // [beamIndex * activeBatchSize + batchIndex]
    const std::vector<IndexType>& batchIndices, // [batchIndex] of activeBatchSize
    int beamSize
) const {
  size_t position = getPosition();

  // get the sync symbols for source and target
  auto sourceBatch   = getBatch()->front();
  Word srcSyncSymbol = sourceBatch->vocab()->getSepId();
  Word trgSyncSymbol = srcSyncSymbol; // @TODO: this is actually wrong, we should make sure to use the correct target vocab

  auto locateInSrc = [&sourceBatch](int batchIdx, int j) {
    return sourceBatch->data()[sourceBatch->locate(batchIdx, j)];
  };

  int dimBatch = (int)batchIndices.size();
  std::vector<SyncCoord> nextSyncPoints;

  // For each hypothesis, create an updated sync point.
  // If the current symbol is not a sync symbol, the sync point is the same as before and gets carried over.
  // If the current symbol is a sync symbol, the sync point target coordinate is updated to the current position
  // and the source coordinate is updated to the next sync symbol in the source sentence.
  for(int i = 0; i < hypIndices.size(); ++i) {
    SyncCoord pos = syncPoints_.empty() 
      ? SyncCoord({-1, -1, (int)batchIndices[i % dimBatch]}) // no sync points yet, initialize with -1 position and current batch index
      : syncPoints_[hypIndices[i]];                          // carry over the sync point from the previous state at first
    auto& [trgPos, srcPos, batchIdx] = pos;

    // note, words were generated at the step before the current position, hence the pos - 1
    if(words[i] == trgSyncSymbol) { // the current word is a sync symbol, so update the sync point
      trgPos = (int)position - 1;
      // find the next sync symbol in the source sentence
      for(int j = srcPos + 1; j < sourceBatch->batchWidth(); ++j) {
        if(locateInSrc(batchIdx, j) == srcSyncSymbol) { // found the next sync symbol in the source
          srcPos = j; // update the sync point source coordinate
          break;      // and stop looking
        }
      }
    }
    nextSyncPoints.push_back(pos);
  }

  return nextSyncPoints;
} 


Ptr<DecoderState> NewDecoderState(Ptr<Options> options,
                                  const rnn::States& states,
                                  Logits logProbs,
                                  const std::vector<Ptr<EncoderState>>& encStates,
                                  Ptr<data::CorpusBatch> batch,
                                  bool isBatchMajor) {
  if(options->get<bool>("transformer-alibi-shift", false)) {
    ABORT_IF(options->get<std::string>("transformer-attention-mask") != "alibi", "transformer-alibi-shift=true only works with transformer-attention-mask=\"alibi\"");
    return New<AlibiDecoderState>(states, logProbs, encStates, batch, isBatchMajor);
  } else {
    return New<DecoderState>(states, logProbs, encStates, batch, isBatchMajor);
  }
}

Ptr<nn::DecoderState> convertDecoderState(Ptr<DecoderState> state, 
                                          Ptr<ExpressionGraph> graph, 
                                          bool decoding) {
  Expr shift;
  auto alibiState = std::dynamic_pointer_cast<AlibiDecoderState>(state);
  if(alibiState)
    shift = alibiState->getAlibiShift(graph, decoding);

  size_t position = state->getPosition();
  auto nnState = New<nn::DecoderStateList>(position);
  for(auto& layerState : state->getStates()) {
    if(alibiState) {
      nnState->append(New<nn::AlibiDecoderStateItem>(layerState.cell, shift, position));
    } else {
      nnState->append(New<nn::DecoderStateItem>(layerState.cell, position));
    }
  }
  return nnState;
}

#ifdef CUDA_FOUND
namespace gpu {
  template <class... Tensors>
  void Alibi(int numHeads, int start, marian::Tensor out, Tensors... tensors);
}
#endif

namespace cpu {
  template <class... Tensors>
  void Alibi(int numHeads, int start, marian::Tensor out, Tensors... tensors) { 
    ABORT("Not implemented");
  }
}

template <class... Tensors>
void Alibi(int numHeads, int start, marian::Tensor out, Tensors... tensors) {
#ifdef CUDA_FOUND
  if(out->getBackend()->getDeviceId().type == DeviceType::gpu)
    gpu::Alibi(numHeads, start, out, tensors...);
  else
#endif
    cpu::Alibi(numHeads, start, out, tensors...);
}


#ifdef CUDA_FOUND
namespace gpu {
  template <class... Tensors>
  void AlibiGrad(int numHeads, int start, marian::Tensor slopesGrad, marian::Tensor biasesGrad, Tensors... tensors);
}
#endif

namespace cpu {
  template <class... Tensors>
  void AlibiGrad(int numHeads, int start, marian::Tensor slopesGrad, marian::Tensor biasesGrad, Tensors... tensors) { 
    ABORT("Not implemented");
  }
}

template <class... Tensors>
void AlibiGrad(int numHeads, int start, marian::Tensor slopesGrad, marian::Tensor biasesGrad, Tensors... inputs) {
#ifdef CUDA_FOUND
  if(slopesGrad->getBackend()->getDeviceId().type == DeviceType::gpu)
    gpu::AlibiGrad(numHeads, start, slopesGrad, biasesGrad, inputs...);
  else
#endif
    cpu::AlibiGrad(numHeads, start, slopesGrad, biasesGrad, inputs...);
}

class AlibiLogMaskNode : public NaryNodeOp {
private:
  int numHeads_{8};
  int start_{0};

  Shape newShape(Expr mask, Expr query, int numHeads) {
    int dimBeam  = query->shape()[-4];
    int dimBatch = query->shape()[-3];
    int dimQuery = query->shape()[-2];
    int dimKeys  = mask->shape()[-2];

    return { dimBeam, dimBatch * numHeads, dimQuery, dimKeys };
  }

public:
  AlibiLogMaskNode(const std::vector<Expr>& nodes, int numHeads, int start)
  : NaryNodeOp(nodes, newShape(/*mask=*/nodes[0], /*query=*/nodes[1], numHeads), nodes[0]->value_type()), 
    numHeads_(numHeads), start_{start}
  {}

  void forward() override {
    Alibi(
          numHeads_, 
          start_,
          val_, 
          /*mask=*/  child(0)->val(),
          /*slopes=*/child(2)->val(), 
          /*biases=*/child(3)->val(), 
          /*shift=*/ children().size() == 5 ? child(4)->val() : nullptr);
  }

  void backward() override {
    if(!trainable())
      return;
    
    AlibiGrad(
          numHeads_, 
          start_,
          // gradients
          /*d_f/d_slopes=*/child(2)->grad(), 
          /*d_f/d_biases=*/child(3)->grad(), 
          // inputs
          /*mask=*/   child(0)->val(),
          /*slopes=*/ child(2)->val(), 
          /*biases=*/ child(3)->val(), 
          /*shift=*/  children().size() == 5 ? child(4)->val() : nullptr,
          // adjoint
          /*d_J/d_f=*/adj_);
  }

  virtual size_t hash() override {
    size_t seed = NaryNodeOp::hash();
    util::hash_combine(seed, numHeads_);
    util::hash_combine(seed, start_);
    return seed;
  }

  virtual bool equal(Expr node) override {
    if(!NaryNodeOp::equal(node))
      return false;
    auto cnode = std::dynamic_pointer_cast<AlibiLogMaskNode>(node);
    if(!cnode)
      return false;
    if(numHeads_ != cnode->numHeads_)
      return false;
    if(start_ != cnode->start_)
      return false;
    return true;
  }

  const std::string type() override { return "alibi-log-mask"; }
};

Expr alibiLogMask(Expr mask, Expr query, Expr slopes, Expr biases, Expr shift, int numHeads, int start) {
  std::vector<Expr> nodes = {mask, query, slopes, biases};
  if(shift)
    nodes.push_back(shift);

  return Expression<AlibiLogMaskNode>(nodes, numHeads, start);
}


} // namespace marian
