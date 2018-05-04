#pragma once

#include "marian.h"
#include "models/states.h"
#include "rnn/types.h"

namespace marian {

namespace rnn {

Expr attOps(Expr va, Expr context, Expr state);

class GlobalAttention : public CellInput {
private:
  Expr Wa_, ba_, Ua_;
  std::vector<Expr>vas_;

  Expr gammaContext_;
  Expr gammaState_;

  Ptr<EncoderState> encState_;
  Expr softmaxMask_;
  Expr mappedContext_;
  std::vector<Expr> contexts_;
  std::vector<Expr> alignments_;
  bool layerNorm_;
  float dropout_;

  Expr contextDropped_;
  Expr dropMaskContext_;
  Expr dropMaskState_;

  // for Nematus-style layer normalization
  Expr Wc_att_lns_, Wc_att_lnb_;
  Expr W_comb_att_lns_, W_comb_att_lnb_;
  bool nematusNorm_;

  int numAttentionHeads_;
  int attentionProjectionDim_;
  bool attentionProjectionLayerNorm_;
  bool attentionProjectionTanH_;

  std::vector<Expr> attentionProjectionMatrices_;
  std::vector<Expr> attentionProjectionMatrixGammas_;
  std::vector<Expr> attentionProjectionMatrixBs_;

public:
  GlobalAttention(Ptr<ExpressionGraph> graph,
                  Ptr<Options> options,
                  Ptr<EncoderState> encState)
      : CellInput(options),
        encState_(encState),
        contextDropped_(encState->getContext()) {
    int dimDecState = options_->get<int>("dimState");
    dropout_ = options_->get<float>("dropout", 0);
    layerNorm_ = options_->get<bool>("layer-normalization", false);
    nematusNorm_ = options_->get<bool>("nematus-normalization", false);
    numAttentionHeads_ = options_->get<int>("attentionHeads", 1);
    attentionProjectionDim_ = options_->get<int>("attentionProjectionDim", -1);
    attentionProjectionLayerNorm_ = options->get<bool>("attentionProjectionLayerNorm", false);
    attentionProjectionTanH_ = options->get<bool>("attentionProjectionTanH", false);
    std::string prefix = options_->get<std::string>("prefix");
    //LOG(info, "Attention heads: {}", numAttentionHeads_);

    int dimEncState = encState_->getContext()->shape()[-1];

    Wa_ = graph->param(prefix + "_W_comb_att",
                       {dimDecState, dimEncState},
                       inits::glorot_uniform);
    Ua_ = graph->param(
        prefix + "_Wc_att", {dimEncState, dimEncState}, inits::glorot_uniform);
    for (int headI = 0; headI < numAttentionHeads_; ++headI) {
      std::string suffix;
      if (headI > 0) {
        suffix += "_" + std::to_string(headI);
      }
      vas_.push_back(graph->param(
          prefix + "_U_att" + suffix, {dimEncState, 1}, inits::glorot_uniform));

      // Attended context projection
      if (attentionProjectionDim_ != -1) {
        attentionProjectionMatrices_.push_back(graph->param(
          prefix + "_projectionMatrix" + suffix, {dimEncState, attentionProjectionDim_}, inits::glorot_uniform));
        if (attentionProjectionLayerNorm_) {
          Expr gamma = graph->param(
            prefix + "_projectionMatrix_gamma" + suffix, {1, attentionProjectionDim_}, inits::from_value(1.0));
          attentionProjectionMatrixGammas_.push_back(gamma);
        }
        Expr beta = (attentionProjectionTanH_)? graph->param(
          prefix + "_projectionMatrix_b" + suffix, {1, attentionProjectionDim_}, inits::zeros):
          nullptr;
        attentionProjectionMatrixBs_.push_back(beta);
      }
    }
    ba_ = graph->param(prefix + "_b_att", {1, dimEncState}, inits::zeros);

    if(dropout_ > 0.0f) {
      dropMaskContext_ = graph->dropout(dropout_, {1, dimEncState});
      dropMaskState_ = graph->dropout(dropout_, {1, dimDecState});
    }

    if(dropMaskContext_)
      contextDropped_ = dropout(contextDropped_, dropMaskContext_);

    if(layerNorm_) {
      if(nematusNorm_) {
        // instead of gammaContext_
        Wc_att_lns_ = graph->param(
            prefix + "_Wc_att_lns", {1, dimEncState}, inits::from_value(1.f));
        Wc_att_lnb_ = graph->param(
            prefix + "_Wc_att_lnb", {1, dimEncState}, inits::zeros);
        // instead of gammaState_
        W_comb_att_lns_ = graph->param(prefix + "_W_comb_att_lns",
                                       {1, dimEncState},
                                       inits::from_value(1.f));
        W_comb_att_lnb_ = graph->param(
            prefix + "_W_comb_att_lnb", {1, dimEncState}, inits::zeros);

        mappedContext_ = layer_norm(affine(contextDropped_, Ua_, ba_),
                                    Wc_att_lns_,
                                    Wc_att_lnb_,
                                    NEMATUS_LN_EPS);
      } else {
        gammaContext_ = graph->param(
            prefix + "_att_gamma1", {1, dimEncState}, inits::from_value(1.0));
        gammaState_ = graph->param(
            prefix + "_att_gamma2", {1, dimEncState}, inits::from_value(1.0));

        mappedContext_
            = layer_norm(dot(contextDropped_, Ua_), gammaContext_, ba_);
      }

    } else {
      mappedContext_ = affine(contextDropped_, Ua_, ba_);
    }

    auto softmaxMask = encState_->getMask();
    if(softmaxMask) {
      Shape shape = {softmaxMask->shape()[-3], softmaxMask->shape()[-2]};
      softmaxMask_ = transpose(reshape(softmaxMask, shape));
    }

  }

  Expr apply(State state) {
    using namespace keywords;
    auto recState = state.output;

    int dimBatch = contextDropped_->shape()[-2];
    int srcWords = contextDropped_->shape()[-3];
    int dimBeam = 1;
    if(recState->shape().size() > 3)
      dimBeam = recState->shape()[-4];

    if(dropMaskState_)
      recState = dropout(recState, dropMaskState_);

    auto mappedState = dot(recState, Wa_);
    if(layerNorm_)
      if(nematusNorm_)
        mappedState = layer_norm(
            mappedState, W_comb_att_lns_, W_comb_att_lnb_, NEMATUS_LN_EPS);
      else
        mappedState = layer_norm(mappedState, gammaState_);

    Expr first_e;
    std::vector<Expr> alignedSources;
    for (int headI = 0; headI < numAttentionHeads_; ++headI) { 	// TODO: Compute all attention heads in a single CUDA kernel

      auto attReduce = attOps(vas_[headI], mappedContext_, mappedState);

      // @TODO: horrible ->
      auto e = reshape(transpose(softmax(transpose(attReduce), softmaxMask_)),
                       {dimBeam, srcWords, dimBatch, 1});
      // <- horrible

      auto alignedSource = scalar_product(encState_->getAttended(), e, axis = -3);

      // Attended context projection
      //LOG(info, "alignedSource shape (before proj): {}", alignedSource->shape());
      if (attentionProjectionDim_ != -1) {
        alignedSource = dot(alignedSource, attentionProjectionMatrices_[headI]);
        //LOG(info, "alignedSource shape (after proj): {}", alignedSource->shape());
        if (attentionProjectionLayerNorm_) {
          alignedSource = layer_norm(alignedSource, attentionProjectionMatrixGammas_[headI], attentionProjectionMatrixBs_[headI]);
        }
        if (attentionProjectionTanH_) {
          if (attentionProjectionLayerNorm_) {
            alignedSource = tanh(alignedSource);
          }
          else {
            alignedSource = tanh(alignedSource + attentionProjectionMatrixBs_[headI]);
          }
        }
      }
      alignedSources.push_back(alignedSource);
      if (headI == 0) {
        // Note: we return the first set of attention weights
        // for multi-head attention this might not be necessarily the most relevant one
        first_e = e;
      }
    }

    auto concatenatedAlignedSources = concatenate(alignedSources, axis=-1);
    contexts_.push_back(concatenatedAlignedSources);
    alignments_.push_back(first_e);
    return concatenatedAlignedSources;
  }

  std::vector<Expr>& getContexts() { return contexts_; }

  Expr getContext() { return concatenate(contexts_, keywords::axis = -3); }

  std::vector<Expr>& getAlignments() { return alignments_; }

  virtual void clear() {
    contexts_.clear();
    alignments_.clear();
  }

  int dimOutput() {
    int alignedSourcesDimPerHead = (attentionProjectionDim_ == -1)? (encState_->getContext()->shape()[-1]): attentionProjectionDim_;
    return alignedSourcesDimPerHead * numAttentionHeads_;
  }
};

using Attention = GlobalAttention;
}
}
