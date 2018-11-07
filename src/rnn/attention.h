#pragma once

#include "marian.h"
#include "models/states.h"
#include "rnn/types.h"

namespace marian {
namespace rnn {

Expr attOps(Expr va, Expr context, Expr state);

class GlobalAttention : public CellInput {
private:
  // Expr vectors are indexed by head id

  std::vector<Expr> Was_, bas_, Uas_;
  std::vector<Expr>vas_;

  std::vector<Expr> gammaContexts_;
  std::vector<Expr> gammaStates_;

  Ptr<EncoderState> encState_;
  Expr softmaxMask_;

  std::vector<Expr> mappedContexts_;
  std::vector<Expr> time_transposed_mapped_contexts_;
  std::vector<Expr> contexts_;
  std::vector<Expr> alignments_;

  bool layerNorm_;
  float dropout_;
  float attentionDropout_;

  Expr contextDropped_;
  Expr dropMaskContext_;
  Expr dropMaskState_;
  Expr dropMaskAttention_;

  // for Nematus-style layer normalization
  std::vector<Expr> Wc_att_lnss_, Wc_att_lnbs_;
  std::vector<Expr> W_comb_att_lnss_, W_comb_att_lnbs_;
  bool nematusNorm_;

  int numAttentionHeads_;             // number of heads for multi-head attention
  int attentionLookupDim_;            // dimension of attention hidden state for MLP attention or
                                      // attention key and query for dot-product attention
  int attentionProjectionDim_;			  // per-head dimension of the projected attended state, if
                                      // projection is enabled
  bool attentionIndependentHeads_;		// whether MLP attention heads have separate hidden states
  bool attentionBilinearLookup_; 		  // whether to use dot-product (bi-linear) attention
  bool attentionProjectionLayerNorm_; // apply layer normalization on attended context after projection

  std::string attentionProjectionActivation_;	// activation function (with bias) on attended
                                              // context after projection, or "identity"

  std::vector<Expr> attentionProjectionMatrices_;
  std::vector<Expr> attentionProjectionMatrixGammas_;
  std::vector<Expr> attentionProjectionMatrixBs_;

  int filteredHeadI(int headI) {return (attentionIndependentHeads_)? headI: 0;}

  std::function<Expr(Expr)> attentionProjectionActivationFcn_;

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

    // TODO: why options_ and options are mixed?
    numAttentionHeads_             = options_->get<int>("attentionHeads", 1);
    attentionLookupDim_            = options_->get<int>("attentionLookupDim", -1);
    attentionProjectionDim_        = options_->get<int>("attentionProjectionDim", -1);
    attentionIndependentHeads_     = options->get<bool>("attentionIndependentHeads", false);
    attentionBilinearLookup_       = options->get<bool>("attentionBilinearLookup", false);
    attentionProjectionLayerNorm_  = options->get<bool>("attentionProjectionLayerNorm", false);
    attentionProjectionActivation_ = options->get<std::string>("attentionProjectionActivation");
    attentionDropout_              = options_->get<float>("attentionDropout", 0);

    std::string prefix = options_->get<std::string>("prefix");

    int dimEncState = encState_->getContext()->shape()[-1];

    // defaults attention hidden state dimension to dimEncState
    attentionLookupDim_ = (attentionLookupDim_ == -1) ? dimEncState : attentionLookupDim_;
    // dot-product attention implies independent heads
    attentionIndependentHeads_ = (attentionBilinearLookup_) ? true : attentionIndependentHeads_;
    // pointer to the projection activation function, if applicable
    attentionProjectionActivationFcn_ = (attentionProjectionActivation_ != "identity")
                                            ? activationByName(attentionProjectionActivation_)
                                            : 0;

    for (int headI = 0; headI < numAttentionHeads_; ++headI) {
      std::string suffix;
      if (headI > 0) {
        suffix += "_" + std::to_string(headI);
      }

      if ((headI == 0) || attentionIndependentHeads_) {
        Was_.push_back(graph->param(prefix + "_W_comb_att" + suffix,
                       {dimDecState, attentionLookupDim_},
                       inits::glorot_uniform));
        Uas_.push_back(graph->param(prefix + "_Wc_att" + suffix, {dimEncState, attentionLookupDim_}, inits::glorot_uniform));
        bas_.push_back(graph->param(prefix + "_b_att" + suffix, {1, attentionLookupDim_}, inits::zeros));
      }


      if (!attentionBilinearLookup_) {
        vas_.push_back(graph->param(
            prefix + "_U_att" + suffix, {attentionLookupDim_, 1}, inits::glorot_uniform));
      }

      // Attended context projection
      if (attentionProjectionDim_ != -1) {
        attentionProjectionMatrices_.push_back(graph->param(
          prefix + "_projectionMatrix" + suffix, {dimEncState, attentionProjectionDim_}, inits::glorot_uniform));
        if (attentionProjectionLayerNorm_) {
          Expr gamma = graph->param(prefix + "_projectionMatrix_gamma" + suffix,
                                    {1, attentionProjectionDim_},
                                    inits::from_value(1.0));
          attentionProjectionMatrixGammas_.push_back(gamma);
        }
        Expr beta = (attentionProjectionActivation_ != "identity")
                        ? graph->param(prefix + "_projectionMatrix_b" + suffix,
                                       {1, attentionProjectionDim_},
                                       inits::zeros)
                        : nullptr;
        attentionProjectionMatrixBs_.push_back(beta);
      }
    }

    if(dropout_ > 0.0f) {
      dropMaskContext_ = graph->dropout(dropout_, {1, dimEncState});
      dropMaskState_ = graph->dropout(dropout_, {1, dimDecState});
    }

    if(dropMaskContext_)
      contextDropped_ = dropout(contextDropped_, dropMaskContext_);

    for (int headI = 0; headI < numAttentionHeads_; ++headI) {
      std::string suffix;
      if (headI > 0) {
        suffix += "_" + std::to_string(headI);
      }

      if ((headI == 0) || attentionIndependentHeads_) {
        if(layerNorm_) {
          if(nematusNorm_) {
            // instead of gammaContext_
            Wc_att_lnss_.push_back(graph->param(
                prefix + "_Wc_att_lns" + suffix, {1, attentionLookupDim_}, inits::from_value(1.f)));
            Wc_att_lnbs_.push_back(graph->param(
                prefix + "_Wc_att_lnb" + suffix, {1, attentionLookupDim_}, inits::zeros));
            // instead of gammaState_
            W_comb_att_lnss_.push_back(graph->param(prefix + "_W_comb_att_lns" + suffix,
                                       {1, attentionLookupDim_},
                                       inits::from_value(1.f)));
            W_comb_att_lnbs_.push_back(graph->param(
                prefix + "_W_comb_att_lnb" + suffix, {1, attentionLookupDim_}, inits::zeros));

            mappedContexts_.push_back(layerNorm(affine(contextDropped_, Uas_[headI], bas_[headI]),
                                                Wc_att_lnss_[headI],
                                                Wc_att_lnbs_[headI],
                                                NEMATUS_LN_EPS));
          } else {
            gammaContexts_.push_back(graph->param(
                prefix + "_att_gamma1" + suffix, {1, attentionLookupDim_}, inits::from_value(1.0)));
            gammaStates_.push_back(graph->param(
                prefix + "_att_gamma2" + suffix, {1, attentionLookupDim_}, inits::from_value(1.0)));

            mappedContexts_.push_back(layerNorm(dot(contextDropped_, Uas_[headI]), gammaContexts_[headI], bas_[headI]));
          }

      } else {
        mappedContexts_.push_back(affine(contextDropped_, Uas_[headI], bas_[headI]));
      }

      if (attentionBilinearLookup_) {
        time_transposed_mapped_contexts_.push_back(
            transpose(reshape(mappedContexts_[headI],
                              {1,
                               mappedContexts_[headI]->shape()[-3],
                               mappedContexts_[headI]->shape()[-2],
                               mappedContexts_[headI]->shape()[-1]}),
                      {0, 2, 1, 3}));
        }
      }
    }

    auto softmaxMask = encState_->getMask();
    if(softmaxMask) {
      Shape shape = {softmaxMask->shape()[-3], softmaxMask->shape()[-2]};
      softmaxMask_ = transpose(reshape(softmaxMask, shape));
    }

    if (attentionBilinearLookup_) {
      mappedContexts_.clear();
    }
  }

  Expr apply(State state) override {
    auto recState = state.output;

    int dimBatch = contextDropped_->shape()[-2];
    int srcWords = contextDropped_->shape()[-3];

    // uncomment the following line to test with a beam size larger than 1 (debug only)
    //recState = repeat(reshape(recState, {1, recState->shape()[0], recState->shape()[1], recState->shape()[2]}), 3);

    int dimBeam = 1;
    if(recState->shape().size() > 3)
      dimBeam = recState->shape()[-4];

    if(dropMaskState_)
      recState = dropout(recState, dropMaskState_);

    Expr mappedState;
    Expr first_e;
    std::vector<Expr> alignedSources;
    for (int headI = 0; headI < numAttentionHeads_; ++headI) { 	// TODO: Compute all attention heads in a single CUDA kernel
      if (headI == 0 || attentionIndependentHeads_) {
        mappedState = dot(recState, Was_[headI]);
        if(layerNorm_)
          if(nematusNorm_)
            mappedState = layerNorm(mappedState, W_comb_att_lnss_[headI], W_comb_att_lnbs_[headI], NEMATUS_LN_EPS);
          else
            mappedState = layerNorm(mappedState, gammaStates_[headI]);
      }

      Expr e;
      if (!attentionBilinearLookup_) {
        auto attReduce = attOps(vas_[headI], mappedContexts_[filteredHeadI(headI)], mappedState);
        // @TODO: horrible ->
        e = reshape(transpose(softmax(transpose(attReduce), softmaxMask_)),
                    {dimBeam, srcWords, dimBatch, 1});
        // <- horrible
      }
      else {
        auto reshaped_state = reshape(mappedState, {1, dimBatch, attentionLookupDim_, dimBeam});

        auto time_transposed_mapped_context = time_transposed_mapped_contexts_[headI];
        auto bilinear_score = bdot(time_transposed_mapped_context, reshaped_state, false, false, (1.0 / std::sqrt(attentionLookupDim_)));
        e = reshape(transpose(softmax(transpose(bilinear_score, {3, 0, 1, 2}), softmaxMask_)),
                    {dimBeam, srcWords, dimBatch, 1});
      }

      if (attentionDropout_ > 0.0f) {
        e = dropout(e, attentionDropout_); // Transformer-style, non-Bayesian dropout on the attention weights
      }
      auto alignedSource = scalar_product(encState_->getAttended(), e, /*axis =*/ -3);

      // Attended context projection
      if (attentionProjectionDim_ != -1) {
        alignedSource = dot(alignedSource, attentionProjectionMatrices_[headI]);
        if (attentionProjectionLayerNorm_) {
          alignedSource = layerNorm(alignedSource, attentionProjectionMatrixGammas_[headI], attentionProjectionMatrixBs_[headI]);
        }
        if (attentionProjectionActivation_ != "identity") {
          if (attentionProjectionLayerNorm_) {
            alignedSource = attentionProjectionActivationFcn_(alignedSource);
          }
          else {
            alignedSource = attentionProjectionActivationFcn_(alignedSource + attentionProjectionMatrixBs_[headI]);
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

    auto concatenatedAlignedSources = concatenate(alignedSources, /*axis=*/ -1);
    contexts_.push_back(concatenatedAlignedSources);
    alignments_.push_back(first_e);
    return concatenatedAlignedSources;
  }

  std::vector<Expr>& getContexts() { return contexts_; }

  Expr getContext() { return concatenate(contexts_, /*axis =*/ -3); }

  std::vector<Expr>& getAlignments() { return alignments_; }

  virtual void clear() override {
    contexts_.clear();
    alignments_.clear();
  }

  int dimOutput() override {
    int alignedSourcesDimPerHead = (attentionProjectionDim_ == -1)? (encState_->getContext()->shape()[-1]): attentionProjectionDim_;
    return alignedSourcesDimPerHead * numAttentionHeads_;
  }
};

using Attention = GlobalAttention;
}  // namespace rnn
}  // namespace marian
