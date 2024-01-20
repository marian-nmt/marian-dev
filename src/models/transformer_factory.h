// @TODO: rename to transformer.h eventually. This is not a Factory as in factory.h.
#pragma once

#include "marian.h"

#include "layers_new/neuralnet.h"
#include "models/decoder.h"
#include "models/encoder.h"
#include "models/encoder_decoder.h"

namespace marian {
Ptr<EncoderBase> NewEncoderTransformer(Ptr<ExpressionGraph> graph, Ptr<Options> options);
Ptr<DecoderBase> NewDecoderTransformer(Ptr<ExpressionGraph> graph, Ptr<Options> options);

class TransformerLegacy : public EncoderDecoder {
public:
  TransformerLegacy(Ptr<ExpressionGraph> graph, Ptr<Options> options)
   : EncoderDecoder(graph, options), nameMap_(createNameMap()) { }

  void load(Ptr<ExpressionGraph> graph,
            Ptr<io::ModelWeights> modelFile,
            bool markedReloaded = true) override {

    for(auto& item : modelFile->items()) {
      auto pair = nameMap_.find(item.name);
      if(pair != nameMap_.end()) {
        LOG(debug, "Mapping parameter {} to {}", item.name, pair->second);
        const_cast<io::Item&>(item).name = pair->second;

        // reduce shape of bias vectors from {1, dimModel} to {dimModel}
        int dimModel = item.shape[-1];
        if(item.shape == Shape({1, dimModel}))
          const_cast<io::Item&>(item).shape = Shape({dimModel});
      } else {
        LOG(debug, "Could not find parameter {}", item.name);
      }
    }

    // in the new model, linear layers are transposed; we undo that here.
    // @TODO: alternatively, we can transpose the item data
    auto encoder = std::dynamic_pointer_cast<nn::Layer>(encoders_[0]);
    ABORT_IF(!encoder, "Could not cast to new type of encoder??");
    for(auto& linear : encoder->allLayers<nn::Linear>())
      linear->transposed = false;

    auto decoder = std::dynamic_pointer_cast<nn::Layer>(decoders_[0]);
    ABORT_IF(!decoder, "Could not cast to new type of decoder??");
    for(auto& linear : decoder->allLayers<nn::Linear>())
      linear->transposed = false;

    // load items into the graph
    graph->load(modelFile);
  }

private:
  std::map<std::string, std::string> nameMap_;

  std::map<std::string, std::string> createNameMap() {
    std::map<std::string, std::string> nameMap = {
      {"Wemb", "Wemb"},
    };

    // @TODO: This is going to change
    std::string prefix = "TransformerBatchEncoder";

    std::string key, value;
    for(int layerNo = 0; layerNo < opt<int>("enc-depth"); ++layerNo) {
      // name maps for encoder self-attention blocks
      nameMap[fmt::format("encoder_l{}_self_Wq", layerNo + 1)] = fmt::format("{}->encoder->layers->at({})->as<marian::nn::TransformerEncoderLayer>()->selfAttentionBlock->selfAttention->qProj->weight", prefix, layerNo);
      nameMap[fmt::format("encoder_l{}_self_bq", layerNo + 1)] = fmt::format("{}->encoder->layers->at({})->as<marian::nn::TransformerEncoderLayer>()->selfAttentionBlock->selfAttention->qProj->bias", prefix, layerNo);

      nameMap[fmt::format("encoder_l{}_self_Wk", layerNo + 1)] = fmt::format("{}->encoder->layers->at({})->as<marian::nn::TransformerEncoderLayer>()->selfAttentionBlock->selfAttention->kProj->weight", prefix, layerNo);
      nameMap[fmt::format("encoder_l{}_self_bk", layerNo + 1)] = fmt::format("{}->encoder->layers->at({})->as<marian::nn::TransformerEncoderLayer>()->selfAttentionBlock->selfAttention->kProj->bias", prefix, layerNo);

      nameMap[fmt::format("encoder_l{}_self_Wv", layerNo + 1)] = fmt::format("{}->encoder->layers->at({})->as<marian::nn::TransformerEncoderLayer>()->selfAttentionBlock->selfAttention->vProj->weight", prefix, layerNo);
      nameMap[fmt::format("encoder_l{}_self_bv", layerNo + 1)] = fmt::format("{}->encoder->layers->at({})->as<marian::nn::TransformerEncoderLayer>()->selfAttentionBlock->selfAttention->vProj->bias", prefix, layerNo);

      nameMap[fmt::format("encoder_l{}_self_Wo", layerNo + 1)] = fmt::format("{}->encoder->layers->at({})->as<marian::nn::TransformerEncoderLayer>()->selfAttentionBlock->selfAttention->oProj->weight", prefix, layerNo);
      nameMap[fmt::format("encoder_l{}_self_bo", layerNo + 1)] = fmt::format("{}->encoder->layers->at({})->as<marian::nn::TransformerEncoderLayer>()->selfAttentionBlock->selfAttention->oProj->bias", prefix, layerNo);

      nameMap[fmt::format("encoder_l{}_self_Wo_ln_scale", layerNo + 1)] = fmt::format("{}->encoder->layers->at({})->as<marian::nn::TransformerEncoderLayer>()->selfAttentionBlock->postprocessor->norm->weight", prefix, layerNo);
      nameMap[fmt::format("encoder_l{}_self_Wo_ln_bias", layerNo + 1)]  = fmt::format("{}->encoder->layers->at({})->as<marian::nn::TransformerEncoderLayer>()->selfAttentionBlock->postprocessor->norm->bias", prefix, layerNo);

      // name maps for encoder FFN blocks
      int mult = 3;
      for(int ffnLayerNo = 0; ffnLayerNo < opt<int>("transformer-ffn-depth"); ++ffnLayerNo) {
        std::string layerType = "Linear";
        // multiplying with 3 since in new model activation and dropout are also layers that are always added
        if(opt<std::string>("transformer-ffn-activation") == "relu" && ffnLayerNo < opt<int>("transformer-ffn-depth") - 1) {
          mult = 1;
          layerType = "LinearReluDropout";
        }
        nameMap[fmt::format("encoder_l{}_ffn_W{}", layerNo + 1, ffnLayerNo + 1)] = fmt::format("{}->encoder->layers->at({})->as<marian::nn::TransformerEncoderLayer>()->filterBlock->layers->at({})->as<marian::nn::{}>()->weight", prefix, layerNo, mult * ffnLayerNo, layerType);
        nameMap[fmt::format("encoder_l{}_ffn_b{}", layerNo + 1, ffnLayerNo + 1)] = fmt::format("{}->encoder->layers->at({})->as<marian::nn::TransformerEncoderLayer>()->filterBlock->layers->at({})->as<marian::nn::{}>()->bias", prefix, layerNo, mult * ffnLayerNo, layerType);
      }
      nameMap[fmt::format("encoder_l{}_ffn_ffn_ln_scale", layerNo + 1)] = fmt::format("{}->encoder->layers->at({})->as<marian::nn::TransformerEncoderLayer>()->filterBlock->postprocessor->norm->weight", prefix, layerNo);
      nameMap[fmt::format("encoder_l{}_ffn_ffn_ln_bias", layerNo + 1)]  = fmt::format("{}->encoder->layers->at({})->as<marian::nn::TransformerEncoderLayer>()->filterBlock->postprocessor->norm->bias", prefix, layerNo);
    }

    prefix = "TransformerBatchDecoder";
    for(int layerNo = 0; layerNo < opt<int>("dec-depth"); ++layerNo) {
      // name maps for decoder self-attention blocks
      nameMap[fmt::format("decoder_l{}_self_Wq", layerNo + 1)] = fmt::format("{}->decoder->layers->at({})->as<marian::nn::TransformerDecoderLayer>()->selfAttentionBlock->selfAttention->qProj->weight", prefix, layerNo);
      nameMap[fmt::format("decoder_l{}_self_bq", layerNo + 1)] = fmt::format("{}->decoder->layers->at({})->as<marian::nn::TransformerDecoderLayer>()->selfAttentionBlock->selfAttention->qProj->bias", prefix, layerNo);

      nameMap[fmt::format("decoder_l{}_self_Wk", layerNo + 1)] = fmt::format("{}->decoder->layers->at({})->as<marian::nn::TransformerDecoderLayer>()->selfAttentionBlock->selfAttention->kProj->weight", prefix, layerNo);
      nameMap[fmt::format("decoder_l{}_self_bk", layerNo + 1)] = fmt::format("{}->decoder->layers->at({})->as<marian::nn::TransformerDecoderLayer>()->selfAttentionBlock->selfAttention->kProj->bias", prefix, layerNo);

      nameMap[fmt::format("decoder_l{}_self_Wv", layerNo + 1)] = fmt::format("{}->decoder->layers->at({})->as<marian::nn::TransformerDecoderLayer>()->selfAttentionBlock->selfAttention->vProj->weight", prefix, layerNo);
      nameMap[fmt::format("decoder_l{}_self_bv", layerNo + 1)] = fmt::format("{}->decoder->layers->at({})->as<marian::nn::TransformerDecoderLayer>()->selfAttentionBlock->selfAttention->vProj->bias", prefix, layerNo);

      nameMap[fmt::format("decoder_l{}_self_Wo", layerNo + 1)] = fmt::format("{}->decoder->layers->at({})->as<marian::nn::TransformerDecoderLayer>()->selfAttentionBlock->selfAttention->oProj->weight", prefix, layerNo);
      nameMap[fmt::format("decoder_l{}_self_bo", layerNo + 1)] = fmt::format("{}->decoder->layers->at({})->as<marian::nn::TransformerDecoderLayer>()->selfAttentionBlock->selfAttention->oProj->bias", prefix, layerNo);

      nameMap[fmt::format("decoder_l{}_self_Wo_ln_scale", layerNo + 1)] = fmt::format("{}->decoder->layers->at({})->as<marian::nn::TransformerDecoderLayer>()->selfAttentionBlock->postprocessor->norm->weight", prefix, layerNo);
      nameMap[fmt::format("decoder_l{}_self_Wo_ln_bias", layerNo + 1)]  = fmt::format("{}->decoder->layers->at({})->as<marian::nn::TransformerDecoderLayer>()->selfAttentionBlock->postprocessor->norm->bias", prefix, layerNo);

      // name maps for decoder SSRU
      nameMap[fmt::format("decoder_l{}_rnn_W", layerNo + 1)] = fmt::format("{}->decoder->layers->at({})->as<marian::nn::TransformerDecoderLayer>()->autoRegressiveBlock->rnn->cell->iProj->weight", prefix, layerNo);

      nameMap[fmt::format("decoder_l{}_rnn_Wf", layerNo + 1)] = fmt::format("{}->decoder->layers->at({})->as<marian::nn::TransformerDecoderLayer>()->autoRegressiveBlock->rnn->cell->fProj->weight", prefix, layerNo);
      nameMap[fmt::format("decoder_l{}_rnn_bf", layerNo + 1)] = fmt::format("{}->decoder->layers->at({})->as<marian::nn::TransformerDecoderLayer>()->autoRegressiveBlock->rnn->cell->fProj->bias", prefix, layerNo);

      nameMap[fmt::format("decoder_l{}_rnn_Wo", layerNo + 1)] = fmt::format("{}->decoder->layers->at({})->as<marian::nn::TransformerDecoderLayer>()->autoRegressiveBlock->rnn->oProj->weight", prefix, layerNo);
      nameMap[fmt::format("decoder_l{}_rnn_bo", layerNo + 1)] = fmt::format("{}->decoder->layers->at({})->as<marian::nn::TransformerDecoderLayer>()->autoRegressiveBlock->rnn->oProj->bias", prefix, layerNo);

      nameMap[fmt::format("decoder_l{}_rnn_ffn_ln_scale", layerNo + 1)] = fmt::format("{}->decoder->layers->at({})->as<marian::nn::TransformerDecoderLayer>()->autoRegressiveBlock->postprocessor->norm->weight", prefix, layerNo);
      nameMap[fmt::format("decoder_l{}_rnn_ffn_ln_bias", layerNo + 1)]  = fmt::format("{}->decoder->layers->at({})->as<marian::nn::TransformerDecoderLayer>()->autoRegressiveBlock->postprocessor->norm->bias", prefix, layerNo);

      // name maps for decoder cross-attention blocks
      nameMap[fmt::format("decoder_l{}_context_Wq", layerNo + 1)] = fmt::format("{}->decoder->layers->at({})->as<marian::nn::TransformerDecoderLayer>()->crossAttentionBlock->crossAttention->qProj->weight", prefix, layerNo);
      nameMap[fmt::format("decoder_l{}_context_bq", layerNo + 1)] = fmt::format("{}->decoder->layers->at({})->as<marian::nn::TransformerDecoderLayer>()->crossAttentionBlock->crossAttention->qProj->bias", prefix, layerNo);

      nameMap[fmt::format("decoder_l{}_context_Wk", layerNo + 1)] = fmt::format("{}->decoder->layers->at({})->as<marian::nn::TransformerDecoderLayer>()->crossAttentionBlock->crossAttention->kProj->weight", prefix, layerNo);
      nameMap[fmt::format("decoder_l{}_context_bk", layerNo + 1)] = fmt::format("{}->decoder->layers->at({})->as<marian::nn::TransformerDecoderLayer>()->crossAttentionBlock->crossAttention->kProj->bias", prefix, layerNo);

      nameMap[fmt::format("decoder_l{}_context_Wv", layerNo + 1)] = fmt::format("{}->decoder->layers->at({})->as<marian::nn::TransformerDecoderLayer>()->crossAttentionBlock->crossAttention->vProj->weight", prefix, layerNo);
      nameMap[fmt::format("decoder_l{}_context_bv", layerNo + 1)] = fmt::format("{}->decoder->layers->at({})->as<marian::nn::TransformerDecoderLayer>()->crossAttentionBlock->crossAttention->vProj->bias", prefix, layerNo);

      nameMap[fmt::format("decoder_l{}_context_Wo", layerNo + 1)] = fmt::format("{}->decoder->layers->at({})->as<marian::nn::TransformerDecoderLayer>()->crossAttentionBlock->crossAttention->oProj->weight", prefix, layerNo);
      nameMap[fmt::format("decoder_l{}_context_bo", layerNo + 1)] = fmt::format("{}->decoder->layers->at({})->as<marian::nn::TransformerDecoderLayer>()->crossAttentionBlock->crossAttention->oProj->bias", prefix, layerNo);

      nameMap[fmt::format("decoder_l{}_context_Wo_ln_scale", layerNo + 1)] = fmt::format("{}->decoder->layers->at({})->as<marian::nn::TransformerDecoderLayer>()->crossAttentionBlock->postprocessor->norm->weight", prefix, layerNo);
      nameMap[fmt::format("decoder_l{}_context_Wo_ln_bias", layerNo + 1)]  = fmt::format("{}->decoder->layers->at({})->as<marian::nn::TransformerDecoderLayer>()->crossAttentionBlock->postprocessor->norm->bias", prefix, layerNo);

      // name maps for decoder FFN blocks
      int mult = 3;
      for(int ffnLayerNo = 0; ffnLayerNo < opt<int>("transformer-ffn-depth"); ++ffnLayerNo) {
        std::string layerType = "Linear";
        // multiplying with 3 since in new model activation and dropout are also layers that are always added
        if(opt<std::string>("transformer-ffn-activation") == "relu" && ffnLayerNo < opt<int>("transformer-ffn-depth") - 1) {
          mult = 1;
          layerType = "LinearReluDropout";
        }
        nameMap[fmt::format("decoder_l{}_ffn_W{}", layerNo + 1, ffnLayerNo + 1)] = fmt::format("{}->decoder->layers->at({})->as<marian::nn::TransformerDecoderLayer>()->filterBlock->layers->at({})->as<marian::nn::{}>()->weight", prefix, layerNo, mult * ffnLayerNo, layerType);
        nameMap[fmt::format("decoder_l{}_ffn_b{}", layerNo + 1, ffnLayerNo + 1)] = fmt::format("{}->decoder->layers->at({})->as<marian::nn::TransformerDecoderLayer>()->filterBlock->layers->at({})->as<marian::nn::{}>()->bias", prefix, layerNo, mult * ffnLayerNo, layerType);
      }
      nameMap[fmt::format("decoder_l{}_ffn_ffn_ln_scale", layerNo + 1)] = fmt::format("{}->decoder->layers->at({})->as<marian::nn::TransformerDecoderLayer>()->filterBlock->postprocessor->norm->weight", prefix, layerNo);
      nameMap[fmt::format("decoder_l{}_ffn_ffn_ln_bias", layerNo + 1)]  = fmt::format("{}->decoder->layers->at({})->as<marian::nn::TransformerDecoderLayer>()->filterBlock->postprocessor->norm->bias", prefix, layerNo);
    }

    return nameMap;
  }
};

}  // namespace marian
