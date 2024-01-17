#pragma once

#include "layers_new/transformer.h"

#include "models/encoder.h"
#include "layers/constructors.h"

namespace marian {
namespace models {

class CometEncoder final : public nn::TransformerEncoder {
private:
  // This seems to be a mix of LayerNorm and BatchNorm and present in the original Unbabel code.
  // It norms over time, not batch, also should be optimized. Seems safe to disable for custom
  // models trained by us, but required when doing inference with Unbabel models.
  Expr cometNorm(Expr x, Expr binaryMask) const {
    Expr output;

    if(opt<bool>("comet-mix-norm", false)) {
      registerParameterLazy(gamma, Shape({ 1 }), inits::ones());
      int dimModel = x->shape()[-1];

      // Convert type to fp32 for better accumulation. This is a no-op if things are already fp32.
      Type origType = x->value_type();
      x             = marian::cast(x,       Type::float32);
      binaryMask    = marian::cast(binaryMask, Type::float32);

      x = x * binaryMask;
      auto denom = (float)dimModel * sum(binaryMask, -2);
      auto mu    = sum(sum(x, -1), -2) / denom; // sum over model and time
      auto sigma = sum(sum(square(x - mu), -1), -2) / denom;

      auto normed = (x - mu) / sqrt(sigma + 1e-12f);
      output = marian::cast(gamma, Type::float32) * sum(normed * binaryMask, -2) / sum(binaryMask, -2);

      // Undo conversion to fp32 if not originally fp32 (most likely fp16 then)
      output = marian::cast(output, origType);
    } else if(opt<bool>("comet-mix", false)) {
      // average over time dimension
      registerParameterLazy(gamma, Shape({ 1 }), inits::ones());
      output = gamma * sum(x * binaryMask, -2) / sum(binaryMask, -2);
    } else {
      output = sum(x * binaryMask, -2) / sum(binaryMask, -2);
    }

    return output;
  };

public:
  Expr weights;
  Expr gamma;

  CometEncoder(Ptr<ExpressionGraph> graph,
               Ptr<Options> options)
    : TransformerEncoder(graph, options) {}

  Expr apply(Expr input, Expr mask) const override {
    auto output = marian::nn::swapTimeBatch(input); // [beam depth=1, batch size, max length, vector dim]

    auto binaryMask = marian::nn::swapTimeBatch(mask);   // [beam depth=1, batch size, max length, vector dim=1]

    // apply positional embeddings to contextual input
    output = positionEmbedding->apply(output);

    // apply dropout or layer-norm to embeddings if required
    output = preprocessor->apply(output);
    auto logMask = maskProcessor->apply(output, binaryMask); // [beam depth=1, batch size * numHeads, max length, vector dim=1]

    std::vector<Expr> pooler;
    if(opt<bool>("comet-mix", false))
      pooler.push_back(cometNorm(output, binaryMask));

    // traverse the layers, use the same mask for each
    for(auto layer : *layers) {
      output = layer->apply(output, logMask);
      if(opt<bool>("comet-mix", false))
        pooler.push_back(cometNorm(output, binaryMask)); // [ batch, time, modelDim ]
    }

    if(opt<bool>("comet-mix", false)) {
      registerParameterLazy(weights, Shape({ opt<int>("enc-depth") + 1 }), inits::zeros());
      // comet22 has a sparsemax here
      auto normFn = opt<std::string>("comet-mix-transformation", "softmax");
      auto weightsNorm = (normFn == "sparsemax") ? sparsemax(weights) : softmax(weights);
      weightsNorm = reshape(weightsNorm, {weights->shape()[-1], 1});
      output = sum(weightsNorm * concatenate(pooler, /*axis=*/-2), -2); // [batch, 1, modelDim]
    } else {
      // just use last layer, average over time dim
      output = cometNorm(output, binaryMask); // [batch, 1, modelDim]
    }

    return output;
  }
};

// Wrapper for backwards compatibility that uses current encoder/decoder framework
struct CometBatchEncoder final : public nn::LayerWithOptions,
                                 public nn::IEmbeddingLayer,  // TransformerBatchEncoder is an IEmbeddingLayer that produces contextual embeddings
                                 public EncoderBase {         // @TODO: should all encoders be IEmbeddingLayer?
  Ptr<CometEncoder> encoder;

  CometBatchEncoder(Ptr<ExpressionGraph> graph,
                    Ptr<Options> options)
    : LayerWithOptions(graph, options),
      EncoderBase(graph, options)
  {
    encoder = New<CometEncoder>(graph, options);
    registerLayer(encoder);
  }

  // @TODO: subBatch should be of type Expr
  virtual std::tuple<Expr/*embeddings*/, Expr/*mask*/> apply(Ptr<data::SubBatch> subBatch) const override {
    // @TODO: this is still using the bad old interface
    auto embeddingLayer = getEmbeddingLayer(EncoderBase::opt<bool>("ulr", false));
    const auto& [batchEmbedding, batchMask] = embeddingLayer->apply(subBatch);

    auto batchContext = encoder->apply(batchEmbedding, batchMask); // [-4: beam depth=1, -3: batch size, -2: max length, -1: vector dim]
    return std::make_tuple(batchContext, batchMask);
  }

  virtual Expr apply(const Words& words, const Shape& shape) const override final {
    return applyIndices(toWordIndexVector(words), shape);
  }

  // alternative from indices directly
  virtual Expr applyIndices(const std::vector<WordIndex>& wordIndices, const Shape& shape) const override final {
    auto embeddingLayer = getEmbeddingLayer(EncoderBase::opt<bool>("ulr", false));
    Expr batchEmbedding = embeddingLayer->applyIndices(wordIndices, shape);
    auto batchContext = encoder->apply(batchEmbedding, /*mask=*/nullptr); // [-4: beam depth=1, -3: batch size, -2: max length, -1: vector dim]
    return batchContext;
  }

  // @TODO: currently here for backwards compat, should be replaced with apply()
  virtual Ptr<EncoderState> build(Ptr<ExpressionGraph> graph,
                                  Ptr<data::CorpusBatch> batch) override {
#if 1
    // @TODO: this should be removed, currently hack to init graph. Should happen in graph groups and constructors
    EncoderBase::graph_ = graph;
    setGraph(graph);
    // This makes sure that the graph passed into the model during construction and now evaluation are identical.
    // A good check to have for catching weird situations early.
    ABORT_IF(this->graph() != graph, "Graph used for construction and graph parameter do not match");
#endif

    const auto& [batchEmbedding, batchMask] = apply((*batch)[batchIndex_]);
    return New<EncoderState>(batchEmbedding, batchMask, batch);
  }

  virtual void clear() override {
    Layer::clear();
  }
};

// Dummpy pooler that only returns the encoder context
class CometEmbeddingPooler final : public nn::LayerWithOptions,
                                   public PoolerBase {
public:
  CometEmbeddingPooler(Ptr<ExpressionGraph> graph, Ptr<Options> options)
  : LayerWithOptions(graph, options),
    PoolerBase(graph, options) {}

  std::vector<Expr> apply(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch, const std::vector<Ptr<EncoderState>>& encoderStates) override {
    auto usage = (models::usage)LayerWithOptions::opt<int>("usage");
    ABORT_IF(usage != models::usage::embedding, "This pooler should only be used for generating embeddings??");
    ABORT_IF(encoderStates.size() != 1, "Size of encoderStates {} != 1", encoderStates.size());

    return { encoderStates[0]->getContext() };
  }

  void clear() override {}
};

// Actual COMET-like pooler, works for COMET-QE and COMET models (prior to WMT22)
class CometMetricPooler final : public nn::LayerWithOptions,
                                public PoolerBase {
private:
  Ptr<nn::Sequential> layers;
  std::mt19937 rng{(uint32_t)Config::seed};

public:
  CometMetricPooler(Ptr<ExpressionGraph> graph, Ptr<Options> options)
  : LayerWithOptions(graph, options),
    PoolerBase(graph, options) {

    float dropoutProb = LayerWithOptions::opt<float>("comet-dropout", 0.1f);
    auto ffnHidden = LayerWithOptions::opt<std::vector<int>>("comet-pooler-ffn", {2048, 1024});

    layers = New<nn::Sequential>(graph);
    for(auto dim : ffnHidden) {
      layers->append(New<nn::Linear>(graph, dim));
      layers->append(New<nn::Tanh>(graph));
      layers->append(New<nn::Dropout>(graph, dropoutProb));
    }
    layers->append(New<nn::Linear>(graph, 1));

    if(LayerWithOptions::opt<bool>("comet-final-sigmoid"))
      layers->append(New<nn::Sigmoid>(graph));

    registerLayer(layers);
  }

  std::vector<Expr> apply(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch, const std::vector<Ptr<EncoderState>>& encoderStates) override {
#if 1
    // @TODO: this should be removed, currently hack to init graph. Should happen in graph groups and constructors
    PoolerBase::graph_ = graph;
    setGraph(graph);
    // This makes sure that the graph passed into the model during construction and now evaluation are identical.
    // A good check to have for catching weird situations early.
    ABORT_IF(this->graph() != graph, "Graph used for construction and graph parameter do not match");
#endif

    auto beta = [](float alpha, std::mt19937& gen) {
      // Generate random numbers x and y from gamma distributions with the given alpha and beta parameters
      std::gamma_distribution<float> gamma(alpha, 1.f);
      float x = gamma(gen);
      float y = gamma(gen);
      return x / (x + y);
    };

    auto mixup = [&](Expr x, Expr y, float alpha, bool reg=true) -> Expr2 {
      if(alpha == 0.f)
        return {x, y};

      int dimBatch = x->shape()[-3];
      Type xType = x->value_type();

      std::vector<IndexType> indices(dimBatch);
      std::iota(indices.begin(), indices.end(), 0);

      // permute the indices and select batch entries accordingly
      std::shuffle(indices.begin(), indices.end(), rng);
      auto xPrime = index_select(x, -3, indices);
      auto yPrime = index_select(y, -3, indices);

      std::vector<float> lambdasVec(dimBatch);
      std::generate(lambdasVec.begin(), lambdasVec.end(), [&]{ return beta(alpha, rng); });
      auto lambdas = graph->constant({dimBatch, 1, 1}, inits::fromVector(lambdasVec), Type::float32);

      auto xMixup = (1.f - marian::cast(lambdas, xType)) * x + marian::cast(lambdas, xType) * xPrime;
      auto yMixup = (1.f - lambdas) * y + lambdas * yPrime;

      if(reg) {
        // return original and mixed samples
        xMixup = concatenate({x, xMixup}, /*axis=*/-2);
        yMixup = concatenate({y, yMixup}, /*axis=*/-2);
      }

      return {xMixup, yMixup};
    };

    // add bad examples to the batch
    auto addBad = [&](Expr src, Expr mt, int& dimBad) -> Expr2 {
      int dimBatch = src->shape()[-3];
      float badRatio = LayerWithOptions::opt<float>("comet-augment-bad", 0.f);
      dimBad = (int)std::ceil(dimBatch * badRatio); // use ceiling to make sure it's at least 1

      if(dimBad > 0) {
        LOG_ONCE(info, "Adding {:.1f} percent of bad examples to batch with label 0.0f", badRatio * 100);

        // select dimBad random examples from the batch and add them to the end
        std::vector<IndexType> indicesSrc(dimBatch), indicesMt(dimBatch);
        std::iota(indicesSrc.begin(), indicesSrc.end(), 0);

        // permute the indices and select batch entries accordingly
        std::shuffle(indicesSrc.begin(), indicesSrc.end(), rng);
        indicesSrc.resize(dimBad); // shrink to size
        auto srcSub = index_select(src, -3, indicesSrc);
        src = concatenate({src, srcSub}, /*axis=*/-3);

        std::iota(indicesMt.begin(), indicesMt.end(), 0);
        // permute the indices and select batch entries accordingly
        std::shuffle(indicesMt.begin(), indicesMt.end(), rng);
        indicesMt.resize(dimBad); // shrink to size
        auto mtSub = index_select(mt, -3, indicesMt);
        mt  = concatenate({mt, mtSub}, /*axis=*/-3);
      }

      return {src, mt};
    };

    auto usage = (models::usage)LayerWithOptions::opt<int>("usage");
    ABORT_IF(usage == models::usage::embedding, "Wrong pooler for embedding??");

    auto modelType = LayerWithOptions::opt<std::string>("type");
    ABORT_IF(modelType == "comet-qe" && encoderStates.size() != 2, "Pooler expects exactly two encoder states for comet-qe");
    ABORT_IF(modelType == "comet"    && encoderStates.size() != 3, "Pooler expects exactly three encoder states for comet");

    if(modelType == "comet-qe") {
      auto src = encoderStates[0]->getContext();
      auto mt  = encoderStates[1]->getContext();

      int dimBad = 0; // number of bad examples added to the batch
      if(getMode() == Mode::train) {
        // do not propagate gradients through the encoder if requested
        if(LayerWithOptions::opt<bool>("comet-stop-grad", false)) {
          src = stopGradient(src);
          mt  = stopGradient(mt);
        }

        // add bad examples to the batch
        // dimBad is the number of bad examples added to the batch and gets modified here if used
        auto srcMt = addBad(src, mt, /*out=*/dimBad);
        src = get<0>(srcMt);
        mt  = get<1>(srcMt);
      }

      auto diff = abs(mt - src);
      auto prod = mt * src;

      Expr output;
      if(usage == models::usage::evaluating) {
        auto embFwd  = concatenate({mt, src, prod, diff}, /*axis=*/-1); // [batch, 1, model]
        auto embBwd  = concatenate({src, mt, prod, diff}, /*axis=*/-1); // [batch, 1, model]
        auto emb     = concatenate({embFwd, embBwd}, /*axis=*/-2);

        output = layers->apply(emb);

        int dimBatch = output->shape()[-3];
        output = reshape(output, {dimBatch, 1, 2});
        return { output };
      } else {
        auto emb = concatenate({mt, src, prod, diff}, /*axis=*/-1); // [batch, 1, model]

        auto softLabelsWords = batch->front()->data();
        auto classVocab      = batch->front()->vocab();

        // we add bad examples to the batch, so we need to make sure the soft labels are padded accordingly with 0s
        int dimBatch = (int)softLabelsWords.size() + dimBad;
        std::vector<float> softLabels;
        for(auto w : softLabelsWords) {
          // @TODO: this is a super-ugly hack to get regression values
          float score = w != Word::NONE ? std::stof((*classVocab)[w]) : 0.f;
          softLabels.push_back(score);
        }
        // pad with 0s
        softLabels.resize(dimBatch, 0.f);

        auto labels = graph->constant({dimBatch, 1, 1}, inits::fromVector(softLabels), Type::float32);

        if(getMode() == Mode::train) {
          float mixupAlpha = LayerWithOptions::opt<float>("comet-mixup", 0.f);
          bool mixupReg    = LayerWithOptions::opt<bool>("comet-mixup-reg", false);
          auto xy = mixup(emb, labels, mixupAlpha, mixupReg);
          emb     = get<0>(xy);
          labels  = get<1>(xy);
        }

        output = marian::cast(layers->apply(emb), Type::float32);
        return { output, labels };
      }
    } else if(modelType == "comet") {
      auto src = encoderStates[0]->getContext();
      auto mt  = encoderStates[1]->getContext();
      auto ref = encoderStates[2]->getContext();

      auto diffRef = abs(mt - ref);
      auto prodRef = mt * ref;

      auto diffSrc = abs(mt - src);
      auto prodSrc = mt * src;

      Expr output;
      if(usage == models::usage::evaluating) {
        auto emb  = concatenate({mt, ref, prodRef, diffRef, prodSrc, diffSrc}, /*axis=*/-1); // [batch, 1, model]
        output = layers->apply(emb);
        int dimBatch = output->shape()[-3];
        output = reshape(output, {dimBatch, 1, 1});
        return { output };
      } else {
        // Currently no training for COMET with reference @TODO: add training
        ABORT("Usage other than 'evaluating' not implemented");
      }
    } else {
      ABORT("Unknown model type {}", modelType);
    }
  }

  void clear() override {}
};

// Wraps an EncoderClassifier so it can produce a cost from raw logits. @TODO: Needs refactoring
class CometLoss final : public ICost {
protected:
  Ptr<Options> options_;
  const bool inference_{false};
  const bool rescore_{false};

public:
  CometLoss(Ptr<Options> options)
    : options_(options), inference_(options->get<bool>("inference", false)),
      rescore_(options->get<std::string>("cost-type", "ce-sum") == "ce-rescore") { }

  Ptr<MultiRationalLoss> apply(Ptr<IModel> model,
                               Ptr<ExpressionGraph> graph,
                               Ptr<data::Batch> batch,
                               bool clearGraph = true) override {
    auto encpool = std::static_pointer_cast<EncoderPooler>(model);
    auto corpusBatch = std::static_pointer_cast<data::CorpusBatch>(batch);

    auto inputTypes = options_->get<std::vector<std::string>>("input-types", {});
    ABORT_IF(inputTypes != std::vector<std::string>({"class", "sequence", "sequence"}),
             "Expected input-types to be have fields (class, sequence, sequence)");
    ABORT_IF(corpusBatch->sets() != 3, "Expected 3 sub-batches, not {}", corpusBatch->sets());

    std::function<Expr(Expr, Expr)> lossFn;
    auto lossType = options_->get<std::string>("cost-type", "ce-sum");

    if(lossType == "ce-sum" || lossType == "ce-mean") {
      lossFn = [&](Expr x, Expr y) {
        float eps = 1e-5f;
        if(!options_->get<bool>("comet-final-sigmoid"))
          x = sigmoid(x);
        return -(y * log(x + eps) + (1.f - y) * log((1.f + eps) - x));
      };
    } else if(lossType == "mse") {
      lossFn = [&](Expr x, Expr y) {
        return square(x - y);
      };
    } else if(lossType == "mae") {
      lossFn = [&](Expr x, Expr y) {
        return abs(x - y);
      };
    } else {
      ABORT("Unknown loss type {} for COMET training", lossType);
    }

    auto encoded = encpool->apply(graph, corpusBatch, clearGraph);

    Expr x = encoded[0];
    Expr y = encoded[1];
    auto loss = lossFn(x, y);

    loss = mean(loss, /*axis=*/-2); // this should only do something with mixup regularization

    int dimBatch = loss->shape()[-3];
    if(rescore_)
      loss = reshape(loss, {1, dimBatch, 1});
    else
      loss = sum(loss, /*axis=*/-3); // [1, 1, 1]

    Ptr<MultiRationalLoss> multiLoss = New<SumMultiRationalLoss>();
    RationalLoss lossPiece(loss, (float)dimBatch);
    multiLoss->push_back(lossPiece);

    return multiLoss;
  }
};

} // namespace models
} // namespace marian

