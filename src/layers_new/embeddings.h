#pragma once

#include "layers_new/interface.h"
#include "data/corpus_base.h"
#include "data/factored_vocab.h"

namespace marian {
namespace nn {

// Embedding from corpus sub-batch to (emb, mask)
struct IEmbeddingLayer {
  virtual std::tuple<Expr/*input*/, Expr/*mask*/> apply(Ptr<data::SubBatch> subBatch) const = 0;

  virtual Expr apply(const Words& embIdx, const Shape& shape) const = 0;

  // alternative from indices directly
  virtual Expr applyIndices(const std::vector<WordIndex>& embIdx, const Shape& shape) const = 0;
};

struct IPositionEmbeddingLayer {
  virtual Expr apply(Expr, int startPosition = 0) = 0;
};

// A regular embedding layer.
// Note that this also applies dropout if the option is passed (pass 0 when in inference mode).
// It is best to not use Embedding directly, but rather via getEmbeddingLayer() in
// EncoderDecoderLayerBase, which knows to pass on all required parameters from options.
class Embedding : public LayerWithOptions, public IEmbeddingLayer {
public:
  Expr embeddings;

  Embedding(Ptr<ExpressionGraph> graph, Ptr<Options> options) : LayerWithOptions(graph, options) {
    std::string name = opt<std::string>("prefix");
    int dimVoc = opt<int>("dimVocab");
    int dimEmb = opt<int>("dimEmb");
    bool fixed = opt<bool>("fixed", false);

    factoredVocab_ = FactoredVocab::tryCreateAndLoad(options_->get<std::string>("vocab", ""));
    if (factoredVocab_) {
      dimVoc = (int)factoredVocab_->factorVocabSize();
      LOG_ONCE(info, "[embedding] Factored embeddings enabled");
    }

    // Embedding layer initialization should depend only on embedding size, hence fanIn=false
    auto initFunc = inits::glorotUniform(/*fanIn=*/false, /*fanOut=*/true); // -> embedding vectors have roughly unit length
    
    if(options_->has("embFile")) {
      std::string file = opt<std::string>("embFile");
      if (!file.empty()) {
        bool norm = opt<bool>("normalization", false);
        initFunc = inits::fromWord2vec(file, dimVoc, dimEmb, norm);
      }
    }

    registerParameter(embeddings, Shape({dimVoc, dimEmb}), initFunc);
    embeddings->setTrainable(!fixed); // @TODO: move into registerParam macro
  }

  virtual ~Embedding() = default;
  
  std::tuple<Expr/*embeddings*/, Expr/*mask*/> apply(Ptr<data::SubBatch> subBatch) const override final {
    auto graph = embeddings->graph();
    int dimBatch = (int)subBatch->batchSize();
    int dimEmb = embeddings->shape()[-1];
    int dimWidth = (int)subBatch->batchWidth();

    // factored embeddings:
    //  - regular:
    //     - y = x @ E    x:[B x 1ofV] ; E:[V x D] ; y:[B x D]
    //  - factored:
    //     - u = x @ M    one-hot to U-dimensional multi-hot (all factors in one concatenated space)
    //        - each row of M contains the set of factors for one word => we want a CSR matrix
    //     - y = (x @ M) @ E   (x:[B x 1ofV] ; M:[V x U]) ; E:[U x D] ; y:[B x D]
    //  - first compute x @ M on the CPU
    //     - (Uvalues, Uindices, Uoffsets) = csr_rows(Mvalues, Mindices, Moffsets, subBatch->data()):
    //        - shape (U, specifically) not actually needed here
    //     - foreach input x[i]
    //        - locate row M[i,*]
    //        - copy through its index values (std::vector<push_back>)
    //     - create a matching ones vector (we can keep growing)
    //     - convert to GPU-side CSR matrix. CSR matrix now has #rows equal to len(x)
    //     - CSR matrix product with E
    //     - csr_dot(Uvalues, Uindices, Uoffsets, embeddings, transposeU)
    //        - double-check if all dimensions are specified. Probably not for transpose (which would be like csc_dot()).
    //  - weighting:
    //     - core factors' gradients are sums over all words that use the factors;
    //        - core factors' embeddings move very fast
    //        - words will need to make up for the move; rare words cannot
    //     - so, we multiply each factor with 1/refCount
    //        - core factors get weighed down a lot
    //        - no impact on gradients, as Adam makes up for it; embeddings still move fast just as before
    //        - but forward pass weighs them down, so that all factors are in a similar numeric range
    //        - if it is required to be in a different range, the embeddings can still learn that, but more slowly

    auto batchEmbeddings = apply(subBatch->data(), {dimWidth, dimBatch, dimEmb});
    auto batchMask = graph->constant({dimWidth, dimBatch, 1},
                                     inits::fromVector(subBatch->mask()));
    return std::make_tuple(batchEmbeddings, batchMask);
  }

  Expr apply(const Words& words, const Shape& shape) const override final {
    if (factoredVocab_) {
      Expr selectedEmbs = multiRows(words, opt<float>("dropout", 0.0f));        // [(B*W) x E]
      selectedEmbs = reshape(selectedEmbs, shape); // [W, B, E]
      return selectedEmbs;
    }
    else
      return applyIndices(toWordIndexVector(words), shape);
  }

  Expr applyIndices(const std::vector<WordIndex>& embIdx, const Shape& shape) const override final {
    ABORT_IF(factoredVocab_, "Embedding: applyIndices must not be used with a factored vocabulary");
    auto selectedEmbs = rows(embeddings, embIdx);        // [(B*W) x E]
    selectedEmbs = reshape(selectedEmbs, shape); // [W, B, E]
    // @BUGBUG: We should not broadcast along dimBatch=[-2]. Then we can also dropout before reshape() (test that separately)
    selectedEmbs = dropout(selectedEmbs, opt<float>("dropout", 0.0f), { selectedEmbs->shape()[-3], 1, 1 });
    return selectedEmbs;
  }

private:
  Ptr<FactoredVocab> factoredVocab_;

  // helper to embed a sequence of words (given as indices) via factored embeddings
  Expr multiRows(const Words& data, float dropProb) const {
    auto graph = embeddings->graph();
    auto factoredData = factoredVocab_->csr_rows(data);
    // multi-hot factor vectors are represented as a sparse CSR matrix
    // [row index = word position index] -> set of factor indices for word at this position
    ABORT_IF(factoredData.shape != Shape({(int)factoredData.offsets.size()-1/*=rows of CSR*/, embeddings->shape()[0]}), "shape mismatch??");
    // the CSR matrix is passed in pieces
    auto weights = graph->constant({ (int)factoredData.weights.size() }, inits::fromVector(factoredData.weights), Type::float32);
    auto indices = graph->constant({ (int)factoredData.indices.size() }, inits::fromVector(factoredData.indices), Type::uint32);
    auto offsets = graph->constant({ (int)factoredData.offsets.size() }, inits::fromVector(factoredData.offsets), Type::uint32);
    // apply dropout
    // We apply it to the weights, i.e. factors get dropped out separately, but always as entire vectors.
    weights = dropout(weights, dropProb);
    // perform the product
    return csr_dot(factoredData.shape, weights, indices, offsets, embeddings);
  }
};

// Abstract base class for position embedding layers
struct PositionEmbeddingLayer : public Layer, 
                                public IPositionEmbeddingLayer {
  using Layer::namedLayers_;
  using Layer::namedParameters_;
  using Layer::param;

  int positionAxis;
  int maxLength;

  PositionEmbeddingLayer(Ptr<ExpressionGraph> graph, int positionAxis, int maxLength) 
  : Layer(graph), positionAxis(positionAxis), maxLength(maxLength) {}

  virtual ~PositionEmbeddingLayer() = default;
};

struct SinusoidalPositionEmbedding : public PositionEmbeddingLayer {
  using PositionEmbeddingLayer::positionAxis;
  using PositionEmbeddingLayer::maxLength;

  SinusoidalPositionEmbedding(Ptr<ExpressionGraph> graph, int positionAxis)
   : PositionEmbeddingLayer(graph, positionAxis, /*maxLength=*/-1)
  {}

  virtual ~SinusoidalPositionEmbedding() = default;

  Expr apply(Expr input, int start = 0) override {      
      int dimEmb   = input->shape()[-1];
      int dimWords = input->shape()[positionAxis];

      input = std::sqrt((float)dimEmb) * input; // input were initialized to unit length; so norms will be in order of sqrt(dimEmb)

      Shape posEmbeddingShape;
      posEmbeddingShape.resize(input->shape().size()); // resize to input shape size and fill with 1s
      posEmbeddingShape.set(-1, dimEmb);               // match embedding size
      posEmbeddingShape.set(positionAxis, dimWords);   // match number of items to embed on correct axis

      // the node initializer is dimension agnostic for dimensions other than the last 
      // dimension (embedding dimension) and works with any positionAxis value
      auto posEmbeddings = graph()->constant(posEmbeddingShape,
                                             inits::sinusoidalPositionEmbeddings(start));

      input = input + posEmbeddings;
      return input;
  }
};

struct LearnedPositionEmbedding : public PositionEmbeddingLayer {
  using PositionEmbeddingLayer::positionAxis;
  using PositionEmbeddingLayer::maxLength;

  Expr embeddings;

  LearnedPositionEmbedding(Ptr<ExpressionGraph> graph, int positionAxis, int maxLength)
   : PositionEmbeddingLayer(graph, positionAxis, maxLength)
  {}

  virtual ~LearnedPositionEmbedding() = default;

  Expr apply(Expr input, int start = 0) override {     
      int dimEmb   = input->shape()[-1];
      int dimWords = input->shape()[positionAxis];

      registerParameter(embeddings, 
                        Shape({maxLength, dimEmb}), 
                        inits::glorotUniform(/*fanIn=*/false, /*fanOut=*/true));

      ABORT_IF(start + dimWords > maxLength, 
               "Number of positions ({}) starting at position {} exceeds maximum length {}",
               dimWords, start, maxLength);

      Shape posEmbeddingShape;
      posEmbeddingShape.resize(input->shape().size()); // resize to input shape size and fill with 1s
      posEmbeddingShape.set(-1, dimEmb);               // match embedding size
      posEmbeddingShape.set(positionAxis, dimWords);   // match number of items to embed on correct axis

      auto posEmbeddings = slice(embeddings, -2, Slice(start, start + dimWords));
      posEmbeddings      = reshape(posEmbeddings, posEmbeddingShape);

      input = input + posEmbeddings;
      return input;
  }
};

static Ptr<PositionEmbeddingLayer> positionEmbeddingFromOptions(Ptr<ExpressionGraph> graph, 
                                                                Ptr<Options> options, 
                                                                int positionAxis) {
  bool trainedEmbedding = options->get<bool>("transformer-train-position-embeddings", false);
  if(trainedEmbedding) {
    int maxLength = options->get<int>("max-length");
    return New<LearnedPositionEmbedding>(graph, positionAxis, maxLength);
  } else {
    return New<SinusoidalPositionEmbedding>(graph, positionAxis);
  }
}

} // namespace nn
} // namespace marian
