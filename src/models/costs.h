#pragma once

#include "layers/generic.h"
#include "layers/guided_alignment.h"
#include "layers/loss.h"
#include "layers/weight.h"
#include "models/encoder_decoder.h"

namespace marian {
namespace models {

class CostBase {
public:
  virtual Expr apply(Ptr<ModelBase> model,
                     Ptr<ExpressionGraph> graph,
                     Ptr<data::Batch> batch,
                     bool clearGraph = true)
      = 0;
};

class EncoderDecoderCE : public CostBase {
protected:
  Ptr<Options> options_;

  bool inference_{false};
  bool toBeWeighted_{false};
  Ptr<LossBase> loss_;
  Ptr<WeightingBase> weighter_;

public:
  EncoderDecoderCE(Ptr<Options> options)
      : options_(options), inference_(options->get<bool>("inference", false)) {
    loss_ = LossFactory(options_, inference_);

    toBeWeighted_
        = (options_->has("data-weighting") && !inference_)
          || (options_->has("dynamic-weighting")
              && options_->get<bool>("dynamic-weighting") && !inference_);
    if(toBeWeighted_)
      weighter_ = WeightingFactory(options_);
  }

  Expr apply(Ptr<ModelBase> model,
             Ptr<ExpressionGraph> graph,
             Ptr<data::Batch> batch,
             bool clearGraph = true) override {
    auto encdec = std::static_pointer_cast<EncoderDecoder>(model);
    auto corpusBatch = std::static_pointer_cast<data::CorpusBatch>(batch);

    auto state = encdec->stepAll(graph, corpusBatch, clearGraph);

    Expr weights;
    if(toBeWeighted_)
      weights = weighter_->getWeights(graph, corpusBatch);

    Expr cost;
    cost = loss_->getCost(state->getLogProbs(),
                          state->getTargetIndices(),
                          state->getTargetMask(),
                          weights);

    if(options_->get<std::string>("guided-alignment", "none") != "none" && !inference_) {
      auto alignments = encdec->getDecoders()[0]->getAlignments();
      ABORT_IF(alignments.empty(), "Model does not seem to support alignments");

      auto att = concatenate(alignments, /*axis =*/ -1);

      cost = cost + guidedAlignmentCost(graph, corpusBatch, options_, att);
    }
    
    return cost;
  }
};

class MultiAgentEncoderDecoderCE : public EncoderDecoderCE {
private:
  std::vector<std::vector<float>> scores_;

public:
  MultiAgentEncoderDecoderCE(Ptr<Options> options) 
  : EncoderDecoderCE(options) {

    auto paths = options_->get<std::vector<std::string>>("multi-agent-learning");
    ABORT_IF(paths.empty(), "Multi-agent cost with empty multi-agent scores list");

    for(const auto& path : paths) {
      LOG(info, "Loading multi-agent scores from {}", path);
      ABORT_IF(!filesystem::exists(path), "Multi-agent scores list {} does not exist", path);
      io::InputFileStream in(path);
      std::vector<float> maScores;
      float score;
      while(in >> score)
        maScores.push_back(score);
      
      if(!scores_.empty()) {
        ABORT_IF(maScores.size() != scores_.back().size(), 
                 "Sets of scores have different lengths {} != {}", 
                 maScores.size(), scores_.back().size());
      }
      LOG(info, "Loaded {} scores", maScores.size());
      scores_.push_back(maScores);
    }

  }

  Expr apply(Ptr<ModelBase> model,
             Ptr<ExpressionGraph> graph,
             Ptr<data::Batch> batch,
             bool clearGraph = true) override {

    auto encdec = std::static_pointer_cast<EncoderDecoder>(model);
    auto corpusBatch = std::static_pointer_cast<data::CorpusBatch>(batch);

    auto state = encdec->stepAll(graph, corpusBatch, clearGraph);

    Expr weights;
    if(toBeWeighted_)
      weights = weighter_->getWeights(graph, corpusBatch);

    Expr ce = loss_->getCrossEntropy(state->getLogProbs(),
                                     state->getTargetIndices(),
                                     state->getTargetMask(),
                                     weights);

    Expr sentenceCE = sum(ce, /*axis = */-3);

    const auto& sentenceIds = batch->getSentenceIds();
    std::cerr << sentenceIds.size() << std::endl;
    for(const auto& agent : scores_)  {
      std::vector<float> batchAgent;
      batchAgent.reserve(sentenceIds.size());
      for(size_t i : sentenceIds)
        batchAgent.push_back(agent[i]);
      
      int dim = (int)sentenceIds.size();
      Expr agentProbs = graph->constant({dim, 1}, inits::from_vector(batchAgent));

      float denominator = (float)scores_.size();

      // sentenceCE (a) is negative \sum log(P) and agentProbs (b) is just \sum log(P),
      // hence abs(a + b) to compute absolute difference.
      // Averaged over number of agents (denominator). 
      sentenceCE = sentenceCE + abs(sentenceCE + agentProbs) / denominator;
    }

    // ce-mean-words for now
    auto cost = sum(sentenceCE, /*axis =*/ -2) 
                / sum(sum(state->getTargetMask(), /*axis =*/ -3), /*axis =*/ -2); 

    return cost;
  }

};

class Trainer : public ModelBase {
protected:
  Ptr<ModelBase> model_;
  Ptr<CostBase> cost_;

public:
  Trainer(Ptr<ModelBase> model, Ptr<CostBase> cost)
      : model_(model), cost_(cost) {}

  Ptr<ModelBase> getModel() { return model_; }

  virtual void load(Ptr<ExpressionGraph> graph,
                    const std::string& name,
                    bool markedReloaded = true) override {
    model_->load(graph, name, markedReloaded);
  };

  virtual void save(Ptr<ExpressionGraph> graph,
                    const std::string& name,
                    bool saveTranslatorConfig = false) override {
    model_->save(graph, name, saveTranslatorConfig);
  }

  virtual Expr build(Ptr<ExpressionGraph> graph,
                     Ptr<data::Batch> batch,
                     bool clearGraph = true) override {
    return cost_->apply(model_, graph, batch, clearGraph);
  };

  virtual void clear(Ptr<ExpressionGraph> graph) override { model_->clear(graph); };
};

typedef Trainer Scorer;

class CostStep {
public:
  virtual Ptr<DecoderState> apply(Ptr<DecoderState> state) = 0;
};

class LogSoftmaxStep : public CostStep {
public:
  virtual Ptr<DecoderState> apply(Ptr<DecoderState> state) override {
    // decoder needs normalized probabilities (note: skipped if beam 1 and --skip-cost)
    auto logits = state->getLogProbs();
    
    auto logprobs = logsoftmax(logits);

    state->setLogProbs(logprobs);
    return state;
  }
};

// Gumbel-max noising for sampling during beam-search
// Seems to work well enough with beam-size=1. Turn on
// with --output-sampling during translation with marian-decoder
class GumbelSoftmaxStep : public CostStep {
public:
  virtual Ptr<DecoderState> apply(Ptr<DecoderState> state) override {
    auto logits = state->getLogProbs();
    
    auto logprobs = logsoftmax(logits + constant_like(logits, inits::gumbel));

    state->setLogProbs(logprobs);
    return state;
  }
};

// class to wrap an EncoderDecoderBase and a CostStep that are executed in sequence,
// wrapped again in the EncoderDecoderBase interface
// @TODO: seems we are conflating an interface defition with its implementation?
class Stepwise : public EncoderDecoderBase {
protected:
  Ptr<EncoderDecoderBase> encdec_;
  Ptr<CostStep> cost_;

public:
  Stepwise(Ptr<EncoderDecoderBase> encdec, Ptr<CostStep> cost)
      : encdec_(encdec), cost_(cost) {}

  virtual void load(Ptr<ExpressionGraph> graph,
                    const std::string& name,
                    bool markedReloaded = true) override {
    encdec_->load(graph, name, markedReloaded);
  }

  virtual void mmap(Ptr<ExpressionGraph> graph,
                    const void* ptr,
                    bool markedReloaded = true) override {
    encdec_->mmap(graph, ptr, markedReloaded);
  };

  virtual void save(Ptr<ExpressionGraph> graph,
                    const std::string& name,
                    bool saveTranslatorConfig = false) override {
    encdec_->save(graph, name, saveTranslatorConfig);
  }

  virtual void clear(Ptr<ExpressionGraph> graph) override { encdec_->clear(graph); }

  virtual Expr build(Ptr<ExpressionGraph> graph,
                     Ptr<data::Batch> batch,
                     bool clearGraph = true) override {
    auto corpusBatch = std::static_pointer_cast<data::CorpusBatch>(batch);
    return build(graph, corpusBatch, clearGraph);
  }

  virtual Ptr<DecoderState> startState(Ptr<ExpressionGraph> graph,
                                       Ptr<data::CorpusBatch> batch) override {
    return encdec_->startState(graph, batch);
  }

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState> state,
                                 const std::vector<IndexType>& hypIndices,
                                 const std::vector<IndexType>& embIndices,
                                 int dimBatch,
                                 int beamSize) override {
    auto nextState = encdec_->step(
        graph, state, hypIndices, embIndices, dimBatch, beamSize);
    return cost_->apply(nextState);
  }

  virtual Expr build(Ptr<ExpressionGraph> /*graph*/,
                     Ptr<data::CorpusBatch> /*batch*/,
                     bool /*clearGraph*/ = true) override {
    ABORT("Wrong wrapper. Use models::Trainer or models::Scorer");
    return nullptr;
  }

  virtual Ptr<Options> getOptions() override { return encdec_->getOptions(); };

  virtual void setShortlistGenerator(
      Ptr<data::ShortlistGenerator> shortlistGenerator) override {
    encdec_->setShortlistGenerator(shortlistGenerator);
  };

  virtual Ptr<data::Shortlist> getShortlist() override {
    return encdec_->getShortlist();
  };

  virtual data::SoftAlignment getAlignment() override { return encdec_->getAlignment(); }
};

inline Ptr<ModelBase> add_cost(Ptr<EncoderDecoder> encdec,
                               Ptr<Options> options) {
  switch(options->get<usage>("usage", usage::raw)) {
    case usage::training:
      if(options->has("multi-agent-learning"))
        return New<Trainer>(encdec, New<MultiAgentEncoderDecoderCE>(options));
      else
        return New<Trainer>(encdec, New<EncoderDecoderCE>(options));
    case usage::scoring:
      return New<Scorer>(encdec, New<EncoderDecoderCE>(options));
    case usage::translation:
      if(options->get<bool>("output-sampling", false))
        return New<Stepwise>(encdec, New<GumbelSoftmaxStep>());
      else
        return New<Stepwise>(encdec, New<LogSoftmaxStep>());
    case usage::raw:
    default: return encdec;
  }
}
}  // namespace models
}  // namespace marian
