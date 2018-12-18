#pragma

#include "models/decoder.h"

namespace marian {

class Classifier : public DecoderBase {
public:
  Classifier(Ptr<Options> options) : DecoderBase(options) {}


  virtual Ptr<DecoderState> startState(
      Ptr<ExpressionGraph> graph,
      Ptr<data::CorpusBatch> batch,
      std::vector<Ptr<EncoderState>>& encStates) override {

    rnn::States startStates;
    return New<DecoderState>(startStates, nullptr, encStates, batch);

  }

    virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                   Ptr<DecoderState> state) override {
    
    ABORT_IF(state->getEncoderStates().size() != 1, "Currently only one encoder accepted");
    auto encoderContext = state->getEncoderStates()[0]->getContext();
    
    auto firstWord = marian::step(encoderContext, 0, -3);

    int classes = opt<int>("classes");

    auto layerHidden = mlp::dense(graph)     //
        ("prefix", prefix_ + "_ff_logit_l1")  //
        ("dim", firstWord->shape()[-1])       //
        ("activation", mlp::act::tanh);

    auto layerOut = mlp::output(graph)       //
        ("prefix", prefix_ + "_ff_logit_out") //
        ("dim", classes);

    auto output = mlp::mlp(graph)            //
                  .push_back(layerHidden)     //
                  .push_back(layerOut)        //
                  .construct();

    auto logits = output->apply(firstWord);

    rnn::States decoderStates; // dummy
    auto nextState = New<DecoderState>(decoderStates, logits, state->getEncoderStates(), state->getBatch());
    return nextState;
  }

  virtual void embeddingsFromBatch(Ptr<ExpressionGraph> graph,
                                   Ptr<DecoderState> state,
                                   Ptr<data::CorpusBatch> batch) override {
    int classes = options_->get<int>("classes");
    int dimBatch = batch->size();

    // @TODO: this creates fake ground truth, need to implement reader
    std::vector<IndexType> fakeOutputs(dimBatch, 0);
    for(auto& out : fakeOutputs)
        out = (int)rand() % classes;

    std::vector<float> fakeMask(dimBatch, 1);
    
    auto yMask = graph->constant({1, dimBatch, 1},
                                 inits::fromVector(fakeMask));
    auto yData = graph->indices(fakeOutputs);

    state->setTargetMask(yMask);
    state->setTargetIndices(yData);
  }

  virtual void embeddingsFromPrediction(Ptr<ExpressionGraph> graph,
                                        Ptr<DecoderState> state,
                                        const std::vector<IndexType>& embIdx,
                                        int dimBatch,
                                        int dimBeam) override {}

  virtual const std::vector<Expr> getAlignments(int /*i*/ = 0) override { return {}; };

  virtual void clear() override {};
};

}