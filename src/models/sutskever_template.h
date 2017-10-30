#pragma once

#include "marian.h"

namespace marian {

class EncoderSutskever : public EncoderBase {
public:
  EncoderSutskever(Ptr<Options> options) : EncoderBase(options) {}

  Ptr<EncoderState> build(Ptr<ExpressionGraph> graph,
                          Ptr<data::CorpusBatch> batch) {
    using namespace keywords;

    return New<EncoderState>(nullptr, nullptr, batch);
  }

  void clear() {}
};

class DecoderSutskever : public DecoderBase {
public:
  DecoderSutskever(Ptr<Options> options) : DecoderBase(options) {}

  virtual Ptr<DecoderState> startState(
      Ptr<ExpressionGraph> graph,
      Ptr<data::CorpusBatch> batch,
      std::vector<Ptr<EncoderState>>& encStates) {
    using namespace keywords;

    rnn::States startStates;
    return New<DecoderState>(startStates, nullptr, encStates);
  }

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState> state) {
    using namespace keywords;

    rnn::States decoderStates;
    return New<DecoderState>(decoderStates, nullptr, state->getEncoderStates());
  }

  void clear() {}
};
}
