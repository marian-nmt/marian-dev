#pragma once

#include "layers_new/interface.h"
#include "layers_new/neuralnet.h"

namespace marian {
namespace nn {

struct CellState {
  Expr recurrent;
  size_t position = 0;
};

struct ICell {
  virtual void initState(Ptr<CellState> state) const = 0;
  virtual std::vector<Expr> applyToInput(Expr input) const = 0;
  virtual Expr applyToState(const std::vector<Expr>& inputs, Expr mask, Ptr<CellState> state) const = 0;
};

class SSRU final : public Layer, public ICell {
protected:
  using Layer::namedLayers_;

public:
  Ptr<Linear> iProj; // input projection
  Ptr<Linear> fProj; // forget gate projection
  Ptr<Dropout> dropout;

  int dimState; // state dimension

  SSRU(Ptr<ExpressionGraph> graph, int dimState, float dropProb = 0.f) : Layer(graph), dimState(dimState) {
    iProj = New<Linear>(graph, dimState, /*useBias=*/false);
    registerLayer(iProj);
    fProj = New<Linear>(graph, dimState);
    registerLayer(fProj);
    dropout = New<Dropout>(graph, dropProb, Shape::Axes({-1}));
    registerLayer(dropout);
  }

  virtual void initState(Ptr<CellState> state) const override {
    state->recurrent = graph()->constant({1, 1, 1, dimState}, inits::zeros());
    state->position = 0;
  }

  std::vector<Expr> applyToInput(Expr input) const override {
    int dimModel = input->shape()[-1];
    ABORT_IF(dimModel != dimState, "Model dimension {} has to match state dimension {}", dimModel, dimState);

    input = dropout->apply(input);

    Expr output = iProj->apply(input);
    Expr forget = fProj->apply(input);

    return {output, forget};
  }

  Expr applyToState(const std::vector<Expr>& inputs, Expr mask, Ptr<CellState> state) const override {
    auto prevRecurrent = state->recurrent;
    auto input  = inputs[0];
    auto forget = inputs[1];

    auto nextRecurrent = highway(/*input1=*/prevRecurrent, /*input2=*/input, /*gate=*/forget); // rename to "gate"?
    auto nextOutput    = relu(nextRecurrent);

    // @TODO: not needed? nextRecurrent = mask ? mask * nextRecurrent : nextRecurrent;
    state->recurrent = nextRecurrent;

    nextOutput    = mask ? mask * nextOutput : nextOutput;
    return nextOutput;
  }
};

template <class Cell>
class RNN final : public Layer, public IBinaryLayer, public IBinaryDecoderLayer {
protected:
  using Layer::namedLayers_;

public:
  Ptr<Cell> cell;
  Ptr<Linear> oProj;

  RNN(Ptr<ExpressionGraph> graph, int dimState, bool outputProjection = false)
  : Layer(graph) {
    cell = New<Cell>(graph, dimState);
    registerLayer(cell);

    if(outputProjection) {
      oProj = New<Linear>(graph, dimState);
      registerLayer(oProj);
    }
  }

  virtual void initState(Ptr<DecoderState> state) const override {
    ABORT("Remove this abort once this is actually used in the decoder");
    auto cellState = New<CellState>();
    cell->initState(/*in/out=*/cellState);
    state->as<nn::DecoderStateItem>()->set(cellState->recurrent);
    state->setPosition(cellState->position);
  }

  virtual Expr apply(Expr input, Expr inputMask = nullptr) const override {
    auto state = New<DecoderStateItem>(graph()->constant({1, 1, 1, cell->dimState}, inits::zeros()), /*position=*/0);
    return apply(input, inputMask, state);
  }

  virtual Expr apply(Expr input, Expr inputMask, Ptr<DecoderState> state) const override {
    auto cellState = New<CellState>();
    cellState->recurrent = state->as<nn::DecoderStateItem>()->get();

    // during decoding time is of dimension 1, so this is a no-op (reshape in fact)
    input = swapTimeBatch(input); // [beam, time, batch, dim]
    if(inputMask)
      // same here
      inputMask = swapTimeBatch(inputMask);
    int dimTimeAxis = -3;

    std::vector<Expr> inputs = cell->applyToInput(input);

    // @TODO: this could be implemented as a special kernel/operator
    std::vector<Expr> outputs;
    for(int i = 0; i < input->shape()[dimTimeAxis]; ++i) {
      std::vector<Expr> stepInputs(inputs.size());
      std::transform(inputs.begin(), inputs.end(), stepInputs.begin(),
                     [i, dimTimeAxis](Expr e) { return slice(e, dimTimeAxis, i); });
      cellState->position = state->getPosition() + i;
      auto stepMask = inputMask;
      if(stepMask)
         stepMask = slice(inputMask, dimTimeAxis, i);

      Expr output = cell->applyToState(stepInputs, stepMask, /*in/out=*/cellState);
      outputs.push_back(output);
    }

    state->as<nn::DecoderStateItem>()->set(cellState->recurrent);

    // during decoding again, this is a no-op
    Expr output = swapTimeBatch(concatenate(outputs, dimTimeAxis));
    if(oProj)
      output = oProj->apply(output);

    return output;
  }
};

}
}