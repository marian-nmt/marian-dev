#pragma once

#include "common/utils.h"
#include "graph/expression_graph.h"
#include "graph/expression_operators.h"
#include "graph/node_initializers.h"

#include "layers_new/interface.h"

namespace marian {
namespace nn {

// Interface: decoder state
class DecoderState : public IClassName, public std::enable_shared_from_this<DecoderState> {
protected:
  size_t position{0};

public:
  DecoderState(size_t position) : position(position) {}
  virtual ~DecoderState() {}

  virtual void incrementPosition() {
    position++;
  }

  virtual size_t getPosition() {
    return position;
  }

  virtual void setPosition(size_t pos) {
    this->position = pos;
  }

  // Dynamic cast to requested layer type. Will return nullptr if not possible
  template <class StateType>
  Ptr<StateType> as() {
    return std::dynamic_pointer_cast<StateType>(shared_from_this());
  }

  // Dynamic cast to requested layer type. Will return nullptr if not possible
  template <class StateType>
  Ptr<StateType> as() const {
    return const_cast<DecoderState*>(this)->as<StateType>();
  }

  // Dynamic cast to requested layer type. Will abort if the cast is not possible.
  template <class StateType>
  Ptr<StateType> cast() {
    auto stateCast = as<StateType>();
    ABORT_IF(!stateCast, "State {} cannot be cast to requested type {}",
             className(),
             utils::cxxTypeName<StateType>());
    return stateCast;
  }

  template <class StateType>
  Ptr<StateType> cast() const {
    return const_cast<DecoderState*>(this)->cast<StateType>();
  }
};

class DecoderStateItem : public DecoderState {
private:
  Expr state_; // state of the decoder at a given position, can be nullptr

public:
  DecoderStateItem(Expr state, size_t position) : DecoderState(position), state_(state) {}
  DecoderStateItem(size_t position) : DecoderState(position) {}
  virtual ~DecoderStateItem() = default;

  Expr get() { return state_; }
  void set(Expr state) { state_ = state; }
};

class DecoderStateList : public DecoderState {
private:
  std::vector<Ptr<DecoderStateItem>> items_;

public:
  DecoderStateList(size_t position) : DecoderState(position) {}
  virtual ~DecoderStateList() = default;

  void incrementPosition() override {
    DecoderState::incrementPosition();
    for(auto& item : items_) {
      item->incrementPosition();
      ABORT_IF(position != item->getPosition(), "Positions out of sync??");
    }
  }

  void setPosition(size_t pos) override {
    DecoderState::setPosition(pos);
    for(auto& item : items_)
      item->setPosition(pos);
  }

  void append(Ptr<DecoderStateItem> item) {
    ABORT_IF(position != item->getPosition(), "DecoderStateList.position ({}) != DecoderStateItem.position ({}) ?", position, item->getPosition());
    items_.push_back(item);
  }

  /**
   * Retrieve DecoderStateItem at index i
   */
  Ptr<DecoderStateItem> at(size_t i) const {
    return items_[i];
  }

  auto begin() -> decltype(items_.begin()) const {
    return items_.begin();
  }

  auto end() -> decltype(items_.end()) const {
    return items_.end();
  }

  size_t size() const { return items_.size(); }
};

class EncoderContext {
private:
  Expr context_;
  Expr contextMask_;

public:
  EncoderContext(Expr context, Expr contextMask)
  : context_(context), contextMask_(contextMask) {}

  virtual Expr getContext() const { return context_; }
  virtual Expr getContextMask() const { return contextMask_; }
};

class DecoderSeq2SeqState : public DecoderStateList, public EncoderContext {
public:
  DecoderSeq2SeqState(size_t position, Expr context, Expr contextMask)
  : DecoderStateList(position), EncoderContext(context, contextMask) {}
};

// Interface: Unary function
struct IUnaryDecoderLayer {
  virtual void initState(Ptr<DecoderState> /*state*/) const  = 0;
  virtual Expr apply(Expr /*input*/, Ptr<DecoderState> /*state*/) const = 0;
};

// Interface: Binary function
struct IBinaryDecoderLayer {
  virtual void initState(Ptr<DecoderState> /*state*/) const  = 0;
  virtual Expr apply(Expr, Expr, Ptr<DecoderState> /*state*/) const = 0;
};

// Interface: Ternary function
struct ITernaryDecoderLayer {
  virtual void initState(Ptr<DecoderState> /*state*/) const  = 0;
  virtual Expr apply(Expr, Expr, Expr, Ptr<DecoderState> /*state*/) const = 0;
};

// Interface: 4ary function
struct IQuaternaryDecoderLayer {
  virtual void initState(Ptr<DecoderState> /*state*/) const  = 0;
  virtual Expr apply(Expr, Expr, Expr, Expr, Ptr<DecoderState> /*state*/) const = 0;
};

// Interface: N-Ary function
struct INaryLayerDecoderLayer {
  virtual void initState(Ptr<DecoderState> /*state*/) const  = 0;
  virtual Expr apply(const std::vector<Expr>& /*inputs*/, Ptr<DecoderState> /*state*/) const = 0;
};

} // namespace nn
} // namespace marian
