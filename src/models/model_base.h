#pragma once

#include <string>
#include "marian.h"
#include "layers/loss.h"
#include "layers/generic.h"

namespace marian {
namespace models {

enum struct usage { raw, training, scoring, translation, embedding };
}
}  // namespace marian


// 'FASTOPT_REGISTER_TYPE'
#if FASTOPT
namespace marian {
namespace fastopt_helpers {

template <>
struct As<marian::models::usage> {
  static marian::models::usage apply(const FastOpt& node) {
    return static_cast<marian::models::usage>(As<int>::apply(node));
  }
};
}  // namespace fastopt_helpers
}  // namespace marian
#endif

namespace marian {
namespace models {

// model = input -> predictions
class IModel {
public:
  virtual void load(Ptr<ExpressionGraph>,
                    const std::string&,
                    bool markReloaded = true)
      = 0;
  virtual void save(Ptr<ExpressionGraph>,
                    const std::string&,
                    bool saveTranslatorConfig = false)
      = 0;

  virtual Logits build(Ptr<ExpressionGraph> graph,
                       Ptr<data::Batch> batch,
                       bool clearGraph = true)
      = 0;

  virtual void clear(Ptr<ExpressionGraph> graph) = 0;
};

// criterion = (input, reference) -> loss
// @TODO: Is there a better name?
class ICriterionFunction {
public:
  virtual ~ICriterionFunction() {}

  virtual void load(Ptr<ExpressionGraph>,
                    const std::string&,
                    bool markReloaded = true)
      = 0;
  virtual void save(Ptr<ExpressionGraph>,
                    const std::string&,
                    bool saveTranslatorConfig = false)
      = 0;

  virtual Ptr<RationalLoss> build(Ptr<ExpressionGraph> graph,
                                  Ptr<data::Batch> batch,
                                  bool clearGraph = true)
      = 0;

  virtual void clear(Ptr<ExpressionGraph> graph) = 0;
};

}  // namespace models
}  // namespace marian
