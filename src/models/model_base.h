#pragma once

#include <string>
#include "marian.h"
#include "common/io_item.h"
#include "layers/loss.h"
#include "layers/generic.h"

namespace marian {
namespace models {

enum struct usage {
  raw,
  training,
  scoring,
  translation,
  embedding,   // used for laser and other models to produce embedding vectors
  evaluating   // evaluating is a special mode for neural metrics, different from (probabilistic) scoring
};

}  // namespace models
}  // namespace marian

YAML_REGISTER_TYPE(marian::models::usage, int)

namespace marian {
namespace models {

// model = input -> predictions
class IModel {
public:
  virtual void load(Ptr<ExpressionGraph>,
                    Ptr<io::ModelWeights>,
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
                    Ptr<io::ModelWeights>,
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
