#pragma once

#include <string>
#include "marian.h"
#include "layers/loss.h"

namespace marian {
namespace models {

enum struct usage { raw, training, scoring, translation };
}
}  // namespace marian

YAML_REGISTER_TYPE(marian::models::usage, int)

namespace marian {
namespace models {

class ModelBase {
public:
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

template <class MultiLossType>
class MultiModel : public ModelBase {
private:
  std::vector<Ptr<ModelBase>> models_;

public:
  void push_back(Ptr<ModelBase> model) {
    models_.push_back(model);
  }
 
  virtual void load(Ptr<ExpressionGraph> graph,
                    const std::string& filename,
                    bool markReloaded = true) override {
    ABORT_IF(models_.empty(), "No models in multi model");
    models_[0]->load(graph, filename, markReloaded);
  }
      
  virtual void save(Ptr<ExpressionGraph> graph,
                    const std::string& filename,
                    bool saveTranslatorConfig = false) override {
    ABORT_IF(models_.empty(), "No models in multi model");
    models_[0]->save(graph, filename, saveTranslatorConfig);
  }
      

  virtual Ptr<RationalLoss> build(Ptr<ExpressionGraph> graph,
                                  Ptr<data::Batch> batch,
                                  bool clearGraph = true) override {
    ABORT_IF(models_.empty(), "No models in multi model");
    bool first = true;
    auto sumLoss = New<MultiLossType>();
    for(auto& model : models_) {
      auto partialLoss = model->build(graph, batch, clearGraph && first);
      sumLoss->push_back(*partialLoss);
      first = false;
    }
    return sumLoss;
  }

  virtual void clear(Ptr<ExpressionGraph> graph) override {
    ABORT_IF(models_.empty(), "No models in multi model");
    for(auto& model : models_)
      model->clear(graph);
  }
};

}  // namespace models
}  // namespace marian
