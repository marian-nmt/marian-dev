#include "translator/scorers.h"
#include "common/io.h"

namespace marian {

Ptr<Scorer> scorerByType(const std::string& fname,
                         float weight,
                         const std::string& model,
                         Ptr<Options> options) {
  options->set("inference", true);
  std::string type = options->get<std::string>("type");

  // @TODO: solve this better
  if(type == "lm" && options->has("input")) {
    size_t index = options->get<std::vector<std::string>>("input").size();
    options->set("index", index);
  }

  bool skipCost = options->get<bool>("skip-cost");
  auto encdec = models::createModelFromOptions(
      options, skipCost ? models::usage::raw : models::usage::translation);

  LOG(info, "Loading scorer of type {} as feature {}", type, fname);

  return New<ScorerWrapper>(encdec, fname, weight, model);
}

Ptr<Scorer> scorerByType(const std::string& fname,
                         float weight,
                         const void* ptr,
                         Ptr<Options> options) {
  options->set("inference", true);
  std::string type = options->get<std::string>("type");

  // @TODO: solve this better
  if(type == "lm" && options->has("input")) {
    size_t index = options->get<std::vector<std::string>>("input").size();
    options->set("index", index);
  }

  bool skipCost = options->get<bool>("skip-cost");
  auto encdec = models::createModelFromOptions(
      options, skipCost ? models::usage::raw : models::usage::translation);

  LOG(info, "Loading scorer of type {} as feature {}", type, fname);

  return New<ScorerWrapper>(encdec, fname, weight, ptr);
}

std::vector<Ptr<Scorer>> createScorers(Ptr<Options> options) {
  std::vector<Ptr<Scorer>> scorers;

  auto models = options->get<std::vector<std::string>>("models");

  std::vector<float> weights(models.size(), 1.f);
  if(options->hasAndNotEmpty("weights"))
    weights = options->get<std::vector<float>>("weights");

  size_t i = 0;
  for(auto model : models) {
    std::string fname = "F" + std::to_string(i);

    // load options specific for the scorer
    auto modelOptions = New<Options>(options->clone());
    try {
      if(!options->get<bool>("ignore-model-config")) {
        YAML::Node modelYaml;
        io::getYamlFromModel(modelYaml, "special:model.yml", model);
        modelOptions->merge(modelYaml, true);
      }
    } catch(std::runtime_error&) {
      LOG(warn, "No model settings found in model file");
    }

    scorers.push_back(scorerByType(fname, weights[i], model, modelOptions));
    i++;
  }

  return scorers;
}

std::vector<Ptr<Scorer>> createScorers(Ptr<Options> options, const std::vector<const void*>& ptrs) {
  std::vector<Ptr<Scorer>> scorers;

  std::vector<float> weights(ptrs.size(), 1.f);
  if(options->hasAndNotEmpty("weights"))
    weights = options->get<std::vector<float>>("weights");

  size_t i = 0;
  for(auto ptr : ptrs) {
    std::string fname = "F" + std::to_string(i);

    // load options specific for the scorer
    auto modelOptions = New<Options>(options->clone());
    try {
      if(!options->get<bool>("ignore-model-config")) {
        YAML::Node modelYaml;
        io::getYamlFromModel(modelYaml, "special:model.yml", ptr);
        modelOptions->merge(modelYaml, true);

        // check if weight matrices are packed format in case packed gemms are used.
        std::string gemmType = modelOptions->get<std::string>("gemm-type");
        bool packedWeight = modelOptions->get<bool>("packed-weight");
        ABORT_IF(
            !packedWeight && (gemmType == "fp16packed" || gemmType == "int8packed"),
            "Weight matrices should be in packed format when packed gemms are used: " + gemmType);
        ABORT_IF(packedWeight && gemmType != "fp16packed" && gemmType != "int8packed",
                 "Weight matrices should not be in packed format when normal gemms are used: "
                     + gemmType);
      }
    } catch(std::runtime_error&) {
      LOG(warn, "No model settings found in model file");
    }

    scorers.push_back(scorerByType(fname, weights[i], ptr, modelOptions));
    i++;
  }

  return scorers;
}

void convertModelScorer(Ptr<Options> options, Ptr<ExpressionGraph> graph) {
  auto models = options->get<std::vector<std::string>>("models");

  size_t i = 0;
  for(auto model : models) {
    std::string fname = "F" + std::to_string(i);

    // load options specific for the scorer
    auto modelOptions = New<Options>(options->clone());
    try {
      if(!options->get<bool>("ignore-model-config")) {
        YAML::Node modelYaml;
        io::getYamlFromModel(modelYaml, "special:model.yml", model);
        modelOptions->merge(modelYaml, true);

        // check if weight matrices are packed format in case packed gemms are used.
        bool packedWeight = modelOptions->get<bool>("packed-weight");

        // need to pack and save only if the model is not already packed.
        if(!packedWeight) {
          auto scorer = scorerByType(fname, 1.f, model, modelOptions);

          // load models
          scorer->init(graph);
          graph->forward();

          //std::string name = modelOptions->get<std::string>("model");

          scorer->save(graph, model + ".pack.npz");
        }
      }
    } catch(std::runtime_error&) {
      LOG(warn, "No model settings found in model file");
    }
    i++;
  }
}

}  // namespace marian
