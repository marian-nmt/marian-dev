#include "translator/scorers.h"
#include "common/io.h"

namespace marian {

Ptr<Scorer> scorerByType(const std::string& fname,
                         float weight,
                         Ptr<io::ModelWeights> modelFile,
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

  return New<ScorerWrapper>(encdec, fname, weight, modelFile);
}

std::vector<Ptr<Scorer>> createScorers(Ptr<Options> options, const std::vector<Ptr<io::ModelWeights>>& modelFiles) {
  std::vector<Ptr<Scorer>> scorers;

  std::vector<float> weights(modelFiles.size(), 1.f);
  if(options->hasAndNotEmpty("weights"))
    weights = options->get<std::vector<float>>("weights");

  bool isPrevRightLeft = false;  // if the previous model was a right-to-left model
  size_t i = 0;
  for(auto modelFile : modelFiles) {
    std::string fname = "F" + std::to_string(i);

    // load options specific for the scorer
    auto modelOptions = options->clone();
    if(!options->get<bool>("ignore-model-config")) {
      YAML::Node modelYaml = modelFile->getYamlFromModel("special:model.yml");
      if(!modelYaml.IsNull()) {
        LOG(info, "Loaded model config");
        modelOptions->merge(modelYaml, true);
      }
      else {
        LOG(warn, "No model settings found in model file");
      }
    }

    // l2r and r2l cannot be used in the same ensemble
    if(modelFiles.size() > 1 && modelOptions->has("right-left")) {
      if(i == 0) {
        isPrevRightLeft = modelOptions->get<bool>("right-left", false);
      } else {
        // abort as soon as there are two consecutive models with opposite directions
        ABORT_IF(isPrevRightLeft != modelOptions->get<bool>("right-left", false),
                 "Left-to-right and right-to-left models cannot be used together in ensembles");
        isPrevRightLeft = modelOptions->get<bool>("right-left", false);
      }
    }

    scorers.push_back(scorerByType(fname, weights[i], modelFile, modelOptions));
    i++;
  }

  return scorers;
}

std::vector<Ptr<Scorer>> createScorers(Ptr<Options> options) {
  std::vector<Ptr<io::ModelWeights>> modelFiles;
  auto models = options->get<std::vector<std::string>>("models");
  for(auto model : models) {
    auto modelFile = New<io::ModelWeights>(model);
    modelFiles.push_back(modelFile);
  }

  return createScorers(options, modelFiles);
}

}  // namespace marian
