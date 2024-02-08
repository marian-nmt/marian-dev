#pragma once

#include "common/definitions.h"
#include "common/options.h"
#include "data/batch.h"
#include "data/rng_engine.h"
#include "data/vocab.h"
#include "training/training_state.h"

namespace marian {
namespace data {

template <class SampleType, class Iterator, class Batch>
class DatasetBase {
protected:
  std::vector<std::string> paths_;
  Ptr<Options> options_;
  std::vector<size_t> inputPermutation_; // if not empty, this is used to reorder input fields/batches i.e. [1,0] swaps the first two fields
                                         // currently this is used for comet-kiwi-style metrics where the mt output is the first field
  // Data processing may differ in training/inference settings
  bool inference_{false};

public:
  typedef Batch batch_type;
  typedef Ptr<Batch> batch_ptr; // @TODO: rename to camel case
  typedef Iterator iterator;
  typedef SampleType Sample;

  DatasetBase(std::vector<std::string> paths, Ptr<Options> options)
      : paths_(paths),
        options_(options),
        inputPermutation_(options->get<std::vector<size_t>>("input-reorder", {})),
        inference_(options->get<bool>("inference", false)) {}

  DatasetBase(Ptr<Options> options) : DatasetBase({}, options) {}

  virtual Iterator begin() = 0;
  virtual Iterator end() = 0;
  virtual void shuffle() = 0;

  virtual Sample next() = 0;

  virtual batch_ptr toBatch(const std::vector<Sample>&) = 0;

  virtual void reset() {}
  virtual void prepare() {}
  virtual void restore(Ptr<TrainingState>) {}

  // @TODO: remove after cleaning training/training.h
  virtual Ptr<Options> options() { return options_; }
};


}  // namespace data
}  // namespace marian
