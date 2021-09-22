#pragma once
/**
 * Support for swapping and resetting models for the self-adaptive translation
 * mode. The intended use case is to store a read-only copy of the model in
 * `CPULoadedModel`, optionally train on a copy of the parameters using
 * `SwappableModelTrainer` and then transfer either the trained or original
 * model parameters into `GPULoadedModel` for translation. `GPUEngineTrain` and
 * `GPUEngineTranslate` are used for storing the expression graphs for training
 * and translation, respectively, and other related things. Translation on the
 * CPU currently isn't supported.
 *
 * Originally this code was intended to allow multiple models to share a single
 * GPU for translation and be swapped into GPU memory only when needed. However,
 * parts of it, that weren't needed for self-adaptive translation, have been
 * trimmed down since then. Look into the commit history if you want to revive
 * this functionality.
 */
#include "common/io.h"
#include "data/vocab.h"
#include "marian.h"
#include "training/scheduler.h"
#include "translator/history.h"

#include <string>
#include <vector>
namespace marian {

class SwappableModelTrainer;

class Scorer;

class GPULoadedModel;
class CPULoadedModel;


/**
  * The class wraps an expression graph and a model builder that are used by
  * `SwappableModelTrainer` for training a model.
  */
class GPUEngineTrain {
private:
  friend class SwappableModelTrainer;
  friend class GPULoadedModel;
  Ptr<Options> options_;
  Ptr<ExpressionGraph> graph_;
  Ptr<models::ICriterionFunction> builder_;
  const DeviceId myDeviceId_;

  void RecreateGraphAndBuilder();

public:
  /**
    * @param options The marian options object
    * @param deviceNum The index of the device you want to use for this slot. Note that this is not the deviceID but the index of the device in the
    *                  array of supplied devices. Eg if you provide -d 0 3 5 and you want the Slot to run on GPU 3, you provide deviceNum=1.
    */
  explicit GPUEngineTrain(Ptr<Options> options, size_t deviceNum);

  ~GPUEngineTrain();
};

/**
 * @brief Wraps a `GPUEngineTrain` and a `CPULoadedModel` and performs model
 * training.
 *
 * This class is created with self-adaptive translation in mind. Each invocation
 * of Train() resets the model parameters at the start of training.
 */
class SwappableModelTrainer {
  private:
    friend class GPULoadedModel;

    Ptr<GPUEngineTrain> engine_;

    Ptr<CPULoadedModel> cpuModel_;
    std::vector<Ptr<Vocab>> srcVocabs_;
    Ptr<Vocab> trgVocab_;

  public:
    SwappableModelTrainer(Ptr<GPUEngineTrain> gpu);

    ~SwappableModelTrainer();

    const std::vector<Ptr<Vocab>> &SrcVocabs() const { return srcVocabs_; }

    Ptr<Vocab> TrgVocab() const { return trgVocab_; }

    /// Change the internal pointers to vocabularies and CPULoadedModel to
    /// different ones
    void SetModel(Ptr<CPULoadedModel> from);

    std::vector<MemoryPiece::PtrType> Parameters() const;

    /**
     * @brief resets the training graph, reloads the model parameters and trains
     * the model on the provided inputs.
     *
     * Intended to be used in the self-adaptive translation mode -- training is
     * always performed on the original model parameters, each training
     * invocation resets previous changes.
     *
     * @param input Training data. A vector representing a parallel corpus --
     * vector elements are the different sides of a parallel corpus, each is a
     * newline separated set of sentences in a single language.
     */
    void Train(const std::vector<std::string> &input);
};

/**
  * The class wraps an expression graph and scorers that are used by
  * `GPULoadedModel` for translation.
  */
class GPUEngineTranslate {
private:
  friend class GPULoadedModel;
  Ptr<Options> options_;
  Ptr<ExpressionGraph> graph_;
  std::vector<Ptr<Scorer>> scorers_;
  const DeviceId myDeviceId_;
  Allocator allocator_;

  void SwapPointers(std::vector<MemoryPiece::PtrType> &with);

public:
  /**
    * @param options The marian options object
    * @param deviceNum The index of the device you want to use for this slot. Note that this is not the deviceID but the index of the device in the
    *                  array of supplied devices. Eg if you provide -d 0 3 5 and you want the Slot to run on GPU 3, you provide deviceNum=1.
    */
  explicit GPUEngineTranslate(Ptr<Options> options, size_t deviceNum);

  ~GPUEngineTranslate();
};

/** A model loaded on the GPU that can be overwritten from CPU. Facilitates
  * translation with the model.
  */
class GPULoadedModel {
  private:
    Ptr<GPUEngineTranslate> engine_;

    std::vector<MemoryPiece::PtrType> parameters_;
    std::vector<Ptr<Vocab>> srcVocabs_;
    Ptr<Vocab> trgVocab_;

  public:
    GPULoadedModel(Ptr<GPUEngineTranslate> gpu);

    ~GPULoadedModel();

    const std::vector<Ptr<Vocab>> &SrcVocabs() const { return srcVocabs_; }

    Ptr<Vocab> TrgVocab() const { return trgVocab_; }

    /// Overwrite this model with parameters from a different one.
    void Load(const CPULoadedModel &from);
    /**
     * @brief Set the internal shared pointers to model parameters and
     * vocabularies to different ones
     *
     * The effect is similar to `Load()` but nothing is copied in the process.
     *
     * @param from Swappable model trainer from which to take the shared
     * pointers to model parameters and vocabularies.
     */
    void PointToParams(const SwappableModelTrainer &from);

    Histories Translate(const Ptr<data::CorpusBatch> batch);
};

/**
  * A model loaded on the CPU. Holds model parameters and vocabularies.
  */
class CPULoadedModel {
  private:
    std::vector<io::Item> parameters_;
    std::vector<Ptr<Vocab>> srcVocabs_;
    Ptr<Vocab> trgVocab_;

  public:
    // The parts of Options that relate to model and vocab are ignored. The
    // files provided will be loaded.
    CPULoadedModel(Ptr<Options> options, const std::string &parameters, const std::vector<std::string> &sourceVocabPaths, const std::string &targetVocabPath);

    const std::vector<io::Item> &Parameters() const { return parameters_; }

    const std::vector<Ptr<Vocab>> &SrcVocabs() const { return srcVocabs_; }

    Ptr<Vocab> TrgVocab() const { return trgVocab_; }
};

} // namespace marian
