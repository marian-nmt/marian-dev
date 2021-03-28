#pragma once
/* Support for swapping models in and out of a GPU, when you have more models
 * than fit in the GPU's RAM.  The models must have identical graphs, including
 * size. They can have different parameters and different vocabularies but the
 * vocabularies must have the same size.  To make vocabulary the same size, pad
 * using scripts/contrib/pad_model_vocabulary.py offline.
 */
#include "marian.h"
#include "common/io.h"
#include "data/corpus.h"
#include "data/text_input.h"
#include "translator/translator.h"
#include <vector>
namespace marian {

/* A model loaded on the CPU and possibly on a GPU.
 */
class SwappableModel {
  private:
    std::vector<io::Item> parameters_;
    std::vector<Ptr<Vocab>> srcVocabs_;
    Ptr<Vocab> trgVocab_;

  public:
    // The parts of Options that relate to model and vocab are ignored.  The files provided will be loaded.
    SwappableModel(Ptr<Options> options, const std::string &parameters, const std::vector<std::string> &sourceVocabPaths, const std::string &targetVocabPath);

    const std::vector<io::Item> &Parameters() const { return parameters_; }

    const std::vector<Ptr<Vocab>> &SrcVocabs() const { return srcVocabs_; }

    Ptr<Vocab> TrgVocab() const { return trgVocab_; }
};

/* Reserved space on a GPU with which to translate. If you can afford to fit
 * multiple models on 1 GPU, then each one that fits is a GPUSlot
 */
class SwappableSlot {
	private:
    Ptr<Options> options_;
    Ptr<ExpressionGraph> graph_;
    std::vector<Ptr<Scorer> > scorers_;

    // Last model used for translation.  Used to skip loading.
    const SwappableModel *loadedModel_;

    void Load(const std::vector<io::Item> &parameters);

  public:
    explicit SwappableSlot(Ptr<Options> options);

    void Translate(const SwappableModel &model, const std::vector<std::string> &input);
};

} // namespace marian
