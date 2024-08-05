#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "marian.h"

#include "common/logging.h"
#include "common/timer.h"
#include "data/batch_generator.h"
#include "data/corpus.h"
#include "data/text_input.h"
#include "evaluator/evaluator.h"
#include "models/model_task.h"


using namespace marian;

namespace pymarian {

  //type aliases for convenience
  using StrVector = std::vector<std::string>;
  using StrVectors = std::vector<StrVector>;
  using FloatVector = std::vector<float>;
  using FloatVectors = std::vector<FloatVector>;
  using Evaluator = marian::Evaluate<marian::Evaluator>;
  namespace py = pybind11;

  /**
   * Wrapper for Marian Evaluator.
   *
   * This class is a wrapper for the Marian Evaluator class.
   * It is used to run the evaluator on a given input.
   *
   **/
  class EvaluatorPyWrapper {

  private:
    Ptr<marian::Options> options_;
    Ptr<Evaluator> evaluator_;
    std::vector<Ptr<Vocab>> vocabs_;

  public:
  /**
   * Constructor for the EvaluatorPyWrapper class.
   * @param cliString - the command line string to parse as Marian options
   */
    EvaluatorPyWrapper(const std::string& cliString){
      options_ = parseOptions(cliString, cli::mode::evaluating, true)
      ->with("inference", true, "shuffle", "none");
      evaluator_ = New<Evaluator>(options_);
      vocabs_ = loadVocabs(options_);
    }

    /**
     * @brief Load the vocabularies from the given paths
     * @param options - the options object
     * @return vector of vocabularies
    */
    static auto loadVocabs(Ptr<marian::Options> options) -> std::vector<Ptr<Vocab>> {
      std::vector<Ptr<Vocab>> vocabs;
      auto vocabPaths = options->get<std::vector<std::string>>("vocabs");
      LOG(info, "Loading vocabularies from {}", utils::join(vocabPaths, ", "));
      for (size_t i = 0; i < vocabPaths.size(); ++i) {
        Ptr<Vocab> vocab = New<Vocab>(options, i);
        vocab->load(vocabPaths[i]);
        vocabs.emplace_back(vocab);
      }
      return vocabs;
    }

    /**
     * Given a table of strings (i.e., rows x columns), concatenate each column into a single string.
     *
     * @param data - table of strings : rows x columns
     * @return List of strings, one string for each column, concatenated across rows.
    */
    static auto concatColumns(const StrVectors& data) -> StrVector {
      // Get the number of rows and columns in the data
      int rows = data.size();
      int cols = data[0].size();
      StrVector result(cols);

      for (int j = 0; j < cols; j++) {
        std::string column = "";
        for (int i = 0; i < rows; i++) {
          column += data[i][j];
          // If it is not the last row, add a newline character
          if (i != rows - 1) { column += "\n";}
        }
        result[j] = column;
      }
      return result;
    }

    /**
     * Run the evaluator on the given input.
     * Input is transformed as (in memory) files by concatenating columns.
     *
     * @param inputs - table of strings : rows x columns
     * @return table of floats : rows x columns
     *
    */
    auto run(const StrVectors& inputs) -> FloatVectors {
      StrVector columnFiles = concatColumns(inputs);
      auto corpus = New<data::TextInput>(columnFiles, vocabs_, options_);
      corpus->prepare();

      auto batchGenerator = New<BatchGenerator<data::TextInput>>(corpus, options_, nullptr, /*runAsync=*/false);
      batchGenerator->prepare();

      std::string output = options_->get<std::string>("output");
      Ptr<BufferedVectorCollector> collector = New<BufferedVectorCollector>(output, /*binary=*/false);
      evaluator_->run(batchGenerator, collector);
      FloatVectors outputs = collector->getBuffer();
      return outputs;
    }

    auto getModelConfig() -> std::string {
      return evaluator_->getModelConfig();
    }

  };

}
