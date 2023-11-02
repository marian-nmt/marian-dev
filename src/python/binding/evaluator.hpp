#include "marian.h"
#include "models/model_task.h"
#include "data/batch_generator.h"
#include "data/corpus.h"
#include "data/text_input.h"
#include "evaluator/evaluator.h"
#include "common/timer.h"
#include "common/logging.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"


using namespace marian;

namespace pymarian {

  //type aliases for convenience
  using StrVector = std::vector<std::string>;
  using StrVectors = std::vector<StrVector>;
  using FloatVector = std::vector<float>;
  using FloatVectors = std::vector<FloatVector>;
  using Evaluator = marian::Evaluate<marian::Evaluator>;
  namespace py = pybind11;

  class PyIteratorInput: public data::TextInput {
    protected:
    py::iterator iter_;

    public:
    PyIteratorInput(py::iterator iterator, std::vector<Ptr<Vocab>> vocabs, Ptr<Options> options)
    : data::TextInput({}, vocabs, options) {
      iter_ = iterator;
    }

    auto next() -> data::SentenceTuple override {
      if (iter_ != py::iterator::sentinel()) {
        auto next_row = iter_->cast<std::vector<std::string>>();
        std::cout << "next_row:\n" << utils::join(next_row, "<tab>") << std::endl;
        auto next_ = encode(next_row, ++pos_);
        ++iter_;
        return next_;
      } else {
        std::cout << "next_row: sentinel (end)" << std::endl;
        return SentenceTupleImpl();
      }
    }

  };

  class EvaluatorPyWrapper {
    
  private:
    Ptr<marian::Options> options_;
    Ptr<Evaluator> evaluator_;
    std::vector<Ptr<Vocab>> vocabs_;

  public:
    EvaluatorPyWrapper(const std::string& cliString){
      options_ = parseOptions(cliString, cli::mode::evaluating, true)
      ->with("inference", true, "shuffle", "none");
      evaluator_= New<Evaluator>(options_);
      vocabs_ =loadVocabs(options_);
    }
 
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

    auto run(const StrVectors& inputs) -> FloatVectors {
      /* Input is table of strings : rows x columns
      We fake input as files by concatinating columns
      TODO: support for iterator of rows
      */
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

    auto run(const StrVector& input) -> FloatVector{
      StrVectors inputs = { input };
      return run(inputs)[0];
    }

    auto run_iter(py::iterator pyIter) -> FloatVectors {
      std::cout << "1. run_iter" << std::endl;

      auto corpus = New<PyIteratorInput>(pyIter, vocabs_, options_);
      corpus->prepare();
      std::cout << "2. corpus done" << std::endl;

      auto batchGenerator = New<BatchGenerator<PyIteratorInput>>(corpus, options_, nullptr, /*runAsync=*/false);
      batchGenerator->prepare();
      std::cout << "3. Batch generaror done" << std::endl;

      std::string output = options_->get<std::string>("output");
      Ptr<BufferedVectorCollector> collector = New<BufferedVectorCollector>(output, /*binary=*/false);

      evaluator_->run(batchGenerator, collector);
      std::cout << "4. run done" << std::endl;

      FloatVectors outputs = collector->getBuffer();
      std::cout << "5. getBuffer done" << std::endl;
      return outputs;
    }
  };


}
