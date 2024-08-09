#include "marian.h"

// This contains the main function for the aggregate command line that allows to specify
// one of the Marian executables as the first argument. This is done by including all
// individual .cpp files into a single .cpp, using a #define to rename the respective
// main functions.
// For example, the following two are equivalent:
//  marian-scorer ARGS
//  marian score  ARGS
// The supported sub-commands are:
//  train
//  decode
//  score
//  embed
//  vocab
//  convert
// Currently, marian_server is not supported, since it is a special use case with lots of extra dependencies.

#define main mainTrainer
#include "marian_train.cpp"
#undef main
#define main mainDecoder
#include "marian_decoder.cpp"
#undef main
#define main mainScorer
#include "marian_scorer.cpp"
#undef main
#define main mainEmbedder
#include "marian_embedder.cpp"
#undef main
#define main mainEvaluator
#include "marian_evaluator.cpp"
#undef main
#define main mainVocab
#include "marian_vocab.cpp"
#undef main
#define main mainConv
#include "marian_conv.cpp"
#undef main

#include <string>
#include <map>
#include <tuple>
#include "3rd_party/ExceptionWithCallStack.h"
#include "3rd_party/spdlog/details/format.h"

int main(int argc, char** argv) {
  using namespace marian;
  using MainFunc = int(*)(int, char**);
  std::map<std::string, std::tuple<MainFunc, std::string>> subcmds = {
    {"train", {&mainTrainer, "Train a model (default)"}},
    {"decode", {&mainDecoder, "Decode or translate text"}},
    {"score", {&mainScorer, "Score translations"}},
    {"embed", {&mainEmbedder, "Embed text"}},
    {"evaluate", {&mainEvaluator, "Run Evaluator metric"}},
    {"vocab", {&mainVocab, "Create vocabulary"}},
    {"convert", {&mainConv, "Convert model file format"}}
  };
  // no arguments, or the first arg is "?"", print help message
   if (argc == 1 || (argc == 2 && (std::string(argv[1]) == "?") )) {
    std::cout << "Usage: " << argv[0] << " COMMAND [ARGS]" << std::endl;
    std::cout << "Commands:" << std::endl;
    for (auto&& [name, val] : subcmds) {
      std::cerr << fmt::format("{:10}    : {}\n", name, std::get<1>(val));
    }
    return 0;
  }

  if (argc > 1 && argv[1][0] != '-') {
    std::string cmd = argv[1];
    argc--;
    argv[1] = argv[0];
    argv++;
    if (subcmds.count(cmd) > 0) {
      auto [func, desc] = subcmds[cmd];
      return func(argc, argv);
    }
    else {
      std::cerr << "Unknown command: " << cmd << ". Known commands are:" << std::endl;
      for (auto&& [name, val] : subcmds) {
        std::cerr << fmt::format("{:10}    : {}\n", name, std::get<1>(val));
      }
      return 1;
    }
  }
  else
    return mainTrainer(argc, argv);
}
