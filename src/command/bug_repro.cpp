#include "../common/config_parser.h"
#include "../common/options.h"
#include "../data/text_input.h"
#include "../models/model_factory.h"
#include "../models/model_task.h"
#include "../training/scheduler.h"
#include "marian.h"

namespace marian {

class ReproTask : public marian::ModelTask {
public:
  ReproTask() {
  }
  void run() override {
    auto parser = ConfigParser(cli::mode::training);
    // i'm prob leaking memory at the end of run() but i don't care
    const char* argseasy[]
        = {"marian",
           "-c",
           "/home/rihards/exp/marian-adaptive-crash-repro/models/model.npz.repro.yml",
           "-t", "dummy-value", "-t", "dummy-value",
           "--after-batches", "20",
           "--after-epochs", "4",
           "--learn-rate", "0.1",
           "--shuffle", "none",
           "--mini-batch", "1"};
    int argc = sizeof(argseasy) / sizeof(char*);
    // this is as close as i could get to initializing a char** in a sane manner
    char** args = new char*[argc];
    for (int i = 0; i < argc; i++) {
      args[i] = strdup(argseasy[i]);
    }
    auto options = parser.parseOptions(argc, args, false);

    auto builder = models::createCriterionFunctionFromOptions(options, models::usage::training);
    auto optimizer = Optimizer(New<Options>("optimizer", "adam", "learn-rate", 0.01));

    std::vector<std::string> vocabPaths
        = {"/home/rihards/exp/marian-adaptive-crash-repro/models/train.1-to-1.bpe.en-lv.yml",
      "/home/rihards/exp/marian-adaptive-crash-repro/models/train.1-to-1.bpe.en-lv.yml"};
    std::vector<int> maxVocabs = {500, 500};

    std::vector<Ptr<Vocab>> vocabs;
    for(size_t i = 0; i < vocabPaths.size(); i++) {
      Ptr<Vocab> vocab = New<Vocab>(options, i);
      vocab->load(vocabPaths[i], maxVocabs[i]);
      vocabs.emplace_back(vocab);
    }
    std::string sources = "del@@ e@@ tions affecting 13 q 14 are also the most frequent structural genetic ab@@ "
          "err@@ ations in chronic lym@@ pho@@ cy@@ tic leu@@ ka@@ emia ( C@@ ll ) 6,@@ 7 , 8 "
          ".\nthis region is found to be heter@@ oz@@ y@@ g@@ ously deleted in 30 ¬ 60 % and hom@@ "
      "oz@@ y@@ g@@ ously deleted in 10 ¬ 20 % of C@@ ll patien@@ ts@@ 9 .";
    std::string targets
        = "del@@ ē@@ cijas , kas ietekmē 13 q 14 , arī ir visbiežāk sastopa@@ mās strukturālās "
          "ģenē@@ tiskās ab@@ er@@ ācijas hron@@ iskā lim@@ foc@@ ī@@ tiskajā leik@@ ēm@@ ijā ( "
          "H@@ LL ) 6,@@ 7 , 8 .\n30 –@@ 60 % H@@ LL pacientu ir konstatēta šī reģiona heter@@ "
          "oz@@ ig@@ ota del@@ ē@@ cija , savukārt 10 –@@ 20 % H@@ LL pacientu ir konstatēta šī "
      "reģiona hom@@ oz@@ ig@@ ota del@@ ē@@ c@@ ij@@ a@@ 9 .";
    // auto inputs = New<data::TextInput>(std::vector<std::string>({sources, targets}), vocabs, options);
    // auto batches = New<data::BatchGenerator<data::TextInput>>(inputs, options);

    for(size_t i = 0; i < 10; i++) {
      LOG(info, "# NEW OUTER ITER");
      auto state = New<TrainingState>(options->get<float>("learn-rate"));
      auto scheduler = New<Scheduler>(options, state);
      scheduler->registerTrainingObserver(scheduler);
      scheduler->registerTrainingObserver(optimizer);

      Ptr<ExpressionGraph> graph;

      bool first = true;
      scheduler->started();

      graph = New<ExpressionGraph>();
      graph->setDevice({0, DeviceType::cpu});
      graph->reserveWorkspaceMB(128);
      while(scheduler->keepGoing()) {
        LOG(info, "## NEW INNER ITER");
        // if inputs aren't initialized for each epoch, their internal istringstreams get exhausted
        auto inputs
            = New<data::TextInput>(std::vector<std::string>({sources, targets}), vocabs, options);
        auto batches = New<data::BatchGenerator<data::TextInput>>(inputs, options);
        // auto batches = New<data::BatchGenerator<data::TextInput>>(inputs, options);
        batches->prepare();

        for(auto batch : *batches) {
          LOG(info, "### NEW BATCH");
          if(!scheduler->keepGoing()) {
            break;
          }

          auto lossNode = builder->build(graph, batch);
          if (first) {
            graph->graphviz("graph-" + std::to_string(i) + ".gv");
            first = false;
          }
          graph->forward();
          StaticLoss loss = *lossNode;
          graph->backward();

          optimizer->update(graph, 1);
          scheduler->update(loss, batch);
        }

        if(scheduler->keepGoing())
          scheduler->increaseEpoch();
      }
      scheduler->finished();
    }
  }
};
}

int main(int argc, char **argv) {
  auto task = marian::ReproTask();
  task.run();
  return 0;
}
