#pragma once

#include "common/config.h"
#include "common/utils.h"
#include "data/batch_generator.h"
#ifndef _MSC_VER // @TODO: include SqLite in Visual Studio project
#include "data/corpus_sqlite.h"
#endif
#include "models/model_task.h"
#include "training/scheduler.h"
#include "training/validator.h"

namespace marian {

template <class ModelWrapper>
class Train : public ModelTask {
private:
  Ptr<Options> options_;
  void installCustomSignalHandlers();

public:
  Train(Ptr<Options> options) : options_(options) {}

  void run() override {
    using namespace data;

    // MPI init should be first thing in training
    auto mpi = initMPI(/*multiThreaded=*/!options_->get<bool>("sync-sgd")); // @TODO: do we need the multiThreaded distinction at all?

    if(mpi) { // if we run MPI, then make sure to sync seed across processes as first action
      mpi->bCast(&Config::seed, 1, IMPIWrapper::getDataType(&Config::seed));
      LOG(info, "Synced seed {}", Config::seed);
    }

    Ptr<CorpusBase> dataset;
    auto corpusSeed = Config::seed + (mpi ? mpi->myMPIRank() : 0); // @BUGBUG: no correct resume right now
    if(!options_->get<std::string>("sqlite").empty())
#ifndef _MSC_VER // @TODO: include SqLite in Visual Studio project
      dataset = New<CorpusSQLite>(options_, /*translate=*/false, corpusSeed);
#else
      ABORT("SqLite presently not supported on Windows");
#endif
    else
      dataset = New<Corpus>(options_, /*translate=*/false, corpusSeed);

    dataset->prepare();

    // We run training in a do-while loop. It should only restart if a fp16 training run was interrupted
    // via the throwing of a DivergenceException from training/scheduler.h and if --throw-on-divergence and
    // --fp16-fallback-to-fp32 are enabled. 
    // The repeated training run will continue from last checkpoint (similar to a manually interrupted training) 
    // but attempt training in fp32. If that training run or any other fp32 training happens to diverge, 
    // training will exit with an unhandled DivergenceException. This is on purpose to indicate a fatal error.
    bool restartTraining;
    do {
      try {
        // there will be only one training loop execution unless in special situations,
        // for example, when fp16 training diverges and it is restarted with fp32
        restartTraining = false;

        Ptr<BatchStats> stats;
        if(options_->get<bool>("mini-batch-fit")) {
          LOG(info,
              "[batching] Collecting statistics for batch fitting with step size {}",
              options_->get<size_t>("mini-batch-fit-step"));
          // @TODO this should receive a function object that can generate a fake batch;
          // that way vocabs would not be exposed.
          auto model = New<ModelWrapper>(options_, mpi);

          // use temporary scheduler to make sure everything gets destroyed properly
          // otherwise the scheduler believes that registered objects still exist
          auto tempTrainState = New<TrainingState>(options_->get<float>("learn-rate"));
          auto tempScheduler = New<Scheduler>(options_, tempTrainState, mpi);

          model->setScheduler(tempScheduler); // collectStats() needs to know about dynamic MB scaling
          stats = model->collectStats(dataset->getVocabs());
          LOG(info, "[batching] Done. Typical MB size is {} target words", utils::withCommas(stats->estimateTypicalTrgWords()));
        }

        auto trainState = New<TrainingState>(options_->get<float>("learn-rate"));
        auto scheduler = New<Scheduler>(options_, trainState, mpi);

        if((options_->hasAndNotEmpty("valid-sets") || options_->hasAndNotEmpty("valid-script-path"))
          && SchedulingParameter::parse(options_->get<std::string>("valid-freq"))) {
          for(auto validator : Validators(dataset->getVocabs(), options_))
            scheduler->addValidator(validator);
        }

        auto batchGenerator = New<CorpusBatchGenerator>(dataset, options_, stats);

        scheduler->registerTrainingObserver(batchGenerator);

        auto model = New<ModelWrapper>(options_, mpi);
        model->setScheduler(scheduler);
        model->setTypicalTrgBatchWords(batchGenerator->estimateTypicalTrgBatchWords()); // needed for dynamic MB scaling
        model->load();

        bool restored = !options_->get<bool>("no-restore-corpus")
                        && batchGenerator->restore(trainState);

        // We only want custom behavior once training starts.
        installCustomSignalHandlers();

        // -- main training loop
        scheduler->started();
        while(scheduler->keepGoing()) {
          if(!restored)
            batchGenerator->prepare();
          restored = false;

          // main training loop for one epoch
          for(auto batch : *batchGenerator) {
            if (!scheduler->keepGoing())
              break;
            model->update(batch);
          }

          if(scheduler->keepGoing())
            scheduler->increaseEpoch();
        }
        scheduler->finished();

        model->finalize(); // allow async to sync before final save   --@TODO: rename, or move into save()

        // Avoid saving the model twice if it has been loaded and training did not progress
        if(!trainState->loaded)
          model->save(true);

        // Signal success to a potential MPI runner
        model = nullptr;     // release any reference to MPI that model may hold
        scheduler = nullptr; // as above
        finalizeMPI(std::move(mpi));

      } catch(DivergenceException& e) { // handling divergent training if scheduler is configured 
        // to throw via --throw-on-divergence
        if(options_->get<bool>("fp16-fallback-to-fp32", false)) {
          auto precisions = options_->get<std::vector<std::string>>("precision");
          Type parameterType = typeFromString(precisions[0]);
          if(parameterType == Type::float16) {
            // we diverged, but we were apparently training with fp16 and fallback to fp32
            // is enabled. There is a chance we can rescue the training run by restarting
            // from the last checkpoint but using fp32 precision training.
            LOG(warn, "Training diverged, but --fp16-fallback-to-fp32 is enabled. "
                      "Attempting restart from the last checkpoint with fp32 precision.");

            // undo all options that would be set for fp16 training
            options_ = options_->with(
              "fp16", false,
              "precision", std::vector<std::string>({"float32", "float32"}),
              "cost-scaling", std::vector<std::string>({})
            );

            // this gets checked at final do-while condition
            restartTraining = true;
          } else {
            // We diverged and fallback is enabled, but we are already training with fp32, 
            // hence rethrow and let training die with error.
            LOG(warn, "Training diverged, rethrowing divergence exception");
            throw e;
          }
        } else {
          // We diverged and no fallback enabled, hence rethrow and let training die with error.
          LOG(warn, "Training diverged, rethrowing divergence exception");
          throw e;
        }
      }
    } while(restartTraining);
  }
};

template <class ModelWrapper>
void Train<ModelWrapper>::installCustomSignalHandlers(){
  const std::string sigTermAction = options_->get<std::string>("sigterm");
  if (sigTermAction == "save-and-exit") {
    LOG(debug, "Will save before exiting upon SIGTERM.");
    signal(SIGTERM, requestSaveAndExit);
  }
  else if (sigTermAction != "exit-immediately")
    ABORT("Unrecognized value '{}' for --sigterm", sigTermAction);
}

}  // namespace marian
