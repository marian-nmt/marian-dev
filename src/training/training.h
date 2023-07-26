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

    // We run training in a do-while loop. It should only restart if a training run was interrupted
    // via the throwing of a DivergenceException from training/scheduler.h and if --throw-on-divergence and
    // custom-fallbacks are specified (directly or the via alias fp16-fallback-to-fp32) otherwise it will die with the rethrown exception. 
    // The repeated training run will continue from the last checkpoint (similar to a manually interrupted training) 
    // but attempt training with the options specified in the current fallback. If that training run in turn happens to diverge, 
    // training will move on to the next defined fallback or exit with an unhandled DivergenceException if there are no more fallbacks. 
    // The unhandled exception is on purpose to indicate a fatal error.

    auto originalOptions = options_->clone(); // clone in order to keep unaltered option object around
    bool restartTraining;      // record if training should be restarted after catching a DivergenceException
    size_t restartCounter = 0; // count how many restarts occured. Used to progress through the list of fallbacks

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

        // get the list of possible fallback set of options
        auto fallbacks = options_->get<std::vector<YAML::Node>>("custom-fallbacks", {});

        // check if we exceeded the number of available fallbacks, if not, take the current one
        if(restartCounter < fallbacks.size()) {
            auto fallback = fallbacks[restartCounter];
            fallback.SetStyle(YAML::EmitterStyle::Flow);

            // we diverged, but a set of fallback options is specified. There is a chance we can rescue the training run by 
            // restarting from the last checkpoint with the options from the current fallback.
            LOG(warn, "Training diverged, but fallback is enabled. Attempting restart from the last checkpoint with these options: {}", YAML::Dump(fallback));

            // overwrite all original options with fallback options
            options_ = originalOptions->with(fallback);

            // this gets checked at final do-while condition
            restartTraining = true;
            restartCounter++;
        } else {
          // we diverged and no fallback is available, hence rethrow and let training die with error.
          LOG(warn, "Training diverged and there are either no fallbacks or we exceeded the number of defined fallbacks, rethrowing divergence exception");
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
