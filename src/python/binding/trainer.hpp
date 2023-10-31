#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include <signal.h>
#include "marian.h"

#include "common/signal_handling.h"
#include "training/graph_group_async.h"
#include "training/graph_group_singleton.h"
#include "training/graph_group_sync.h"
#include "training/training.h"

#include "3rd_party/ExceptionWithCallStack.h"


namespace py = pybind11;
using namespace marian;


namespace pymarian {


    class PyTrainer {

    private:
        Ptr<marian::Options> options_;
        Ptr<Train<SyncGraphGroup>> trainer_;

    public:
        PyTrainer(const std::string& cliString){
            options_ = parseOptions(cliString, cli::mode::training, true);
            LOG(info, "Using synchronous SGD");
            trainer_ = New<Train<SyncGraphGroup>>(options_);
        }

        int train() {
            //TODO: add options_ override from args to train()
            //TODO:  read input from args instead of STDIN

            trainer_->run();
            // If we exit due to a graceful exit request via SIGTERM, exit with 128 + SIGTERM,
            // as suggested for bash in http://tldp.org/LDP/abs/html/exitcodes.html. This allows parent
            // scripts to determine if training terminated naturally or via SIGTERM.
            // An alternative would be to exit with code 124, which is what the timeout command
            // returns for timeout -s SIGTERM <seconds> ...., because exiting after SIGTERM
            // is not technically a fatal error (which is what the 128+x convention usually
            // stands for).
            exit(getSignalFlag(SIGTERM) ? 128 + SIGTERM : EXIT_SUCCESS);
        }
    };

}