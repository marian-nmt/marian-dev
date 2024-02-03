#include "marian.h"

#include "common/timer.h"
#include "embedder/embedder.h"
#include "models/model_task.h"


using namespace marian;

namespace pymarian {
 class PyEmbedder {
    private:
        Ptr<marian::Options> options_;
        Ptr<Embed<Embedder>> embedder_;
    public:
        PyEmbedder(const std::string& cliString) {
            options_ = parseOptions(cliString, cli::mode::embedding, true);
            embedder_ = New<Embed<Embedder>>(options_);
        }

        int embed() {
            //TODO: add options_ override from args to embed()
            //TODO:  read input from args instead of STDIN
            embedder_->run();
            return 0;
        }
    };

} // namespace pymarian