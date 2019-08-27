#include "common/config_parser.h"
#include "common/definitions.h"

namespace marian {

/**
 * Add all aliases
 *
 * An alias is a shortcut option for a predefined set of options. It is triggered if the option has
 * the requested value. The alias option has to be first defined using cli.add<T>(). Defining
 * multiple aliases for the same option name but with different value is allowed.
 *
 * Values are compared as std::string. If the alias option is a vector, the alias will be triggered
 * if the requested value exists in that vector at least once.
 *
 * @see CLIWrapper::alias()
 *
 * The order of alias definitions *does* matter: options from later aliases override earlier
 * regardless of its order in the command line or config file.
 */
void ConfigParser::addAliases(cli::CLIWrapper& cli) {
  cli.alias("fp16", "true", [](YAML::Node& config) {
    config["precision"] = std::vector<std::string>({"float16", "float32", "float32"});
    config["cost-scaling"] = std::vector<std::string>({"7", "2000", "2", "0.05", "10", "1"});
  });

  cli.alias("no-shuffle", "true", [](YAML::Node& config) {
    config["shuffle"] = "none";
  });

  // Options setting the BiDeep architecture proposed in http://www.aclweb.org/anthology/W17-4710
  cli.alias("best-deep", "true", [](YAML::Node& config) {
    // Model options
    config["layer-normalization"] = true;
    config["tied-embeddings"] = true;
    config["enc-type"] = "alternating";
    config["enc-cell-depth"] = 2;
    config["enc-depth"] = 4;
    config["dec-cell-base-depth"] = 4;
    config["dec-cell-high-depth"] = 2;
    config["dec-depth"] = 4;
    config["skip"] = true;

    // Training specific options
    config["learn-rate"] = 0.0003;
    config["cost-type"] = "ce-mean-words";
    config["lr-decay-inv-sqrt"] = 16000;
    config["label-smoothing"] = 0.1;
    config["clip-norm"] = 5;
    config["sync-sgd"] = true;
    config["exponential-smoothing"] = 1e-4;
    config["mini-batch-fit"] = true;
    config["mini-batch"] = 1000;
    config["maxi-batch"] = 1000;
    // config["workspace"] = 13000;
    // config["workspace"] = "max";
  });

  // Architecture and proposed training settings for a Transformer "base" model introduced in
  // https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf
  cli.alias("task", "transformer-base", [](YAML::Node& config) {
    // Model options
    config["type"] = "transformer";
    config["enc-depth"] = 6;
    config["dec-depth"] = 6;
    config["dim-emb"] = 512;
    config["tied-embeddings-all"] = true;
    config["transformer-dim-ffn"] = 2048;
    config["transformer-heads"] = 8;
    config["transformer-postprocess"] = "dan";
    config["transformer-preprocess"] = "d";
    config["transformer-ffn-activation"] = "relu";
    config["transformer-dropout"] = 0.1;

    // Training specific options
    config["learn-rate"] = 0.0003;
    config["cost-type"] = "ce-mean-words";
    config["lr-warmup"] = 16000;
    config["lr-decay-inv-sqrt"] = 16000;
    config["label-smoothing"] = 0.1;
    config["clip-norm"] = 5;
    config["sync-sgd"] = true;
    config["exponential-smoothing"] = 1e-4;
    config["max-length"] = 100;
    config["mini-batch-fit"] = true;
    config["mini-batch"] = 1000;
    config["maxi-batch"] = 1000;
    // config["workspace"] = 13000;
    // config["workspace"] = "max";
  });

  // Architecture and proposed training settings for a Transformer "big" model introduced in
  // https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf
  cli.alias("task", "transformer-big", [](YAML::Node& config) {
    // Model options
    config["type"] = "transformer";
    config["enc-depth"] = 6;
    config["dec-depth"] = 6;
    config["dim-emb"] = 1024;
    config["tied-embeddings-all"] = true;
    config["transformer-dim-ffn"] = 4096;
    config["transformer-heads"] = 16;
    config["transformer-postprocess"] = "dan";
    config["transformer-preprocess"] = "d";
    config["transformer-ffn-activation"] = "relu";
    config["transformer-dropout"] = 0.1;

    // Training specific options
    config["learn-rate"] = 0.0002;
    config["cost-type"] = "ce-mean-words";
    config["lr-warmup"] = 8000;
    config["lr-decay-inv-sqrt"] = 8000;
    config["label-smoothing"] = 0.1;
    config["clip-norm"] = 5;
    config["sync-sgd"] = true;
    config["exponential-smoothing"] = 1e-4;
    config["max-length"] = 100;
    config["mini-batch-fit"] = true;
    config["mini-batch"] = 1000;
    config["maxi-batch"] = 1000;
    // config["workspace"] = 13000;
    // config["workspace"] = "max";
  });

  cli.alias("task", "bert-base-pretrain", [](YAML::Node& config) {
    // Transformer encoder options
    config["type"] = "bert";
    config["enc-depth"] = 12;
    config["dec-depth"] = 12;
    config["dim-emb"] = 768;
    config["tied-embeddings-all"] = true;
    config["transformer-dim-ffn"] = 3072;
    config["transformer-heads"] = 12;
    config["transformer-postprocess"] = "dan";
    config["transformer-preprocess"] = "nd";
    config["transformer-ffn-activation"] = "gelu";
    config["transformer-dropout"] = 0.1;
    config["transformer-dropout-attention"] = 0.1;
    config["transformer-train-position-embeddings"] = true;

    // BERT specific options - Masked LM
    config["bert-mask-symbol"] = "[MASK]";
    config["bert-sep-symbol"] = "[SEP]";
    config["bert-class-symbol"] = "[CLS]";
    config["bert-masking-fraction"] = 0.15f;
    config["input-types"] = std::vector<std::string>({"sequence", "class"});

    // BERT specific options - Next sentence classifier
    config["bert-train-type-embeddings"] = true;
    config["bert-type-vocab-size"] = 2;

    // Training specific options
    config["learn-rate"] = 0.0002;
    config["cost-type"] = "ce-mean-words";
    config["multi-loss-type"] = "scaled";
    config["lr-warmup"] = 16000;
    config["lr-decay-inv-sqrt"] = 16000;
    config["clip-norm"] = 5;
    config["sync-sgd"] = true;
    config["exponential-smoothing"] = 1e-4;
    config["max-length"] = 512;
    config["mini-batch-fit"] = true;
    config["mini-batch"] = 1000;
    config["maxi-batch"] = 1000;
    config["maxi-batch-sort"] = "src";
    config["workspace"] = 13000;

    // // Validation settings
    // config["valid-heldout"] = 5000;
    // config["valid-metrics"] = std::vector<std::string>({
    //   "ce-mean-words",
    //   "bert-lm-accuracy",
    //   "bert-sentence-accuracy"
    // });
    // config["valid-freq"] = 5000;
    // config["save-freq"] = 5000;
    // config["disp-freq"] = 500;
  });

  // cli.alias("task", "bert-base-finetune-classifier", [](YAML::Node& config) {
  //   // Model options
  //   config["type"] = "bert";
  //   config["enc-depth"] = 12;
  //   config["dec-depth"] = 12;
  //   config["dim-emb"] = 768;
  //   config["transformer-dim-ffn"] = 3072;
  //   config["transformer-heads"] = 12;
  //   config["transformer-postprocess"] = "dan";
  //   config["transformer-preprocess"] = "nd";
  //   config["transformer-ffn-activation"] = "gelu";
  //   config["transformer-dropout"] = 0.1;
  //   config["transformer-dropout-attention"] = 0.1;
  //   //config["transformer-dropout-ffn"] = 0.1;

  //   // Training specific options
  //   config["learn-rate"] = 0.0002;
  //   config["cost-type"] = "ce-mean-words";
  //   config["lr-warmup"] = 16000;
  //   config["lr-decay-inv-sqrt"] = 16000;
  //   config["clip-norm"] = 5;
  //   config["sync-sgd"] = true;
  //   config["exponential-smoothing"] = 1e-4;
  // });
}

}  // namespace marian
