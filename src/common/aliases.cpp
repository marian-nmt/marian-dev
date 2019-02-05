#include "common/config_parser.h"

namespace marian {

void ConfigParser::addAliases(cli::CLIWrapper& cli) {
  // The order of aliases does matter as later options overwrite earlier

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
    config["workspace"] = 13000;
    // config["workspace"] = "max";
  });

  cli.alias("task", "transformer-base", [](YAML::Node& config) {
    // Model options
    config["type"] = "transformer";
    config["enc-depth"] = 6;
    config["dec-depth"] = 6;
    config["dim-emb"] = 512;
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
    config["mini-batch-fit"] = true;
    config["mini-batch"] = 1000;
    config["maxi-batch"] = 1000;
    config["workspace"] = 13000;
    // config["workspace"] = "max";
  });

  cli.alias("task", "transformer-big", [](YAML::Node& config) {
    // Model options
    config["type"] = "transformer";
    config["enc-depth"] = 6;
    config["dec-depth"] = 6;
    config["dim-emb"] = 1024;
    config["transformer-dim-ffn"] = 4096;
    config["transformer-heads"] = 16;
    config["transformer-postprocess"] = "dan";
    config["transformer-preprocess"] = "d";
    config["transformer-ffn-activation"] = "relu";
    config["transformer-dropout"] = 0.1;
    config["transformer-dropout-attention"] = 0.1;
    config["transformer-dropout-ffn"] = 0.1;

    // Training specific options
    config["learn-rate"] = 0.0002;
    config["cost-type"] = "ce-mean-words";
    config["lr-warmup"] = 8000;
    config["lr-decay-inv-sqrt"] = 8000;
    config["label-smoothing"] = 0.1;
    config["clip-norm"] = 5;
    config["sync-sgd"] = true;
    config["exponential-smoothing"] = 1e-4;
    config["mini-batch-fit"] = true;
    config["mini-batch"] = 1000;
    config["maxi-batch"] = 1000;
    config["workspace"] = 13000;
    // config["workspace"] = "max";
  });

  cli.alias("task", "bert-base-pretrain", [](YAML::Node& config) {
    // Transformer encoder options
    config["type"] = "bert";
    config["enc-depth"] = 12;
    config["dec-depth"] = 12;
    config["dim-emb"] = 768;
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

    // BERT specific options - Next sentence classifier
    config["bert-train-type-embeddings"] = true;
    config["bert-type-vocab-size"] = 2;

    //config["transformer-dropout-ffn"] = 0.1;

    // Training specific options
    config["learn-rate"] = 0.0002;
    config["cost-type"] = "ce-mean-words";
    config["multi-loss-type"] = "scaled";
    config["lr-warmup"] = 16000;
    config["lr-decay-inv-sqrt"] = 16000;
    config["clip-norm"] = 5;
    config["sync-sgd"] = true;
    config["exponential-smoothing"] = 1e-4;
    config["mini-batch-fit"] = true;
    config["mini-batch"] = 1000;
    config["maxi-batch"] = 1000;
    config["workspace"] = 13000;

    // // Validation settings
    // config["valid-heldout"] = 5000;
    // config["valid-metrics"] = std::vector<std::string>({
    //   "ce-mean-words",
    //   "bert-masked-lm-accuracy",
    //   "bert-next-sentence-accuracy"
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

}