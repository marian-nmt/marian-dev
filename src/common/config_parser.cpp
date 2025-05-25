#include "common/authors.h"
#include "common/build_info.h"
#include "common/cli_helper.h"
#include "common/config.h"
#include "common/config_parser.h"
#include "common/config_validator.h"
#include "common/definitions.h"
#include "common/file_stream.h"
#include "common/logging.h"
#include "common/options.h"
#include "common/regex.h"
#include "common/utils.h"
#include <algorithm>
#include <set>
#include <stdexcept>
#include <string>

#if MKL_FOUND
#include <mkl.h>
#else
#if BLAS_FOUND
#include <cblas.h>
#endif
#endif

#if USE_SSL
#include "common/crypt.h"
#endif

namespace marian {

// TODO: Move this to CLIWrapper and allow to mark options as paths in the same place they are
// defined
// clang-format off
const std::set<std::string> PATHS = {
  "model",
  "models",
  "train-sets",
  "vocabs",
  "embedding-vectors",
  "valid-sets",
  "valid-script-path",
  "valid-script-args",
  "valid-log",
  "valid-translation-output",
  "input",            // except: 'stdin', handled in makeAbsolutePaths and interpolateEnvVars
  "output",           // except: 'stdout', handled in makeAbsolutePaths and interpolateEnvVars
  "pretrained-model",
  "data-weighting",
  "log",
  "sqlite",           // except: 'temporary', handled in the processPaths function
  "shortlist",        // except: only the first element in the sequence is a path, handled in the
                      //  processPaths function
};
// clang-format on

std::string escapeCmdLine(int argc, char** argv){
  std::string cmdLine;
  for(int i = 0; i < argc; i++) {
    std::string arg = argv[i];
    std::string quote; // attempt to quote special chars
    if(arg.empty() || arg.find_first_of(" #`\"'\\${}|&^?*!()%><") != std::string::npos)
      quote = "'";
    arg = regex::regex_replace(arg, regex::regex("'"), "'\\''");
    if(!cmdLine.empty())
      cmdLine.push_back(' ');
    cmdLine += quote + arg + quote;
  }
  return cmdLine;
}

std::string const& ConfigParser::cmdLine() const {
  return cmdLine_;
}

ConfigParser::ConfigParser(cli::mode mode)
  : cli_(config_,"Marian: Fast Neural Machine Translation in C++",
         "General options", "", 40),
    mode_(mode == cli::mode::server ? cli::mode::translation : mode) {

  addOptionsGeneral(cli_);
  if (mode == cli::mode::server)
    addOptionsServer(cli_);
  addOptionsModel(cli_);

  // clang-format off
  switch(mode_) {
    case cli::mode::training:
      addOptionsTraining(cli_);
      addOptionsValidation(cli_);
      break;
    case cli::mode::translation:
      addOptionsTranslation(cli_);
      break;
    case cli::mode::scoring:
      addOptionsScoring(cli_);
      break;
    case cli::mode::embedding:
      addOptionsEmbedding(cli_);
      break;
    case cli::mode::evaluating:
      addOptionsEvaluating(cli_);
      break;
    default:
      ABORT("wrong CLI mode");
      break;
  }

  addAliases(cli_);
  // clang-format on
}

void ConfigParser::addOptionsGeneral(cli::CLIWrapper& cli) {
  int defaultWorkspace = (mode_ == cli::mode::translation) ? 512 : 2048;

  cli.switchGroup("General options");

  // clang-format off
  cli.add<bool>("--authors",
    "Print list of authors and exit");
  cli.add<bool>("--cite",
    "Print citation and exit");
  cli.add<std::string>("--build-info",
    "Print CMake build options and exit. Set to 'all' to print advanced options")
    ->implicit_val("basic");
  cli.add<std::vector<std::string>>("--config,-c",
    "Configuration file(s). If multiple, later overrides earlier");
  cli.add<int>("--workspace,-w",
    "Preallocate arg MB of work space. Negative `--workspace -N` value allocates workspace as total available GPU memory minus N megabytes.",
    defaultWorkspace);
  cli.add<std::string>("--log",
    "Log training process information to file given by arg");
  cli.add<std::string>("--log-level",
    "Set verbosity level of logging: trace, debug, info, warn, err(or), critical, off",
    "info");
  cli.add<std::string>("--log-time-zone",
    "Set time zone for the date shown on logging");
  cli.add<bool>("--quiet",
    "Suppress all logging to stderr. Logging to files still works");
  cli.add<bool>("--quiet-translation",
    "Suppress logging for translation");
  cli.add<size_t>("--seed",
    "Seed for all random number generators. 0 means initialize randomly");
  cli.add<bool>("--check-nan",
    "Check for NaNs or Infs in forward and backward pass. Will abort when found. "
    "This is a diagnostic option that will slow down computation significantly");
  cli.add<bool>("--interpolate-env-vars",
    "allow the use of environment variables in paths, of the form ${VAR_NAME}");
  cli.add<bool>("--relative-paths",
    "All paths are relative to the config file location");
  cli.add<std::string>("--dump-config",
    "Dump current (modified) configuration to stdout and exit. Possible values: full, minimal, expand")
    ->implicit_val("full");
  if(mode_ == cli::mode::training) {
    // --sigterm is deliberately not a boolean, to allow for a consistent
    // pattern of specifying custom signal handling in the future.
    // (e.g., dump model but continue training upon SIGUSR1, or report current
    // training status upon SIGINFO.)
    cli.add<std::string>("--sigterm",
      "What to do with SIGTERM: save-and-exit or exit-immediately.",
      "save-and-exit");
  }
  // clang-format on
}

void ConfigParser::addOptionsServer(cli::CLIWrapper& cli) {
  // clang-format off
  auto previous_group = cli.switchGroup("Server options");
  cli.add<size_t>("--port,-p",
      "Port number for web socket server",
      8080);
  cli.switchGroup(previous_group);
  // clang-format on
}

void ConfigParser::addOptionsModel(cli::CLIWrapper& cli) {
  auto previous_group = cli.switchGroup("Model options");

  // clang-format off
  if(mode_ == cli::mode::translation) {
    cli.add<std::vector<std::string>>("--models,-m",
      "Paths to model(s) to be loaded. Supported file extensions: .npz, .bin");
  } else {
    cli.add<std::string>("--model,-m",
      "Path prefix for model to be saved/resumed. Supported file extensions: .npz, .bin",
      "model.npz");

    if(mode_ == cli::mode::training) {
      cli.add<std::string>("--pretrained-model",
        "Path prefix for pre-trained model to initialize model weights");
    }
  }
#ifdef COMPILE_CPU
  if(mode_ == cli::mode::translation) {
    cli.add<bool>("--model-mmap",
      "Use memory-mapping when loading model (CPU only)");
  }
#endif
  cli.add<bool>("--ignore-model-config",
      "Ignore the model configuration saved in npz file");
  cli.add<std::string>("--type",
      "Model type: amun, nematus, s2s, multi-s2s, transformer",
      "amun");
  cli.add<std::vector<int>>("--dim-vocabs",
      "Maximum items in vocabulary ordered by rank, 0 uses all items in the provided/created vocabulary file",
      {0, 0});
  cli.add<int>("--dim-emb",
      "Size of embedding vector",
      512);
  cli.add<int>("--factors-dim-emb",
      "Embedding dimension of the factors. Only used if concat is selected as factors combining form");
  cli.add<std::string>("--factors-combine",
    "How to combine the factors and lemma embeddings. Options available: sum, concat",
    "sum");
  cli.add<std::string>("--lemma-dependency",
      "Lemma dependency method to use when predicting target factors. Options: soft-transformer-layer, hard-transformer-layer, lemma-dependent-bias, re-embedding");
  cli.add<int>("--lemma-dim-emb",
      "Re-embedding dimension of lemma in factors",
      0);
  cli.add<int>("--dim-rnn",
      "Size of rnn hidden state", 1024);
  cli.add<std::string>("--enc-type",
      "Type of encoder RNN : bidirectional, bi-unidirectional, alternating (s2s)",
      "bidirectional");
  cli.add<std::string>("--enc-cell",
      "Type of RNN cell: gru, lstm, tanh (s2s)", "gru");
  cli.add<int>("--enc-cell-depth",
      "Number of transitional cells in encoder layers (s2s)",
      1);
  cli.add<int>("--enc-depth",
      "Number of encoder layers (s2s)",
      1);
  cli.add<std::string>("--dec-cell",
      "Type of RNN cell: gru, lstm, tanh (s2s)",
      "gru");
  cli.add<int>("--dec-cell-base-depth",
      "Number of transitional cells in first decoder layer (s2s)",
      2);
  cli.add<int>("--dec-cell-high-depth",
      "Number of transitional cells in next decoder layers (s2s)",
      1);
  cli.add<int>("--dec-depth",
      "Number of decoder layers (s2s)",
      1);
  cli.add<bool>("--skip",
      "Use skip connections (s2s)");
  cli.add<bool>("--layer-normalization",
      "Enable layer normalization");
  cli.add<bool>("--right-left",
      "Train right-to-left model");
  cli.add<std::vector<std::string>>("--input-types",
      "Provide type of input data if different than 'sequence'. "
      "Possible values: sequence, class, alignment, weight. "
      "You need to provide one type per input file (if --train-sets) or per TSV field (if --tsv).",
      {});
  cli.add<std::vector<size_t>>("--input-reorder",
      "Reorder input data to this order according to this permutation. If empty no reordering is done. "
      "If non-empty, you need to provide one type per input file (if --train-sets) or per TSV field (if --tsv). "
      "Usually, there should be no need to provide these on the command line, the model should have them saved.",
      {});
  cli.add<bool>("--input-join-fields",
      "Join input fields (from files or TSV) into a single sequence "
      "(mostly used single-encoder models like BLEURT and COMET-KIWI)",
      false);
  cli.add<bool>("--best-deep",
      "Use Edinburgh deep RNN configuration (s2s)");
  cli.add<bool>("--tied-embeddings",
      "Tie target embeddings and output embeddings in output layer");
  cli.add<bool>("--tied-embeddings-src",
      "Tie source and target embeddings");
  cli.add<bool>("--tied-embeddings-all",
      "Tie all embedding layers and output layer");
  cli.add<bool>("--output-omit-bias",
      "Do not use a bias vector in decoder output layer");

  // Transformer options
  cli.add<int>("--transformer-heads",
      "Number of heads in multi-head attention (transformer)",
      8);
  cli.add<bool>("--transformer-no-projection",
      "Omit linear projection after multi-head attention (transformer)");
  cli.add<bool>("--transformer-rnn-projection",
      "Add linear projection after rnn layer (transformer)");
  cli.add<bool>("--transformer-pool",
      "Pool encoder states instead of using cross attention (selects first encoder state, best used with special token)");
  cli.add<int>("--transformer-dim-ffn",
      "Size of position-wise feed-forward network (transformer)",
      2048);
  cli.add<int>("--transformer-decoder-dim-ffn",
      "Size of position-wise feed-forward network in decoder (transformer). Uses --transformer-dim-ffn if 0.",
      0);
  cli.add<int>("--transformer-ffn-depth",
      "Depth of filters (transformer)",
      2);
  cli.add<int>("--transformer-decoder-ffn-depth",
      "Depth of filters in decoder (transformer). Uses --transformer-ffn-depth if 0",
      0);
  cli.add<std::string>("--transformer-ffn-activation",
      "Activation between filters: swish or relu (transformer)",
      "swish");
  cli.add<int>("--transformer-dim-aan",
      "Size of position-wise feed-forward network in AAN (transformer)",
      2048);
  cli.add<int>("--transformer-aan-depth",
      "Depth of filter for AAN (transformer)",
      2);
  cli.add<std::string>("--transformer-aan-activation",
      "Activation between filters in AAN: swish or relu (transformer)",
      "swish");
  cli.add<bool>("--transformer-aan-nogate",
      "Omit gate in AAN (transformer)");
  cli.add<std::string>("--transformer-decoder-autoreg",
      "Type of autoregressive layer in transformer decoder: self-attention, average-attention (transformer)",
      "self-attention");
  cli.add<std::vector<size_t>>("--transformer-tied-layers",
      "List of tied decoder layers (transformer)");
  cli.add<std::string>("--transformer-guided-alignment-layer",
      "Last or number of layer to use for guided alignment training in transformer",
      "last");
  cli.add<std::string>("--transformer-preprocess",
      "Operation before each transformer layer: d = dropout, a = add, n = normalize");
  cli.add<std::string>("--transformer-postprocess-emb",
      "Operation after transformer embedding layer: d = dropout, a = add, n = normalize",
      "d");
  cli.add<std::string>("--transformer-postprocess",
      "Operation after each transformer layer: d = dropout, a = add, n = normalize",
      "dan");
  cli.add<std::string>("--transformer-postprocess-top",
      "Final operation after a full transformer stack: d = dropout, a = add, n = normalize. The optional skip connection with 'a' by-passes the entire stack.",
      "");
  cli.add<bool>("--transformer-train-position-embeddings",
      "Train positional embeddings instead of using static sinusoidal embeddings");
  cli.add<bool>("--transformer-depth-scaling",
      "Scale down weight initialization in transformer layers by 1 / sqrt(depth)");

  cli.add<std::string>("--transformer-attention-mask",
      "Type of mask/bias in transformer attention: default, alibi",
      "default");
  cli.add<bool>("--transformer-alibi-shift",
      "Use alibi-shifting with sync-points with --transformer-attention-mask alibi");
  cli.add<std::string>("--separator-symbol",
      "Generic separator symbol for different applications, i.e. for transformer-alibi-shift syncpoints, default is [eos] (currently only supported with raw spm models)",
      "[eos]");
  cli.add<bool>("--transformer-disable-position-embeddings",
      "Do not add any position embeddings. Use e.g. with --transformer-attention-mask alibi");

  cli.add<bool>("--transformer-alibi-trainable",
      "Make alibi slopes trainable, default slopes are constant");

  // handy shortcut for the current best setup
  cli.add<bool>("--alibi",
      "Use alibi settings for transformer, this is a shortcut for --transformer-attention-mask alibi --transformer-alibi-shift --transformer-disable-position-embeddings --separator-symbol [eos]");
  cli.alias("alibi", "true", [](YAML::Node& config) {
    // define current-best alibi settings
    config["transformer-attention-mask"] = "alibi";
    config["transformer-alibi-shift"] = true;
    config["transformer-disable-position-embeddings"] = true;
    config["separator-symbol"] = "[eos]";
    config["transformer-alibi-trainable"] = true;
  });

  cli.add<bool>("--transformer-no-bias",
      "Don't use any bias vectors in linear layers");
  cli.add<bool>("--transformer-no-affine",
      "Don't use any scale or bias vectors in layer norm");

  cli.add<std::string>("--bert-mask-symbol", "Masking symbol for BERT masked-LM training", "[MASK]");
  cli.add<std::string>("--bert-sep-symbol", "Sentence separator symbol for BERT next sentence prediction training", "[SEP]");
  cli.add<std::string>("--bert-class-symbol", "Class symbol BERT classifier training", "[CLS]");
  cli.add<float>("--bert-masking-fraction", "Fraction of masked out tokens during training", 0.15f);
  cli.add<bool>("--bert-train-type-embeddings", "Train bert type embeddings, set to false to use static sinusoidal embeddings", true);
  cli.add<int>("--bert-type-vocab-size", "Size of BERT type vocab (sentence A and B)", 2);

  // Options specific for the "comet-qe" model type
  cli.add<bool>("--comet-final-sigmoid", "Add final sigmoid to COMET model");
  cli.add<bool>("--comet-stop-grad", "Do not propagate gradients through COMET model");

  cli.add<bool>("--comet-mix", "Mix encoder layers to produce embedding");
  cli.add<bool>("--comet-mix-norm", "Normalize layers prior to mixing");
  cli.add<std::string>("--comet-pool", "Pooling operation over time dimension (avg, cls, max)", "avg");
  cli.add<std::string>("--comet-mix-transformation", "Which transformation to apply to layer mixing (softmax [default] or sparsemax)", "softmax");
  cli.add<float>("--comet-dropout", "Dropout for pooler layers", 0.1f);
  cli.add<float>("--comet-mixup", "Alpha parameter for Beta distribution for mixup", 0.0f);
  cli.add<bool>("--comet-mixup-reg", "Use original and mixed-up samples in training");
  cli.add<float>("--comet-augment-bad", "Fraction of bad examples added via shuffling for class/label 0.f", 0.0f);
  cli.add<std::vector<int>>("--comet-pooler-ffn", "Hidden sizes for comet pooler", {2048, 1024});
  cli.add<bool>("--comet-prepend-zero", "Add a start symbol to batch entries");
  cli.add<bool>("--comet-use-separator", "Add a sentence separator to batch entries when joining source, target and mt", false);

#ifdef CUDNN
  cli.add<int>("--char-stride",
      "Width of max-pooling layer after convolution layer in char-s2s model",
      5);
  cli.add<int>("--char-highway",
      "Number of highway network layers after max-pooling in char-s2s model",
      4);
  cli.add<std::vector<int>>("--char-conv-filters-num",
      "Numbers of convolution filters of corresponding width in char-s2s model",
      {200, 200, 250, 250, 300, 300, 300, 300});
  cli.add<std::vector<int>>("--char-conv-filters-widths",
      "Convolution window widths in char-s2s model",
      {1, 2, 3, 4, 5, 6, 7, 8});
#endif

  if(mode_ == cli::mode::training) {
    // TODO: add ->range(0,1);
    cli.add<float>("--dropout-rnn",
        "Scaling dropout along rnn layers and time (0 = no dropout)");
    cli.add<float>("--dropout-src",
        "Dropout source words (0 = no dropout)");
    cli.add<float>("--dropout-trg",
        "Dropout target words (0 = no dropout)");
    cli.add<float>("--transformer-dropout",
        "Dropout between transformer layers (0 = no dropout)");
    cli.add<float>("--transformer-dropout-attention",
        "Dropout for transformer attention (0 = no dropout)");
    cli.add<float>("--transformer-dropout-ffn",
        "Dropout for transformer filter (0 = no dropout)");
  }
  cli.switchGroup(previous_group);
  // clang-format on
}

void ConfigParser::addOptionsTraining(cli::CLIWrapper& cli) {
  auto previous_group = cli.switchGroup("Training options");
  // clang-format off
  cli.add<std::string>("--cost-type", // @TODO: rename to loss-type
      "Optimization criterion: ce-mean, ce-mean-words, ce-sum, perplexity", "ce-sum");
  cli.add<std::string>("--multi-loss-type",
      "How to accumulate multi-objective losses: sum, scaled, mean", "sum");
  cli.add<bool>("--unlikelihood-loss",
      "Use word-level weights as indicators for sequence-level unlikelihood training");
  cli.add<bool>("--overwrite",
      "Do not create model checkpoints, only overwrite main model file with last checkpoint. "
      "Reduces disk usage");
  cli.add<bool>("--overwrite-checkpoint",
      "When --overwrite=false (default) only model files get written at saving intervals (with iterations numbers). "
      "Setting --overwrite-checkpoint=false also saves full checkpoints checkpoints with optimizer parameters, etc. "
      "Uses (a lot) more disk space.",
      true);
  cli.add<bool>("--no-reload",
      "Do not load existing model specified in --model arg");
  cli.add<bool>("--no-optimizer-reload",
      "Do not load existing optimizer state from checkpoint specified in --model arg");
  cli.add<std::vector<std::string>>("--train-sets,-t",
      "Paths to training corpora: source target");
  cli.add<std::vector<std::string>>("--vocabs,-v",
      "Paths to vocabulary files have to correspond to --train-sets. "
      "If this parameter is not supplied we look for vocabulary files "
      "source.{yml,json} and target.{yml,json}. "
      "If these files do not exist they are created");
#ifdef USE_SENTENCEPIECE
  cli.add<std::vector<float>>("--sentencepiece-alphas",
      "Sampling factors for SentencePiece vocabulary; i-th factor corresponds to i-th vocabulary");
  cli.add<std::string>("--sentencepiece-options",
      "Pass-through command-line options to SentencePiece trainer");
  cli.add<size_t>("--sentencepiece-max-lines",
      "Maximum lines to train SentencePiece vocabulary, selected with sampling from all data. "
      "When set to 0 all lines are going to be used.",
      2000000);
  cli.add<bool>("--no-spm-encode",
      "Assume the input has already had sentencepiece applied before decoding. "
      "Expects spm pieces, like the ones produced by spm_encode's default format.");
#endif
  // scheduling options

  // @TODO: these should be re-defined as aliases for `--after` but the current frame work matches on value, so not doable.
  cli.add<size_t>("--after-epochs,-e",
      "Finish after this many epochs, 0 is infinity (deprecated, '--after-epochs N' corresponds to '--after Ne')"); // @TODO: replace with alias
  cli.add<size_t>("--after-batches",
      "Finish after this many batch updates, 0 is infinity (deprecated, '--after-batches N' corresponds to '--after Nu')"); // @TODO: replace with alias

  cli.add<std::string>("--after,-a",
      "Finish after this many chosen training units, 0 is infinity (e.g. 100e = 100 epochs, 10Gt = 10 billion target labels, 100Ku = 100,000 updates",
      "0e");
  cli.add<std::string/*SchedulerPeriod*/>("--disp-freq",
      "Display information every arg updates (append 't' for every arg target labels)",
      "1000u");
  cli.add<size_t>("--disp-first",
      "Display information for the first arg updates");
  cli.add<bool>("--disp-label-counts",
      "Display label counts when logging loss progress",
      true);
//   cli.add<int>("--disp-label-index",
//       "Display label counts based on i-th input stream (-1 is last)", -1);
  cli.add<std::string/*SchedulerPeriod*/>("--save-freq",
      "Save model file every arg updates (append 't' for every arg target labels)",
      "10000u");
  cli.add<std::vector<std::string>>("--logical-epoch",
      "Redefine logical epoch counter as multiple of data epochs (e.g. 1e), updates (e.g. 100Ku) or labels (e.g. 1Gt). "
      "Second parameter defines width of fractional display, 0 by default.",
      {"1e", "0"});

  addSuboptionsInputLength(cli);
  addSuboptionsTSV(cli);

  // data management options
  cli.add<std::string>("--shuffle",
      "How to shuffle input data (data: shuffles data and sorted batches; batches: "
      "data is read in order into batches, but batches are shuffled; none: no shuffling). "
      "Use with '--maxi-batch-sort none' in order to achieve exact reading order", "data");
  cli.add<bool>("--no-shuffle",
      "Shortcut for backwards compatiblity, equivalent to --shuffle none (deprecated)");
  cli.add<bool>("--no-restore-corpus",
      "Skip restoring corpus state after training is restarted");
  cli.add<std::string>("--tempdir,-T",
      "Directory for temporary (shuffled) files and database",
      "/tmp");
  cli.add<std::string>("--sqlite",
      "Use disk-based sqlite3 database for training corpus storage, default"
      " is temporary with path creates persistent storage")
    ->implicit_val("temporary");
  cli.add<bool>("--sqlite-drop",
      "Drop existing tables in sqlite3 database");

  addSuboptionsDevices(cli);
  addSuboptionsBatching(cli);

  // optimizer options
  cli.add<std::string>("--optimizer,-o",
     "Optimization algorithm: sgd, adagrad, adam",
     "adam");
  cli.add<std::vector<float>>("--optimizer-params",
     "Parameters for optimization algorithm, e.g. betas for Adam. "
     "Auto-adjusted to --mini-batch-words-ref if given");
  cli.add<float>("--optimizer-delay",
     "SGD update delay (#batches between updates). 1 = no delay. "
     "Can be fractional, e.g. 0.1 to use only 10% of each batch",
     1.f);

  cli.add<bool>("--sync-sgd",
     "Use synchronous SGD instead of asynchronous for multi-gpu training");

  // learning rate options
  cli.add<float>("--learn-rate,-l",
     "Learning rate. "
      "Auto-adjusted to --mini-batch-words-ref if given",
     0.0001f);
  cli.add<bool>("--lr-report",
     "Report learning rate for each update");

  cli.add<float>("--lr-decay",
     "Per-update decay factor for learning rate: lr <- lr * arg (0 to disable)");
  cli.add<std::string>("--lr-decay-strategy",
     "Strategy for learning rate decaying: epoch, batches, stalled, epoch+batches, epoch+stalled",
     "epoch+stalled");
  cli.add<std::vector<size_t>>("--lr-decay-start",
     "The first number of (epoch, batches, stalled) validations to start learning rate decaying (tuple)",
     {10, 1});
  cli.add<size_t>("--lr-decay-freq",
     "Learning rate decaying frequency for batches, requires --lr-decay-strategy to be batches",
     50000);
  cli.add<bool>("--lr-decay-reset-optimizer",
      "Reset running statistics of optimizer whenever learning rate decays");
  cli.add<bool>("--lr-decay-repeat-warmup",
     "Repeat learning rate warmup when learning rate is decayed");
  cli.add<std::vector<std::string/*SchedulerPeriod*/>>("--lr-decay-inv-sqrt",
     "Decrease learning rate at arg / sqrt(no. batches) starting at arg (append 't' or 'e' for sqrt(target labels or epochs)). "
     "Add second argument to define the starting point (default: same as first value)",
     {"0"});

  cli.add<std::string/*SchedulerPeriod*/>("--lr-warmup",
     "Increase learning rate linearly for arg first batches (append 't' for arg first target labels)",
      "0");
  cli.add<float>("--lr-warmup-start-rate",
     "Start value for learning rate warmup");
  cli.add<bool>("--lr-warmup-cycle",
     "Apply cyclic warmup");
  cli.add<bool>("--lr-warmup-at-reload",
     "Repeat warmup after interrupted training");

  cli.add<double>("--label-smoothing",
     "Epsilon for label smoothing (0 to disable)");
  cli.add<double>("--factor-weight",
     "Weight for loss function for factors (factored vocab only) (1 to disable)", 1.0f);
  cli.add<float>("--clip-norm",
     "Clip gradient norm to arg (0 to disable)",
     1.f); // @TODO: this is currently wrong with ce-sum and should rather be disabled or fixed by multiplying with labels
  cli.add<float>("--exponential-smoothing",
     "Maintain smoothed version of parameters for validation and saving with smoothing factor. 0 to disable. "
      "Auto-adjusted to --mini-batch-words-ref if given.",
     0.f)->implicit_val("1e-4");
  cli.add<std::string/*SchedulerPeriod*/>("--exponential-smoothing-replace-freq",
      "When exponential-smoothing is enabled replace master parameters with smoothed parameters once every n steps (possible units u=updates, t=target labels, e=epochs)",
      "0");
  cli.add<std::string>("--guided-alignment",
     "Path to a file with word alignments. Use guided alignment to guide attention or 'none'. "
     "If --tsv it specifies the index of a TSV field that contains the alignments (0-based)",
     "none");
  cli.add<std::string>("--guided-alignment-cost",
     "Cost type for guided alignment: ce (cross-entropy), mse (mean square error), mult (multiplication)",
     "ce");
  cli.add<double>("--guided-alignment-weight",
     "Weight for guided alignment cost",
     0.1);
  cli.add<std::string>("--data-weighting",
     "Path to a file with sentence or word weights. "
     "If --tsv it specifies the index of a TSV field that contains the weights (0-based)");
  cli.add<std::string>("--data-weighting-type",
     "Processing level for data weighting: sentence, word",
     "sentence");

  // embedding options
  cli.add<std::vector<std::string>>("--embedding-vectors",
     "Paths to files with custom source and target embedding vectors");
  cli.add<bool>("--embedding-normalization",
     "Normalize values from custom embedding vectors to [-1, 1]");
  cli.add<bool>("--embedding-fix-src",
     "Fix source embeddings. Affects all encoders");
  cli.add<bool>("--embedding-fix-trg",
     "Fix target embeddings. Affects all decoders");

  // mixed precision training
  cli.add<bool>("--fp16",
      "Shortcut for mixed precision training with float16 and cost-scaling, "
      "corresponds to: --precision float16 float32 --cost-scaling 8.f 10000 1.f 8.f");
  cli.add<std::vector<std::string>>("--precision",
      "Mixed precision training for forward/backward pass and optimizaton. "
      "Defines types for: forward/backward pass, optimization.",
      {"float32", "float32"});
  cli.add<std::vector<std::string>>("--cost-scaling",
      "Dynamic cost scaling for mixed precision training: "
      "scaling factor, frequency, multiplier, minimum factor")
      ->implicit_val("8.f 10000 1.f 8.f");

  cli.add<std::vector<std::string>>("--throw-on-divergence",
      "Throw exception if training diverges. Divergence is detected if the running average loss over arg1 steps "
      "is exceeded by the running average loss over arg2 steps (arg1 >> arg2) by arg3 standard deviations")
      ->implicit_val("1000 10 5.0f");
  cli.add<std::vector<YAML::Node>>("--custom-fallbacks",
      "List of custom fallback options after divergence. Each caught divergence exception thrown when --throw-on-divergence conditions are met progresses through another fallback. "
      "If more exception are caught than fallbacks were specified the process will terminate with an uncaught exception.");

  cli.add<bool>("--fp16-fallback-to-fp32",
      "If fp16 training diverges and throws try to continue training with fp32 precision");
  cli.alias("fp16-fallback-to-fp32", "true", [](YAML::Node& config) {
    // use default custom-fallbacks to handle DivergenceException for fp16
    config["custom-fallbacks"] = std::vector<YAML::Node>({
      YAML::Load("{fp16 : false, precision: [float32, float32], cost-scaling: []}")
     });
  });

  // @TODO: implement this next:
  // cli.add<std::string>("--recover-from-fallback-after",
  //     "Attempt to return to default options once the training has progressed in fallback mode by this many units. "
  //     "Allowed units are the same as for disp-freq (i.e. (u)pdates, (t)okens, (e)pochs)");

  cli.add<size_t>("--gradient-norm-average-window",
      "Window size over which the exponential average of the gradient norm is recorded (for logging and scaling). "
      "After this many updates about 90% of the mass of the exponential average comes from these updates",
      100);
  cli.add<std::vector<std::string>>("--dynamic-gradient-scaling",
      "Re-scale gradient to have average gradient norm if (log) gradient norm diverges from average by arg1 sigmas. "
      "If arg2 = \"log\" the statistics are recorded for the log of the gradient norm else use plain norm")
      ->implicit_val("2.f log");
  cli.add<bool>("--check-gradient-nan",
      "Skip parameter update in case of NaNs in gradient");
  cli.add<bool>("--normalize-gradient",
      "Normalize gradient by dividing with efficient batch size");
  cli.add<bool>("--normalize-gradient-by-ratio",
      "Normalize gradient by scaling with efficient batch size divided by running average batch size");

  cli.add<std::vector<std::string>>("--train-embedder-rank",
      "Override model configuration and train a embedding similarity ranker with the model encoder, "
      "parameters encode margin and an optional normalization factor")
    ->implicit_val("0.3f 0.0f");

  // model quantization training
  addSuboptionsQuantization(cli);

  // add ULR settings
  addSuboptionsULR(cli);

  cli.add<std::vector<std::string>>("--task",
     "Use predefined set of options. Possible values: transformer-base, transformer-big, "
     "transformer-base-prenorm, transformer-big-prenorm");
  cli.switchGroup(previous_group);
  // clang-format on
}

void ConfigParser::addOptionsValidation(cli::CLIWrapper& cli) {
  auto previous_group = cli.switchGroup("Validation set options");

  // clang-format off
  cli.add<std::vector<std::string>>("--valid-sets",
      "Paths to validation corpora: source target");
  cli.add<std::string/*SchedulerPeriod*/>("--valid-freq",
      "Validate model every arg updates (append 't' for every arg target labels)",
      "10000u");
  cli.add<std::vector<std::string>>("--valid-metrics",
      "Metric to use during validation: cross-entropy, ce-mean-words, perplexity, valid-script, "
      "translation, bleu, bleu-detok (deprecated, same as bleu), bleu-segmented, chrf. "
      "Multiple metrics can be specified",
      {"cross-entropy"});
  cli.add<bool>("--valid-reset-stalled",
      "Reset stalled validation metrics when the training is restarted");
  cli.add<bool>("--valid-reset-all",
      "Reset all validation metrics when the training is restarted");
  cli.add<size_t>("--early-stopping",
      "Stop if the first validation metric does not improve for arg consecutive validation steps",
      10);
  cli.add<std::vector<float>>("--early-stopping-epsilon",
      "An improvement lower than or equal to arg does not prevent stalled validation. "
      "i-th value corresponds to i-th metric in --valid-metrics",
      {0});
  cli.add<std::string>("--early-stopping-on",
      "Decide if early stopping should take into account first, all, or any validation metrics. "
      "Possible values: first, all, any",
      "first");

  // decoding options
  cli.add<size_t>("--beam-size,-b",
      "Beam size used during search with validating translator",
      12);
  cli.add<float>("--normalize,-n",
      "Divide translation score by pow(translation length, arg)",
      0)->implicit_val("1");
  cli.add<float>("--max-length-factor",
      "Maximum target length as source length times factor",
      3);
  cli.add<float>("--word-penalty",
      "Subtract (arg * translation length) from translation score");
  cli.add<bool>("--allow-unk",
      "Allow unknown words to appear in output");
  cli.add<bool>("--n-best",
      "Generate n-best list");
  cli.add<bool>("--word-scores",
      "Print word-level scores. One score per subword unit, not normalized even if --normalize");

  // efficiency options
  cli.add<int>("--valid-mini-batch",
      "Size of mini-batch used during validation",
      32);
  cli.add<size_t>("--valid-max-length",
      "Maximum length of a sentence in a validating sentence pair. "
      "Sentences longer than valid-max-length are cropped to valid-max-length",
      1000);

  // options for validation script
  cli.add<std::string>("--valid-script-path",
     "Path to external validation script."
     " It should print a single score to stdout."
     " If the option is used with validating translation, the output"
     " translation file will be passed as a first argument");
  cli.add<std::vector<std::string>>("--valid-script-args",
      "Additional args passed to --valid-script-path. These are inserted"
      " between the script path and the output translation-file path");
  cli.add<std::string>("--valid-translation-output",
     "(Template for) path to store the translation. "
     "E.g., validation-output-after-{U}-updates-{T}-tokens.txt. Template "
     "parameters: {E} for epoch; {B} for No. of batches within epoch; "
     "{U} for total No. of updates; {T} for total No. of tokens seen.");
  cli.add<bool>("--keep-best",
      "Keep best model for each validation metric");
  cli.add<std::string>("--valid-log",
     "Log validation scores to file given by arg");

  // general options for validation
  cli.add<bool>("--quiet-validation",
    "Suppress logging hyp./ref. samples during validation");

  cli.switchGroup(previous_group);
  // clang-format on
}

void ConfigParser::addOptionsTranslation(cli::CLIWrapper& cli) {
  auto previous_group = cli.switchGroup("Translator options");

  // clang-format off
  cli.add<std::vector<std::string>>("--input,-i",
      "Paths to input file(s), stdin by default",
      {"stdin"});
  cli.add<std::string>("--output,-o",
      "Path to output file, stdout by default",
      "stdout");
  cli.add<std::vector<std::string>>("--vocabs,-v",
      "Paths to vocabulary files have to correspond to --input");
  // decoding options
  cli.add<size_t>("--beam-size,-b",
      "Beam size used during search with validating translator",
      12);
  cli.add<float>("--normalize,-n",
      "Divide translation score by pow(translation length, arg)",
      0)->implicit_val("1");
  cli.add<float>("--max-length-factor",
      "Maximum target length as source length times factor",
      3);
  cli.add<float>("--word-penalty",
      "Subtract (arg * translation length) from translation score");
  cli.add<bool>("--allow-unk",
      "Allow unknown words to appear in output");
  cli.add<bool>("--allow-special",
      "Allow special symbols to appear in output, e.g. for SentencePiece with byte-fallback do not suppress the newline symbol");
  cli.add<bool>("--n-best",
      "Generate n-best list");
  cli.add<std::string>("--alignment",
     "Return word alignment. Possible values: 0.0-1.0, hard, soft")
    ->implicit_val("1");
  cli.add<bool>("--force-decode",
     "Use force-decoding of given prefixes. Forces decoding to follow vocab IDs from last stream in the batch (or the first stream, if there is only one). "
     "Use either as `./marian-decoder --force-decode --input source.txt prefixes.txt [...]` where inputs and prefixes align on line-level or as "
     "`paste source.txt prefixes.txt | ./marian-decoder --force-decode --tsv --tsv-fields 2 [...]` when reading from stdin."
  );
  cli.add<bool>("--word-scores",
      "Print word-level scores. One score per subword unit, not normalized even if --normalize");
  cli.add<std::string/*SchedulerPeriod*/>("--stat-freq",
    "Display speed information every arg mini-batches. Disabled by default with 0, set to value larger than 0 to activate",
    "0");
#ifdef USE_SENTENCEPIECE
  cli.add<bool>("--no-spm-decode",
      "Keep the output segmented into SentencePiece subwords");
  cli.add<bool>("--no-spm-encode",
      "Assume the input has already had sentencepiece applied before decoding. "
      "Expects spm pieces, like the ones produced by spm_encode's default format.");
#endif

  addSuboptionsInputLength(cli);
  addSuboptionsTSV(cli);
  addSuboptionsDevices(cli);
  addSuboptionsBatching(cli);

  cli.add<bool>("--fp16",
      "Shortcut for mixed precision inference with float16, corresponds to: --precision float16");
  cli.add<std::vector<std::string>>("--precision",
      "Mixed precision for inference, set parameter type in expression graph",
      {"float32"});
  cli.add<bool>("--skip-cost",
    "Ignore model cost during translation, not recommended for beam-size > 1");

  cli.add<std::vector<std::string>>("--shortlist",
     "Use softmax shortlist: path first best prune");
  cli.add<std::vector<float>>("--weights",
      "Scorer weights");
  cli.add<std::vector<std::string>>("--output-sampling",
     "Noise output layer with gumbel noise. Implicit default is 'full 1.0' for sampling from full distribution"
     " with softmax temperature 1.0. Also accepts 'topk num temp' (e.g. topk 100 0.1) for top-100 sampling with"
     " temperature 0.1")
     ->implicit_val("full 1.0");
  cli.add<std::vector<int>>("--output-approx-knn",
     "Use approximate knn search in output layer (currently only in transformer)")
     ->implicit_val("100 1024");

  // parameters for on-line quantization
  cli.add<bool>("--optimize",
      "Optimize the graph on-the-fly", false);
  cli.add<std::string>("--gemm-type,-g",
     "GEMM Type to be used for on-line quantization/packing: float32, packed16, packed8", "float32");
  cli.add<float>("--quantize-range",
     "Range for the on-line quantiziation of weight matrix in multiple of this range and standard deviation, 0.0 means min/max quantization",
     0.f);

#if 0 // @TODO: Ask Hany if there are any decoding-time options
  // add ULR settings
  addSuboptionsULR(cli);
#endif

#if USE_SSL
  cli.add<std::string>("--encryption-key-path",
      "Path to file with encryption key of model file", "");
#endif

  cli.switchGroup(previous_group);
  // clang-format on
}

void ConfigParser::addOptionsScoring(cli::CLIWrapper& cli) {
  auto previous_group = cli.switchGroup("Scorer options");

  // clang-format off
  cli.add<bool>("--no-reload",
      "Do not load existing model specified in --model arg");
  // TODO: move options like vocabs and train-sets to a separate procedure as they are defined twice
  cli.add<std::vector<std::string>>("--train-sets,-t",
      "Paths to corpora to be scored: source target");
  cli.add<std::string>("--output,-o",
      "Path to output file, stdout by default",
      "stdout");
  cli.add<std::vector<std::string>>("--vocabs,-v",
      "Paths to vocabulary files have to correspond to --train-sets. "
      "If this parameter is not supplied we look for vocabulary files source.{yml,json} and target.{yml,json}. "
      "If these files do not exists they are created");
  cli.add<bool>("--n-best",
      "Score n-best list instead of plain text corpus");
  cli.add<std::string>("--n-best-feature",
      "Feature name to be inserted into n-best list", "Score");
  cli.add<bool>("--normalize,-n",
      "Divide translation score by translation length");
  cli.add<std::string>("--summary",
      "Only print total cost, possible values: cross-entropy (ce-mean), ce-mean-words, ce-sum, perplexity")
      ->implicit_val("cross-entropy");
  cli.add<std::string>("--alignment",
     "Return word alignments. Possible values: 0.0-1.0, hard, soft")
     ->implicit_val("1"),
  cli.add<bool>("--word-scores",
      "Print word-level scores. One score per subword unit, not normalized even if --normalize");

  addSuboptionsInputLength(cli);
  addSuboptionsTSV(cli);
  addSuboptionsDevices(cli);
  addSuboptionsBatching(cli);

  cli.add<bool>("--fp16",
      "Shortcut for mixed precision inference with float16, corresponds to: --precision float16");
  cli.add<std::vector<std::string>>("--precision",
      "Mixed precision for inference, set parameter type in expression graph",
      {"float32"});

  // parameters for on-line quantization
  cli.add<bool>("--optimize",
      "Optimize the graph on-the-fly", false);
  cli.add<std::string>("--gemm-type,-g",
     "GEMM Type to be used for on-line quantization/packing: float32, packed16, packed8", "float32");
  cli.add<float>("--quantize-range",
     "Range for the on-line quantiziation of weight matrix in multiple of this range and standard deviation, 0.0 means min/max quantization",
     0.f);

  cli.switchGroup(previous_group);
  // clang-format on
}

void ConfigParser::addOptionsEmbedding(cli::CLIWrapper& cli) {
  auto previous_group = cli.switchGroup("Embedder options");

  // clang-format off
  cli.add<bool>("--no-reload",
      "Do not load existing model specified in --model arg");
  // TODO: move options like vocabs and train-sets to a separate procedure as they are defined twice
  cli.add<std::vector<std::string>>("--train-sets,-t",
      "Paths to corpora to be scored: source target");
  cli.add<std::string>("--output,-o",
      "Path to output file, stdout by default",
      "stdout");
  cli.add<std::vector<std::string>>("--vocabs,-v",
      "Paths to vocabulary files have to correspond to --train-sets. "
      "If this parameter is not supplied we look for vocabulary files source.{yml,json} and target.{yml,json}. "
      "If these files do not exists they are created");

  cli.add<bool>("--compute-similarity",
      "Expect two inputs and compute cosine similarity instead of outputting embedding vector");
  cli.add<bool>("--binary",
      "Output vectors as binary floats");

  addSuboptionsInputLength(cli);
  addSuboptionsTSV(cli);
  addSuboptionsDevices(cli);
  addSuboptionsBatching(cli);

  cli.add<bool>("--fp16",
      "Shortcut for mixed precision inference with float16, corresponds to: --precision float16");
  cli.add<std::vector<std::string>>("--precision",
      "Mixed precision for inference, set parameter type in expression graph. Supported values: float32, float16",
      {"float32"});

  cli.add<std::string>("--like",
    "Set good defaults for supported embedder types: roberta (works for all COMET flavors)");

  // Short-cut for Unbabel comet-qe metric
  cli.alias("like", "roberta", [](YAML::Node& config) {
    // Model options
    config["train-sets"] = std::vector<std::string>({"stdin"});
    config["input-types"] = std::vector<std::string>({"sequence"});
    config["max-length"] = 512;
    config["max-length-crop"] = true;
    config["mini-batch"] = 32;
    config["maxi-batch"] = 100;
    config["maxi-batch-sort"] = "src";
    config["workspace"] = -4000;
    config["devices"] = std::vector<std::string>({"all"});
  });

  cli.switchGroup(previous_group);
  // clang-format on
}

void ConfigParser::addOptionsEvaluating(cli::CLIWrapper& cli) {
  auto previous_group = cli.switchGroup("Evaluator options");

  cli.add<bool>("--no-reload",
      "Do not load existing model specified in --model arg");
  // @TODO: move options like vocabs and train-sets to a separate procedure as they are defined twice
  cli.add<std::vector<std::string>>("--train-sets,-t",
      "Paths to corpora to be scored: source target");
  cli.add<std::string>("--output,-o",
      "Path to output file, stdout by default",
      "stdout");
  cli.add<std::vector<std::string>>("--vocabs,-v",
      "Paths to vocabulary files have to correspond to --train-sets. "
      "If this parameter is not supplied we look for vocabulary files source.{yml,json} and target.{yml,json}. "
      "If these files do not exists they are created");
  cli.add<size_t>("--width",
      "Floating point precision of metric outputs",
      4);
  cli.add<std::string>("--average",
      "Report average of all sentence-level values. By default the average is appended as the last line. "
      "Alternatively, we can provide `--average only` which supresses other values.",
      "skip")->implicit_val("append");

  addSuboptionsInputLength(cli);
  addSuboptionsTSV(cli);
  addSuboptionsDevices(cli);
  addSuboptionsBatching(cli);

  cli.add<bool>("--fp16",
      "Shortcut for mixed precision inference with float16, corresponds to: --precision float16");
  cli.add<std::vector<std::string>>("--precision",
      "Mixed precision for inference, set parameter type in expression graph. Supported values: float32, float16",
      {"float32"});

  cli.add<std::string>("--like",
      "Set good defaults for supported metric types: comet-qe, comet, bleurt");

  // Short-cut for Unbabel comet-qe metric
  cli.alias("like", "comet-qe", [](YAML::Node& config) {
    // Model options
    config["train-sets"] = std::vector<std::string>({"stdin"});
    config["tsv"] = true;
    config["tsv-fields"] = 2;
    config["input-types"] = std::vector<std::string>({"sequence", "sequence"});
    config["max-length"] = 512;
    config["max-length-crop"] = true;
    config["mini-batch"] = 32;
    config["maxi-batch"] = 100;
    config["maxi-batch-sort"] = "src";
    config["workspace"] = -4000;
    config["devices"] = std::vector<std::string>({"all"});
  });

  // Short-cut for Unbabel comet metric
  cli.alias("like", "comet", [cli](YAML::Node& config) {
    // Model options
    config["train-sets"] = std::vector<std::string>({"stdin"});
    config["tsv"] = true;
    config["tsv-fields"] = 3;
    config["input-types"] = std::vector<std::string>({"sequence", "sequence", "sequence"});
    config["max-length"] = 512;
    config["max-length-crop"] = true;
    config["mini-batch"] = 32;
    config["maxi-batch"] = 100;
    config["maxi-batch-sort"] = "src";
    config["workspace"] = -4000;
    config["devices"] = std::vector<std::string>({"all"});
  });

  // Short-cut for Google bleurt metric
  cli.alias("like", "bleurt", [](YAML::Node& config) {
    // Model options
    config["train-sets"] = std::vector<std::string>({"stdin"});
    config["tsv"] = true;
    config["tsv-fields"] = 2;
    config["input-types"] = std::vector<std::string>({"sequence", "sequence"});
    config["max-length"] = 512;
    config["max-length-crop"] = true;
    config["mini-batch"] = 32;
    config["maxi-batch"] = 100;
    config["maxi-batch-sort"] = "src";
    config["workspace"] = -4000;
    config["devices"] = std::vector<std::string>({"all"});
  });

  cli.switchGroup(previous_group);
  // clang-format on
}


void ConfigParser::addSuboptionsDevices(cli::CLIWrapper& cli) {
  // clang-format off
  cli.add<std::vector<std::string>>("--devices,-d",
      "Specifies GPU ID(s) (e.g. '0 1 2 3' or 'all') to use for training. Defaults to GPU ID 0",
      {"0"});
#ifdef USE_NCCL
  if(mode_ == cli::mode::training) {
    cli.add<bool>("--no-nccl",
      "Disable inter-GPU communication via NCCL");
    cli.add<std::string>("--sharding",
      "When using NCCL and MPI for multi-process training use 'global' (default, less memory usage) "
      "or 'local' (more memory usage but faster) sharding",
      {"global"});
    cli.add<std::string/*SchedulerPeriod*/>("--sync-freq",
      "When sharding is local sync all shards across processes once every n steps (possible units u=updates, t=target labels, e=epochs)",
      "200u");
  }
#endif
#ifdef CUDA_FOUND
  cli.add<size_t>("--cpu-threads",
      "Use CPU-based computation with this many independent threads, 0 means GPU-based computation",
      0)
    ->implicit_val("1");
#else
  cli.add<size_t>("--cpu-threads",
      "Use CPU-based computation with this many independent threads, 0 means GPU-based computation",
      1);
#endif
  // clang-format on
}

void ConfigParser::addSuboptionsBatching(cli::CLIWrapper& cli) {
  int defaultMiniBatch = (mode_ == cli::mode::translation) ? 1 : 64;
  int defaultMaxiBatch = (mode_ == cli::mode::translation) ? 1 : 100;
  std::string defaultMaxiBatchSort = (mode_ == cli::mode::translation) ? "none" : "trg";

  // clang-format off
  cli.add<int>("--mini-batch",
               // set accurate help messages for translation, scoring, or training
               (mode_ == cli::mode::translation)
                   ? "Size of mini-batch used during batched translation" :
               (mode_ == cli::mode::scoring)
                   ? "Size of mini-batch used during batched scoring"
                   : "Size of mini-batch used during update",
               defaultMiniBatch);
  cli.add<int>("--mini-batch-words",
      "Set mini-batch size based on words instead of sentences");

  if(mode_ == cli::mode::training) {
    cli.add<bool>("--mini-batch-fit",
      "Determine mini-batch size automatically based on sentence-length to fit reserved memory");
    cli.add<size_t>("--mini-batch-fit-step",
      "Step size for mini-batch-fit statistics",
      10);
    cli.add<bool>("--gradient-checkpointing",
      "Enable gradient-checkpointing to minimize memory usage");
  }

  cli.add<int>("--maxi-batch",
      "Number of batches to preload for length-based sorting",
      defaultMaxiBatch);
  cli.add<std::string>("--maxi-batch-sort",
      "Sorting strategy for maxi-batch: none, src, trg (not available for decoder)",
      defaultMaxiBatchSort);

  if(mode_ == cli::mode::training) {
    cli.add<bool>("--shuffle-in-ram",
        "Keep shuffled corpus in RAM, do not write to temp file");

#if DETERMINISTIC
    cli.add<size_t>("--data-threads",
        "Number of concurrent threads to use during data reading and processing", 1);
#else
    cli.add<size_t>("--data-threads",
        "Number of concurrent threads to use during data reading and processing", 8);
#endif

    // @TODO: Consider making the next two options options of the vocab instead, to make it more local in scope.
    cli.add<size_t>("--all-caps-every",
        "When forming minibatches, preprocess every Nth line on the fly to all-caps. Assumes UTF-8");
    cli.add<size_t>("--english-title-case-every",
        "When forming minibatches, preprocess every Nth line on the fly to title-case. Assumes English (ASCII only)");

    cli.add<size_t>("--mini-batch-words-ref",
        "If given, the following hyper parameters are adjusted as-if we had this mini-batch size: "
        "--learn-rate, --optimizer-params, --exponential-smoothing, --mini-batch-warmup");
    cli.add<std::string/*SchedulerPeriod*/>("--mini-batch-warmup",
        "Linear ramp-up of MB size, up to this #updates (append 't' for up to this #target labels). "
        "Auto-adjusted to --mini-batch-words-ref if given",
        {"0"});
    cli.add<bool>("--mini-batch-track-lr",
        "Dynamically track mini-batch size inverse to actual learning rate (not considering lr-warmup)");
    cli.add<bool>("--mini-batch-round-up",
        "Round up batch size to next power of 2 for more efficient training, but this can make batch size less stable. Disable with --mini-batch-round-up=false",
        true);
  } else {
#if DETERMINISTIC
    cli.add<size_t>("--data-threads",
        "Number of concurrent threads to use during data reading and processing", 1);
#else
    cli.add<size_t>("--data-threads",
        "Number of concurrent threads to use during data reading and processing", 8);
#endif
  }
  // clang-format on
}

void ConfigParser::addSuboptionsInputLength(cli::CLIWrapper& cli) {
  size_t defaultMaxLength = (mode_ == cli::mode::training) ? 50 : 1000;
  // clang-format off
  cli.add<size_t>("--max-length",
      "Maximum length of a sentence in a training sentence pair",
      defaultMaxLength);
  cli.add<bool>("--max-length-crop",
      "Crop a sentence to max-length instead of omitting it if longer than max-length");
  // clang-format on
}

void ConfigParser::addSuboptionsTSV(cli::CLIWrapper& cli) {
  // clang-format off
  cli.add<bool>("--tsv",
      "Tab-separated input");
  cli.add<size_t>("--tsv-fields",
      "Number of fields in the TSV input. By default, it is guessed based on the model type");
  // clang-format on
}

void ConfigParser::addSuboptionsULR(cli::CLIWrapper& cli) {
  // clang-format off
  // support for universal encoder ULR https://arxiv.org/pdf/1802.05368.pdf
  cli.add<bool>("--ulr",
      "Enable ULR (Universal Language Representation)");
  // reading pre-trained universal embeddings for multi-sources.
  // Note that source and target here is relative to ULR not the translation langs
  // queries: EQ in Fig2 : is the unified embeddings projected to one space.
  cli.add<std::string>("--ulr-query-vectors",
      "Path to file with universal sources embeddings from projection into universal space",
      "");
  // keys: EK in Fig2 : is the keys of the target embeddings projected to unified space (i.e. ENU in
  // multi-lingual case)
  cli.add<std::string>("--ulr-keys-vectors",
      "Path to file with universal sources embeddings of target keys from projection into universal space",
      "");
  cli.add<bool>("--ulr-trainable-transformation",
      "Make Query Transformation Matrix A trainable");
  cli.add<int>("--ulr-dim-emb",
      "ULR monolingual embeddings dimension");
  cli.add<float>("--ulr-dropout",
      "ULR dropout on embeddings attentions. Default is no dropout",
      0.0f);
  cli.add<float>("--ulr-softmax-temperature",
      "ULR softmax temperature to control randomness of predictions. Deafult is 1.0: no temperature",
      1.0f);
  // clang-format on
}

void ConfigParser::addSuboptionsQuantization(cli::CLIWrapper& cli) {
  // clang-format off
  // model quantization training
  cli.add<size_t>("--quantize-bits",
     "Number of bits to compress model to. Set to 0 to disable",
      0);
  cli.add<size_t>("--quantize-optimization-steps",
     "Adjust quantization scaling factor for N steps",
     0);
  cli.add<bool>("--quantize-log-based",
     "Uses log-based quantization");
  cli.add<bool>("--quantize-biases",
     "Apply quantization to biases");
  // clang-format on
}

cli::mode ConfigParser::getMode() const { return mode_; }

Ptr<Options> ConfigParser::parseOptions(int argc, char** argv, bool doValidate) {
  cmdLine_ = escapeCmdLine(argc,argv);

  // parse command-line options and fill wrapped YAML config
  cli_.parse(argc, argv);

  if(get<bool>("authors")) {
    std::cerr << authors() << std::endl;
    exit(0);
  }

  if(get<bool>("cite")) {
    std::cerr << citation() << std::endl;
    exit(0);
  }

  auto buildInfo = get<std::string>("build-info");
  if(!buildInfo.empty() && buildInfo != "false") {
#ifdef BUILD_INFO_AVAILABLE // cmake build options are not available on MSVC based build.
    if(buildInfo == "all")
      std::cerr << cmakeBuildOptionsAdvanced() << std::endl;
    else
      std::cerr << cmakeBuildOptions() << std::endl;
    exit(0);
#else // BUILD_INFO_AVAILABLE
    ABORT("build-info is not available on MSVC based build unless compiled via CMake.");
#endif // BUILD_INFO_AVAILABLE
  }

  // get paths to extra config files
  auto configPaths = findConfigPaths();
  if(!configPaths.empty()) {
    auto config = loadConfigFiles(configPaths);
    cli_.updateConfig(config,
                     cli::OptionPriority::ConfigFile,
                     "There are option(s) in a config file that are not expected");
  }

  if(get<bool>("interpolate-env-vars")) {
    cli::processPaths(config_, cli::interpolateEnvVars, PATHS);
  }

  // Option shortcuts for input from STDIN for trainer and scorer
  if(mode_ == cli::mode::training || mode_ == cli::mode::scoring) {
    auto trainSets = get<std::vector<std::string>>("train-sets");
    YAML::Node config;
    // Assume the input will come from STDIN if --tsv is set but no --train-sets are given
    if(get<bool>("tsv") && trainSets.empty()) {
      config["train-sets"].push_back("stdin");
    // Assume the input is in TSV format if --train-sets is set to "stdin"
    } else if(trainSets.size() == 1 && (trainSets[0] == "stdin" || trainSets[0] == "-")) {
      config["tsv"] = true;
    }
    if(!config.IsNull())
      cli_.updateConfig(config, cli::OptionPriority::CommandLine, "A shortcut for STDIN failed.");
  }

  // remove extra config files from the config to avoid redundancy
  config_.remove("config");

  // dump config and exit
  if(!get<std::string>("dump-config").empty() && get<std::string>("dump-config") != "false") {
    auto dumpMode = get<std::string>("dump-config");
    config_.remove("dump-config");

    if(dumpMode == "expand") {
      cli_.parseAliases();
    }

    if(doValidate) {  // validate before options are dumped and we exit
      ConfigValidator(config_, true).validateOptions(mode_);
    }

    bool minimal = (dumpMode == "minimal" || dumpMode == "expand");
    std::cout << cli_.dumpConfig(minimal) << std::endl;
    exit(0);
  }

  // For TSV input, it is possible to use --input-types to determine fields that contain alignments
  // or weights. In such case, the position of 'alignment' input type in --input-types determines
  // the index of a TSV field that contains word alignments, and respectively, the position of
  // 'weight' in --input-types determines the index of a TSV field that contains weights.
  // Marian will abort if both the --guided-alignment and 'alignment' in --input-types are specified
  // (or --data-weighting and 'weight').
  //
  // Note: this may modify the config, so it is safer to do it after --dump-config.
  if(mode_ == cli::mode::training || get<bool>("tsv")) {
    auto inputTypes = get<std::vector<std::string>>("input-types");
    if(!inputTypes.empty()) {
      bool seenAligns = false;
      bool seenWeight = false;
      YAML::Node config;
      for(size_t i = 0; i < inputTypes.size(); ++i) {
        if(inputTypes[i] == "alignment") {
          ABORT_IF(seenAligns, "You can specify 'alignment' only once in input-types");
          ABORT_IF(has("guided-alignment") && get<std::string>("guided-alignment") != "none",
                   "You must use either guided-alignment or 'alignment' in input-types");
          config["guided-alignment"] = std::to_string(i);
          seenAligns = true;
        }
        if(inputTypes[i] == "weight") {
          ABORT_IF(seenWeight, "You can specify 'weight' only once in input-types");
          ABORT_IF(has("data-weighting") && !get<std::string>("data-weighting").empty(),
                   "You must use either data-weighting or 'weight' in input-types");
          config["data-weighting"] = std::to_string(i);
          seenWeight = true;
        }
      }
      if(!config.IsNull())
        cli_.updateConfig(config,
                          cli::OptionPriority::CommandLine,
                          "Extracting 'alignment' and 'weight' types from input-types failed.");
    }
  }

#if 0 // @TODO: remove once fully deprecated
  // Convert --after-batches N to --after Nu and --after-epochs N to --after Ne, different values get concatenated with ","
  if(mode_ == cli::mode::training && get<size_t>("after-epochs") > 0) {
    auto afterValue = get<size_t>("after-epochs");
    LOG(info, "\"--after-epochs {}\" is deprecated, please use \"--after {}e\" instead (\"e\" stands for epoch)", afterValue, afterValue);
    YAML::Node config;
    std::string prevAfter = get<std::string>("after");
    std::string converted = std::to_string(afterValue) + "e";
    if(prevAfter != "0e")
      config["after"] = prevAfter + "," + converted;
    else
      config["after"] = converted;
    if(!config.IsNull())
      cli_.updateConfig(config,
                        cli::OptionPriority::CommandLine,
                        "Could not update --after with value from --after-epochs");
  }
  if(mode_ == cli::mode::training && get<size_t>("after-batches") > 0) {
    auto afterValue = get<size_t>("after-batches");
    LOG(info, "\"--after-batches {}\" is deprecated, please use \"--after {}u\" instead (\"u\" stands for updates)", afterValue, afterValue);
    YAML::Node config;
    std::string prevAfter = get<std::string>("after");
    std::string converted = std::to_string(afterValue) + "u";
    if(prevAfter != "0e")
      config["after"] = prevAfter + "," + converted;
    else
      config["after"] = converted;
    if(!config.IsNull())
      cli_.updateConfig(config,
                        cli::OptionPriority::CommandLine,
                        "Could not update --after with value from --after-updates");
  }
#endif

#if USE_SSL
  if(mode_ == cli::mode::translation && get<std::string>("encryption-key-path") != "") {
    const std::string keyPath = get<std::string>("encryption-key-path");
    Config::encryptionKey = crypt::read_file_sha256(keyPath);
  }
#endif

  cli_.parseAliases();
  if(doValidate) {  // validate the options after aliases are expanded
    ConfigValidator(config_).validateOptions(mode_);
  }

  auto opts = New<Options>();
  opts->merge(Config(*this).get());
  return opts;
}

std::vector<std::string> ConfigParser::findConfigPaths() {
  std::vector<std::string> paths;

  bool interpolateEnvVars = get<bool>("interpolate-env-vars");
  bool loadConfig = !config_["config"].as<std::vector<std::string>>().empty();

  if(loadConfig) {
    paths = config_["config"].as<std::vector<std::string>>();
    for(auto& path : paths) {
      // (note: this updates the paths array)
      if(interpolateEnvVars)
        path = cli::interpolateEnvVars(path);
    }
  } else if(mode_ == cli::mode::training) {
    auto path = config_["model"].as<std::string>() + ".yml";
    if(interpolateEnvVars)
      path = cli::interpolateEnvVars(path);

    bool reloadConfig = filesystem::exists(path) && !get<bool>("no-reload");
    if(reloadConfig)
      paths = {path};
  }

  return paths;
}

YAML::Node ConfigParser::loadConfigFiles(const std::vector<std::string>& paths) {
  YAML::Node configAll;

  for(auto& path : paths) {
    // load single config file
    io::InputFileStream strm(path);
    YAML::Node config = YAML::Load(strm);

    // expand relative paths if requested
    if(config["relative-paths"] && config["relative-paths"].as<bool>()) {
      // interpolate environment variables if requested in this config file or
      // via command-line options
      bool interpolateEnvVars = (config["interpolate-env-vars"]
                                 && config["interpolate-env-vars"].as<bool>())
                                || get<bool>("interpolate-env-vars");
      if(interpolateEnvVars)
        cli::processPaths(config, cli::interpolateEnvVars, PATHS);

      // replace relative path w.r.t. the config file
      cli::makeAbsolutePaths(config, path, PATHS);
      // remove 'relative-paths' and do not spread it into other config files
      config.remove("relative-paths");
    }

    // merge with previous config files, later file overrides earlier
    for(const auto& it : config) {
      configAll[it.first.as<std::string>()] = YAML::Clone(it.second);
    }
  }

  return configAll;
}

const YAML::Node& ConfigParser::getConfig() const {
  return config_;
}
}  // namespace marian
