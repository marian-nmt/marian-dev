# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Faster ARM64 matmul in `mjdgemm` using `vdotq_s32` instrinsics
- `mjdgemm` as a FBGEMM replacement, also SSE4.2 support and ARM support for 8bit avx512-style packed FBGEMM matrices
- Conpilation om Android
- Added Threads::Threads to `EXT_LIBS`
- Updates to pymarian: building for multiple python versions; disabling tcmalloc; hosting gated COMETs on HuggingFace
- Added `--normalize-gradient-by-ratio` to mildly adapt gradient magnitude if effective batch size diverges from running average effective batch size.
- Added `--no-optimizer-reload` to skip optimizer state loading during continued training or fallback.
- Added `pymarian-eval`, CLI for scoring metrics
- Added `--input-reorder pos1 pos2` option to re-ordering inputs internally when reading in batches. This is mostly a model property.
- Added `pymarian`: python bindings based on pybind11
- Added implementation of COMET-KIWI
- Added implementation of xCOMET-XL/XXL regressor parts (MQM interpolation missing for now)
- Added implementation of COMET-22 (reference-based) model and conversion
- Added sparsemax operator (slow version)
- Added sampling variants nucleus and epsilon, e.g. `--output-sampling nucleus 0.9` and `--output-sampling epsilon 0.02`, respectively.
- Added ALIBI related options to new layer framework.
- Added `--no-spm-encode` option, allowing the model to use vocabulary IDs directly to train/decode.
- Added MSE and MAE costs to COMET-QE training.
- Added augmentation of shuffled examples to COMET-QE training via `--comet-augment-bad`.
- Minor changes and fixes related to metric training.
- Added `--quiet-validation` option that disables printing Hyp/Ref samples during validation
- Added --custom-fallbacks option that allows to specify a list of option sets that get traversed for subsequent fallbacks upon divergence
- Added --overwrite-checkpoint option that (when set to false) can be used to dump checkpoints with iteration numbers.
- Implementations of COMET-20 (reference-based) and BLEURT-20 for inference with conversion scripts.
- `./marian evaluate` sub command for evaluation with COMET-QE-20, COMET-20 and BLEURT-20
- A bunch of scripts for metrics use and early MBR experiments
- LSH vocab filtering for GPU. Speed is not competitive with non-LSH. Checking in for completeness and possible future use of LSH on GPU for non-filtering stuff
- Added --throw-on-divergence and --fp16-fallback-to-fp32 options to detect (fp16 and fp32) and recover (only fp16)
  diverged runs. If not recoverable, exception gets rethrown and goes unhandled to force fatal error and shutdown.
- Re-implementation of COMET-QE for inference and training; conversion scripts from Unbabel-Comet to Marian.
- Validator that generates embeddings and can be used during COMET training with an external script.
- New experimental layer framework for Transformer-like models.

### Fixed
- Do not use shortlist-mapped values to check for force-decoding criterion
- Collapsing beam in force-decoding for beam > 1 and LSH
- Various small fixes for ARM compilation
- Fixed compilation with clang 16.0.6
- Do not mmap files for conversion via Quicksand API
- Fixed ALiBI states and caching in new layer framework
- Throw exception when forcing with FS vocabs
- Fixed force-decoding with LSH
- Fixed force-decoding for beam-size > 1
- Fixed lost node in mt-detect metrics
- Fixed BLEURT logmask computation
- Fixed wrong paramter name for norm in new layer framework
- Fixed unit test for LayerNorm
- Only collect batch statistics during mini-batch-fit up to actual max-length.
- Implemented fully correct version of GELU instead of using bad approximatin via Swish.
- Handle copying from fp32 or fp16 embeddings in embedder mode correctly.
- Correct defaults for factored embeddings such that shared library use works (move out of config.h/cpp).

### Changed
- Globally set mkl_set_num_threads(1)
- Refactoring of model loading, mmapping happens now opportunistically, --mmap-models for decoding forces mmap and croaks if not possible.
- Removed --num-devices N option that wasn't really used by anyone (I assume).


## [1.12.0] - 2023-02-20

### Added
- Fused inplace-dropout in FFN layer in Transformer
- `--force-decode` option for marian-decoder
- `--output-sampling` now works with ensembles (requires proper normalization via e.g `--weights 0.5 0.5`)
- `--valid-reset-all` option

### Fixed
- Make concat factors not break old vector implementation
- Use allocator in hashing
- Read/restore checkpoints from main process only when training with MPI
- Multi-loss casts type to first loss-type before accumulation (aborted before due to missing cast)
- Throw `ShapeSizeException` if total expanded shape size exceeds numeric capacity of the maximum int value (2^31-1)
- During mini-batch-fitting, catch `ShapeSizeException` and use another sizing hint. Aborts outside mini-batch-fitting.
- Fix incorrect/missing gradient accumulation with delay > 1 or large effective batch size of biases of affine operations.
- Fixed case augmentation with multi-threaded reading.
- Scripts using PyYAML now use `safe_load`; see https://msg.pyyaml.org/load
- Fixed check for `fortran_ordering` in cnpy
- Fixed fp16 training/inference with factors-combine concat method
- Fixed clang 13.0.1 compatibility
- Fixed potential vulnerabilities from lxml<4.9.1 or mistune<2.0.31
- Fixed the `--best-deep` RNN alias not setting the s2s model type

### Changed
- Parameter synchronization in local sharding model now executes hash checksum before syncing
- Make guided-alignment faster via sparse memory layout, add alignment points for EOS, remove losses other than ce
- Negative `--workspace -N` value allocates workspace as total available GPU memory minus N megabytes.
- Set default parameters for cost-scaling to 8.f 10000 1.f 8.f, i.e. when scaling scale by 8 and do not try to automatically scale up or down. This seems most stable.
- Make guided-alignment faster via sparse memory layout, add alignment points for EOS, remove losses other than ce.
- Changed minimal C++ standard to C++-17
- Faster LSH top-k search on CPU
- Updated intgemm to the latest upstream version
- Parameters in npz files are no longer implicitly assumed to be row-ordered. Non row-ordered parameters will result in an abort
- Updated Catch2 header from 2.10.1 to 2.13.9

## [1.11.0] - 2022-02-08

### Added
- Parallelized data reading with e.g. `--data-threads 8`
- Top-k sampling during decoding with e.g. `--output-sampling topk 10`
- Improved mixed precision training with `--fp16`
- Set FFN width in decoder independently from encoder with e.g. `--transformer-dim-ffn 4096 --transformer-decoder-dim-ffn 2048`
- Adds option `--add-lsh` to marian-conv which allows the LSH to be memory-mapped.
- Early stopping based on first, all, or any validation metrics via `--early-stopping-on`
- Compute 8.6 support if using CUDA>=11.1
- Support for RMSNorm as drop-in replace for LayerNorm from `Biao Zhang; Rico Sennrich (2019). Root Mean Square Layer Normalization`. Enabled in Transformer model via `--transformer-postprocess dar` instead of `dan`.
- Extend suppression of unwanted output symbols, specifically "\n" from default vocabulary if generated by SentencePiece with byte-fallback. Deactivates with --allow-special
- Allow for fine-grained CPU intrinsics overrides when BUILD_ARCH != native e.g. -DBUILD_ARCH=x86-64 -DCOMPILE_AVX512=off
- Adds custom bias epilogue kernel.
- Adds support for fusing relu and bias addition into gemms when using cuda 11.
- Better suppression of unwanted output symbols, specifically "\n" from SentencePiece with byte-fallback. Can be deactivated with --allow-special
- Display decoder time statistics with marian-decoder --stat-freq 10 ...
- Support for MS-internal binary shortlist
- Local/global sharding with MPI training via `--sharding local`
- fp16 support for factors.
- Correct training with fp16 via `--fp16`.
- Dynamic cost-scaling with `--cost-scaling`.
- Dynamic gradient-scaling with `--dynamic-gradient-scaling`.
- Add unit tests for binary files.
- Fix compilation with OMP
- Added `--model-mmap` option to enable mmap loading for CPU-based translation
- Compute aligned memory sizes using exact sizing
- Support for loading lexical shortlist from a binary blob
- Integrate a shortlist converter (which can convert a text lexical shortlist to a binary shortlist) into marian-conv with --shortlist option

### Fixed
- Fix AVX2 and AVX512 detection on MacOS
- Add GCC11 support into FBGEMM
- Added pragma to ignore unused-private-field error on elementType_ on macOS
- Do not set guided alignments for case augmented data if vocab is not factored
- Various fixes to enable LSH in Quicksand
- Added support to MPIWrappest::bcast (and similar) for count of type size_t
- Adding new validation metrics when training is restarted and --reset-valid-stalled is used
- Missing depth-scaling in transformer FFN
- Fixed an issue when loading intgemm16 models from unaligned memory.
- Fix building marian with gcc 9.3+ and FBGEMM
- Find MKL installed under Ubuntu 20.04 via apt-get
- Support for CUDA 11.
- General improvements and fixes for MPI handling, was essentially non-functional before (syncing, random seeds, deadlocks during saving, validation etc.)
- Allow to compile -DUSE_MPI=on with -DUSE_STATIC_LIBS=on although MPI gets still linked dynamically since it has so many dependencies.
- Fix building server with Boost 1.75
- Missing implementation for cos/tan expression operator
- Fixed loading binary models on architectures where `size_t` != `uint64_t`.
- Missing float template specialisation for elem::Plus
- Broken links to MNIST data sets
- Enforce validation for the task alias in training mode.

### Changed
- MacOS marian uses Apple Accelerate framework by default, as opposed to openblas/mkl.
- Optimize LSH for speed by treating is as a shortlist generator. No option changes in decoder
- Set REQUIRED_BIAS_ALIGNMENT = 16 in tensors/gpu/prod.cpp to avoid memory-misalignment on certain Ampere GPUs.
- For BUILD_ARCH != native enable all intrinsics types by default, can be disabled like this: -DCOMPILE_AVX512=off
- Moved FBGEMM pointer to commit c258054 for gcc 9.3+ fix
- Change compile options a la -DCOMPILE_CUDA_SM35 to -DCOMPILE_KEPLER, -DCOMPILE_MAXWELL,
-DCOMPILE_PASCAL, -DCOMPILE_VOLTA, -DCOMPILE_TURING and -DCOMPILE_AMPERE
- Disable -DCOMPILE_KEPLER, -DCOMPILE_MAXWELL by default.
- Dropped support for legacy graph groups.
- Developer documentation framework based on Sphinx+Doxygen+Breathe+Exhale
- Expresion graph documentation (#788)
- Graph operators documentation (#801)
- Remove unused variable from expression graph
- Factor groups and concatenation: doc/factors.md

## [1.10.0] - 2021-02-06

### Added
- Added `intgemm8(ssse3|avx|avx512)?`, `intgemm16(sse2|avx|avx512)?` types to marian-conv with uses intgemm backend. Types intgemm8 and intgemm16 are hardware-agnostic, the other ones hardware-specific.
- Shortlist is now always multiple-of-eight.
- Added intgemm 8/16bit integer binary architecture agnostic format.
- Add --train-embedder-rank for fine-tuning any encoder(-decoder) model for multi-lingual similarity via softmax-margin loss
- Add --logical-epoch that allows to redefine the displayed epoch counter as a multiple of n data epochs, updates or labels. Also allows to define width of fractional part with second argument.
- Add --metrics chrf for computing ChrF according to https://www.aclweb.org/anthology/W15-3049/ and SacreBLEU reference implementation
- Add --after option which is meant to replace --after-batches and --after-epochs and can take label based criteria
- Add --transformer-postprocess-top option to enable correctly normalized prenorm behavior
- Add --task transformer-base-prenorm and --task transformer-big-prenorm
- Turing and Ampere GPU optimisation support, if the CUDA version supports it.
- Printing word-level scores in marian-scorer
- Optimize LayerNormalization on CPU by 6x through vectorization (ffast-math) and fixing performance regression introduced with strides in 77a420
- Decoding multi-source models in marian-server with --tsv
- GitHub workflows on Ubuntu, Windows, and MacOS
- LSH indexing to replace short list
- ONNX support for transformer models (very experimental)
- Add topk operator like PyTorch's topk
- Use *cblas_sgemm_batch* instead of a for loop of *cblas_sgemm* on CPU as the batched_gemm implementation
- Supporting relative paths in shortlist and sqlite options
- Training and scoring from STDIN
- Support for reading from TSV files from STDIN and other sources during training
  and translation with options --tsv and --tsv-fields n.
- Internal optional parameter in n-best list generation that skips empty hypotheses.
- Quantized training (fixed point or log-based quantization) with --quantize-bits N command
- Support for using Apple Accelerate as the BLAS library

### Fixed
- Segfault of spm_train when compiled with -DUSE_STATIC_LIBS=ON seems to have gone away with update to newer SentencePiece version.
- Fix bug causing certain reductions into scalars to be 0 on the GPU backend. Removed unnecessary warp shuffle instructions.
- Do not apply dropout in embeddings layers during inference with dropout-src/trg
- Print "server is listening on port" message after it is accepting connections
- Fix compilation without BLAS installed
- Providing a single value to vector-like options using the equals sign, e.g. --models=model.npz
- Fix quiet-translation in marian-server
- CMake-based compilation on Windows
- Fix minor issues with compilation on MacOS
- Fix warnings in Windows MSVC builds using CMake
- Fix building server with Boost 1.72
- Make mini-batch scaling depend on mini-batch-words and not on mini-batch-words-ref
- In concatenation make sure that we do not multiply 0 with nan (which results in nan)
- Change Approx.epsilon(0.01) to Approx.margin(0.001) in unit tests. Tolerance is now
  absolute and not relative. We assumed incorrectly that epsilon is absolute tolerance.
- Fixed bug in finding .git/logs/HEAD when Marian is a submodule in another project.
- Properly record cmake variables in the cmake build directory instead of the source tree.
- Added default "none" for option shuffle in BatchGenerator, so that it works in executables where shuffle is not an option.
- Added a few missing header files in shortlist.h and beam_search.h.
- Improved handling for receiving SIGTERM during training. By default, SIGTERM triggers 'save (now) and exit'. Prior to this fix, batch pre-fetching did not check for this sigal, potentially delaying exit considerably. It now pays attention to that. Also, the default behaviour of save-and-exit can now be disabled on the command line with --sigterm exit-immediately.
- Fix the runtime failures for FASTOPT on 32-bit builds (wasm just happens to be 32-bit) because it uses hashing with an inconsistent mix of uint64_t and size_t.
- fix beam_search ABORT_IF(beamHypIdx >= beam.size(), "Out of bounds beamHypIdx??"); when enable openmp and OMP_NUM_THREADS > 1

### Changed
- Remove `--clip-gemm` which is obsolete and was never used anyway
- Removed `--optimize` switch, instead we now determine compute type based on binary model.
- Updated SentencePiece repository to version 8336bbd0c1cfba02a879afe625bf1ddaf7cd93c5 from https://github.com/google/sentencepiece.
- Enabled compilation of SentencePiece by default since no dependency on protobuf anymore.
- Changed default value of --sentencepiece-max-lines from 10000000 to 2000000 since apparently the new version doesn't sample automatically anymore (Not quite clear how that affects quality of the vocabulary).
- Change mini-batch-fit search stopping criterion to stop at ideal binary search threshold.
- --metric bleu now always detokenizes SacreBLEU-style if a vocabulary knows how to, use bleu-segmented to compute BLEU on word ids. bleu-detok is now a synonym for bleu.
- Move label-smoothing computation into Cross-entropy node
- Move Simple-WebSocket-Server to submodule
- Python scripts start with #!/usr/bin/env python3 instead of python
- Changed compile flags -Ofast to -O3 and remove --ffinite-math
- Moved old graph groups to depracated folder
- Make cublas and cusparse handle inits lazy to save memory when unused
- Replaced exception-based implementation for type determination in FastOpt::makeScalar

## [1.9.0] - 2020-03-10

### Added
- An option to print cached variables from CMake
- Add support for compiling on Mac (and clang)
- An option for resetting stalled validation metrics
- Add CMAKE options to disable compilation for specific GPU SM types
- An option to print word-level translation scores
- An option to turn off automatic detokenization from SentencePiece
- Separate quantization types for 8-bit FBGEMM for AVX2 and AVX512
- Sequence-level unliklihood training
- Allow file name templated valid-translation-output files
- Support for lexical shortlists in marian-server
- Support for 8-bit matrix multiplication with FBGEMM
- CMakeLists.txt now looks for SSE 4.2
- Purging of finished hypotheses during beam-search. A lot faster for large batches.
- Faster option look-up, up to 20-30% faster translation
- Added --cite and --authors flag
- Added optional support for ccache
- Switch to change abort to exception, only to be used in library mode
- Support for 16-bit packed models with FBGEMM
- Multiple separated parameter types in ExpressionGraph, currently inference-only
- Safe handling of sigterm signal
- Automatic vectorization of elementwise operations on CPU for tensors dims that
  are divisible by 4 (AVX) and 8 (AVX2)
- Replacing std::shared_ptr<T> with custom IntrusivePtr<T> for small objects like
  Tensors, Hypotheses and Expressions.
- Fp16 inference working for translation
- Gradient-checkpointing

### Fixed
- Replace value for INVALID_PATH_SCORE with std::numer_limits<float>::lowest()
  to avoid overflow with long sequences
- Break up potential circular references for GraphGroup*
- Fix empty source batch entries with batch purging
- Clear RNN chache in transformer model, add correct hash functions to nodes
- Gather-operation for all index sizes
- Fix word weighting with max length cropping
- Fixed compilation on CPUs without support for AVX
- FastOpt now reads "n" and "y" values as strings, not as boolean values
- Fixed multiple reduction kernels on GPU
- Fixed guided-alignment training with cross-entropy
- Replace IntrusivePtr with std::uniq_ptr in FastOpt, fixes random segfaults
  due to thread-non-safty of reference counting.
- Make sure that items are 256-byte aligned during saving
- Make explicit matmul functions respect setting of cublasMathMode
- Fix memory mapping for mixed paramter models
- Removed naked pointer and potential memory-leak from file_stream.{cpp,h}
- Compilation for GCC >= 7 due to exception thrown in destructor
- Sort parameters by lexicographical order during allocation to ensure consistent
  memory-layout during allocation, loading, saving.
- Output empty line when input is empty line. Previous behavior might result in
  hallucinated outputs.
- Compilation with CUDA 10.1

### Changed
- Combine two for-loops in nth_element.cpp on CPU
- Revert LayerNorm eps to old position, i.e. sigma' = sqrt(sigma^2 + eps)
- Downgrade NCCL to 2.3.7 as 2.4.2 is buggy (hangs with larger models)
- Return error signal on SIGTERM
- Dropped support for CUDA 8.0, CUDA 9.0 is now minimal requirement
- Removed autotuner for now, will be switched back on later
- Boost depdendency is now optional and only required for marian_server
- Dropped support for g++-4.9
- Simplified file stream and temporary file handling
- Unified node intializers, same function API.
- Remove overstuff/understuff code

## [1.8.0] - 2019-09-04

### Added
- Alias options and new --task option
- Automatic detection of CPU intrisics when building with -arch=native
- First version of BERT-training and BERT-classifier, currently not compatible with TF models
- New reduction operators
- Use Cmake's ExternalProject to build NCCL and potentially other external libs
- Code for Factored Vocabulary, currently not usable yet without outside tools

### Fixed
- Issue with relative paths in automatically generated decoder config files
- Bug with overlapping CXX flags and building spm_train executable
- Compilation with gcc 8
- Overwriting and unsetting vector options
- Windows build with recent changes
- Bug with read-ahead buffer
- Handling of "dump-config: false" in YAML config
- Errors due to warnings
- Issue concerning failed saving with single GPU training and --sync-sgd option.
- NaN problem when training with Tensor Cores on Volta GPUs
- Fix pipe-handling
- Fix compilation with GCC 9.1
- Fix CMake build types

### Changed
- Error message when using left-to-right and right-to-left models together in ensembles
- Regression tests included as a submodule
- Update NCCL to 2.4.2
- Add zlib source to Marian's source tree, builds now as object lib
- -DUSE_STATIC_LIBS=on now also looks for static versions of CUDA libraries
- Include NCCL build from github.com/marian-nmt/nccl and compile within source tree
- Set nearly all warnings as errors for Marian's own targets. Disable warnings for 3rd party
- Refactored beam search

## [1.7.0] - 2018-11-27

### Added
- Word alignment generation in scorer
- Attention output generation in decoder and scorer with `--alignment soft`
- Support for SentencePiece vocabularies and run-time segmentation/desegmentation
- Support for SentencePiece vocabulary training during model training
- Group training files by filename when creating vocabularies for joint vocabularies
- Updated examples
- Synchronous multi-node training (early version)

### Fixed
- Delayed output in line-by-line translation

### Changed
- Generated word alignments include alignments for target EOS tokens
- Boost::program_options has been replaced by another CLI library
- Replace boost::file_system with Pathie
- Expansion of unambiguous command-line arguments is no longer supported

## [1.6.0] - 2018-08-08

### Added
- Faster training (20-30%) by optimizing gradient popagation of biases
- Returning Moses-style hard alignments during decoding single models,
  ensembles and n-best lists
- Hard alignment extraction strategy taking source words that have the
  attention value greater than the threshold
- Refactored sync sgd for easier communication and integration with NCCL
- Smaller memory-overhead for sync-sgd
- NCCL integration (version 2.2.13)
- New binary format for saving/load of models, can be used with _*.bin_
  extension (can be memory mapped)
- Memory-mapping of graphs for inferece with `ExpressionGraph::mmap(const void*
  ptr)` function. (assumes _*.bin_ model is mapped or in buffer)
- Added SRU (--dec-cell sru) and ReLU (--dec-cell relu) cells to inventory of
  RNN cells
- RNN auto-regression layers in transformer (`--transformer-decoder-autreg
  rnn`), work with gru, lstm, tanh, relu, sru cells
- Recurrently stacked layers in transformer (`--transformer-tied-layers 1 1 1 2
  2 2` means 6 layers with 1-3 and 4-6 tied parameters, two groups of
  parameters)
- Seamless training continuation with exponential smoothing

### Fixed
- A couple of bugs in "selection" (transpose, shift, cols, rows) operators
  during back-prob for a very specific case: one of the operators is the first
  operator after a branch, in that case gradient propgation might be
  interrupted. This did not affect any of the existing models as such a case
  was not present, but might have caused future models to not train properly
- Bug in mini-batch-fit, tied embeddings would result in identical embeddings
  in fake source and target batch. Caused under-estimation of memory usage and
  re-allocation

## [1.5.0] - 2018-06-17

### Added
- Average Attention Networks for Transformer model
- 16-bit matrix multiplication on CPU
- Memoization for constant nodes for decoding
- Autotuning for decoding

### Fixed
- GPU decoding optimizations, about 2x faster decoding of transformer models
- Multi-node MPI-based training on GPUs

## [1.4.0] - 2018-03-13

### Added
- Data weighting with `--data-weighting` at sentence or word level
- Persistent SQLite3 corpus storage with `--sqlite file.db`
- Experimental multi-node asynchronous training
- Restoring optimizer and training parameters such as learning rate, validation
  results, etc.
- Experimental multi-CPU training/translation/scoring with `--cpu-threads=N`
- Restoring corpus iteration after training is restarted
- N-best-list scoring in marian-scorer

### Fixed
- Deterministic data shuffling with specific seed for SQLite3 corpus storage
- Mini-batch fitting with binary search for faster fitting
- Better batch packing due to sorting


## [1.3.1] - 2018-02-04

### Fixed
- Missing final validation when done with training
- Differing summaries for marian-scorer when used with multiple GPUs

## [1.3.0] - 2018-01-24

### Added
- SQLite3 based corpus storage for on-disk shuffling etc. with `--sqlite`
- Asynchronous maxi-batch preloading
- Using transpose in SGEMM to tie embeddings in output layer

## [1.2.1] - 2018-01-19

### Fixed
- Use valid-mini-batch size during validation with "translation" instead of
  mini-batch
- Normalize gradients with multi-gpu synchronous SGD
- Fix divergence between saved models and validated models in asynchronous SGD

## [1.2.0] - 2018-01-13

### Added
- Option `--pretrained-model` to be used for network weights initialization
  with a pretrained model
- Version number saved in the model file
- CMake option `-DCOMPILE_SERVER=ON`
- Right-to-left training, scoring, decoding with `--right-left`

### Fixed
- Fixed marian-server compilation with Boost 1.66
- Fixed compilation on g++-4.8.4
- Fixed compilation without marian-server if openssl is not available

## [1.1.3] - 2017-12-06

### Added
- Added back gradient-dropping

### Fixed
- Fixed parameters initialization for `--tied-embeddings` during translation

## [1.1.2] - 2017-12-05

### Fixed
- Fixed ensembling with language model and batched decoding
- Fixed attention reduction kernel with large matrices (added missing
  `syncthreads()`), which should fix stability with large batches and beam-size
  during batched decoding

## [1.1.1] - 2017-11-30

### Added
- Option `--max-length-crop` to be used together with `--max-length N` to crop
  sentences to length N rather than omitting them.
- Experimental model with convolution over input characters

### Fixed
- Fixed a number of bugs for vocabulary and directory handling

## [1.1.0] - 2017-11-21

### Added
- Batched translation for all model types, significant translation speed-up
- Batched translation during validation with translation
- `--maxi-batch-sort` option for `marian-decoder`
- Support for CUBLAS_TENSOR_OP_MATH mode for cublas in cuda 9.0
- The "marian-vocab" tool to create vocabularies

## [1.0.0] - 2017-11-13

### Added
- Multi-gpu validation, scorer and in-training translation
- summary-mode for scorer
- New "transformer" model based on [Attention is all you
  need](https://arxiv.org/abs/1706.03762)
- Options specific for the transformer model
- Linear learning rate warmup with and without initial value
- Cyclic learning rate warmup
- More options for learning rate decay, including: optimizer history reset,
  repeated warmup
- Continuous inverted square root decay of learning (`--lr-decay-inv-sqrt`)
  rate based on number of updates
- Exposed optimizer parameters (e.g. momentum etc. for Adam)
- Version of deep RNN-based models compatible with Nematus (`--type nematus`)
- Synchronous SGD training for multi-gpu (enable with `--sync-sgd`)
- Dynamic construction of complex models with different encoders and decoders,
  currently only available through the C++ API
- Option `--quiet` to suppress output to stderr
- Option to choose different variants of optimization criterion: mean
  cross-entropy, perplexity, cross-entropy sum
- In-process translation for validation, uses the same memory as training
- Label Smoothing
- CHANGELOG.md
- CONTRIBUTING.md
- Swish activation function default for Transformer
  (https://arxiv.org/pdf/1710.05941.pdf)

### Changed
- Changed shape organization to follow numpy.
- Changed option `--moving-average` to `--exponential-smoothing` and inverted
  formula to `s_t = (1 - \alpha) * s_{t-1} + \alpha * x_t`, `\alpha` is now
  `1-e4` by default
- Got rid of thrust for compile-time mathematical expressions
- Changed boolean option `--normalize` to `--normalize [arg=1] (=0)`. New
  behaviour is backwards-compatible and can also be specified as
  `--normalize=0.6`
- Renamed "s2s" binary to "marian-decoder"
- Renamed "rescorer" binary to "marian-scorer"
- Renamed "server" binary to "marian-server"
- Renamed option name `--dynamic-batching` to `--mini-batch-fit`
- Unified cross-entropy-based validation, supports now perplexity and other CE
- Changed `--normalize (bool)` to `--normalize (float)arg`, allow to change
  length normalization weight as `score / pow(length, arg)`

### Removed
- Temporarily removed gradient dropping (`--drop-rate X`) until refactoring.
