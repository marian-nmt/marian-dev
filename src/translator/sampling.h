namespace marian {

namespace sampling {

// Prunning functions for sampling from the output distribution
// All functions take a logits tensor and return a tensor of the same shape and pruned values removed.
// The logits tensor is assumed to be in log-space (i.e. logprobs) and the returned tensor is also in log-space.
// The pruned distribution can be renormalized via logsoftmax to ensure that the sum of the probabilities is 1.
// However this doesn't matter much for sampling since the gumbel max trick works for unnormalized distributions.

// Prune logits via top-k pruning
Expr topkPruning(Expr scores, int k, bool normalize = false) {
  Expr val, idx;

  // note, for around k>200 topk is slower on the GPU than sorting and then selecting the top-k values
  std::tie(val, idx) = topk(scores, k, /*axis=*/-1, /*descending=*/true);
  if(normalize)
    val = logsoftmax(val); // renormalize via logsoftmax

  // Scatter gumbelled values back into logits to fill with usable values
  auto invalid = constant_like(scores, inits::fromValue(std::log(0.f)));
  return scatter(invalid, /*axis=*/-1, idx, val);
}

// Prune logits via nucleus pruning
Expr nucleusPruning(Expr scores, float threshold, bool normalize = false) {
  // normalization would make sense here since we compare against a meaningful threshold and
  // we don't know what other manipulations have been done to the logits before, but
  // leaving it to the user for now. We do set it to true in beam_search.cpp
  if(normalize)
    scores = logsoftmax(scores); // renormalize via logsoftmax

  // sort scores in descending order, this way we can use the cumulative sum to find the nucleus
  Expr val, idx;
  std::tie(val, idx) = sort(scores, /*axis=*/-1, /*descending=*/true);

  // logcumsumexp because we have logprobs, exclusive because we keep at least the first element
  // we can skip the numerical stability trick here since we are in log-space
  auto lcse     = logcumsumexp(val, /*axis=*/-1, /*reverse=*/false, /*exclusive=*/true, /*fast=*/true);

  // mask out all values that for which the cumulative sum is larger than the threshold (i.e. they are outside the nucleus)
  auto lcseMask = log(le(lcse, std::log(threshold)));
  val           = minimum(val, lcseMask); // mask out all values outside the nucleus

  if(normalize)
    val = logsoftmax(val); // renormalize via logsoftmax

  // scatter the masked values back into the correct positions (undo sorting)
  return scatter(scores, /*axis=*/-1, idx, val);
}

// Prune logits via epsilon pruning
Expr epsilonPruning(Expr scores, float epsilon, bool normalize = false) {
  // normalization would make sense here since we compare against a meaningful threshold and
  // we don't know what other manipulations have been done to the logits before
  if(normalize)
    scores = logsoftmax(scores); // renormalize via logsoftmax

  // make sure the epsilon is not larger than the largest value in the scores
  // otherwise we will mask out all values
  // equivalent to union of top-1 and log(epsilon)
  auto safeThreshold = minimum(max(scores, /*axis=*/-1), std::log(epsilon));

  // create mask for all values that are smaller than the epsilon
  auto logEpsMask   = log(ge(scores, safeThreshold)); // -inf for all values smaller than epsilon
  auto logEpsScores = minimum(scores, logEpsMask); // mask out all values smaller than epsilon

  if(normalize)
    logEpsScores = logsoftmax(logEpsScores); // renormalize after masking via logsoftmax
  return logEpsScores;
}

Expr gumbelMaxTrick(Expr scores, float temperature) {
  // scale scores by temperature
  if(temperature != 1.f)
    scores = scores / temperature;
  // add Gumbel noise to all values and renormalize via logsoftmax
  return logsoftmax(scores + constant_like(scores, inits::gumbel()));
}
} // namespace sampling

class DistModifier {
private:
  Ptr<ExpressionGraph> graph_;
  Ptr<Options> options_;
  Ptr<data::Shortlist> shortlist_;

  bool forceDecode_{false};

  bool sampling_{false};
  std::function<Expr(Expr, bool)> samplingFn_;

  Ptr<data::CorpusBatch> batch_;
  float invalidPathScore_;

  Expr forceBatch_;

  void lazyCreateForceBatch() {
    if(!forceBatch_) {
      // turn the batch into a cached tensor that lives in the computation graph
      std::vector<WordIndex> forceWords;
      for(auto& word : batch_->back()->data())
        forceWords.push_back(word.toWordIndex());
      int dimTime  = (int)batch_->back()->batchWidth();
      int dimBatch = (int)batch_->back()->batchSize();
      forceBatch_ = graph_->constant({1, dimTime, dimBatch, 1}, inits::fromVector(forceWords), Type::uint32); // [1, dimTime, dimBatch, 1]
    }
  }

public:
  DistModifier(Ptr<ExpressionGraph> graph, Ptr<Options> options, Ptr<data::CorpusBatch> batch, float invalidPathScore, Ptr<data::Shortlist> shortlist = nullptr) :
    graph_(graph),
    options_(options),
    shortlist_(shortlist),
    forceDecode_(options_->get<bool>("force-decode", false)),
    batch_(batch),
    invalidPathScore_(invalidPathScore) {

    forceDecode_ = forceDecode_ && batch->sets() > 1; // force-decoding if we have multiple sets in the batch

    // if we are force-decoding with a short list we need to set the forced token ids early
    if(shortlist_ && forceDecode_) {
      lazyCreateForceBatch();
      auto lsh = std::dynamic_pointer_cast<data::LSHShortlist>(shortlist_);
      ABORT_IF(!lsh, "Force-decoding not supported with shortlists other than LSH");
      ABORT_IF(!forceBatch_, "forceBatch_ is undefined??");
      Expr forceIndices = slice(forceBatch_, /*axis=*/-3, 0);   // [1, 1, dimBatch, 1]
      lsh->setForcedIndices(forceIndices);
    }

    if(options_->hasAndNotEmpty("output-sampling")) {
      sampling_ = true;
      auto samplingOpts = options_->get<std::vector<std::string>>("output-sampling", {});
      std::string samplingMethod = samplingOpts.size() > 0 ? samplingOpts[0] : "full";

      if(samplingMethod == "0") { // for backcompat with boolean values
        sampling_ = false;
        samplingMethod = "";
      } else if(samplingMethod == "1") { // for backcompat with boolean values
        sampling_ = true;
        samplingMethod = "full";
      }

      if(samplingMethod == "full") {
        float temperature = 1.f;
        if(samplingOpts.size() > 1)
          temperature = std::stof(samplingOpts[1]);

        LOG_ONCE(info, "Output sampling from the full softmax distribution with temperature {}", temperature);

        samplingFn_ = [temperature](Expr logits, bool normalize = false) {
          // full softmax sampling is just gumbel trick with temperature 1 and optional prior renormalization
          return sampling::gumbelMaxTrick(normalize ? logsoftmax(logits) : logits, temperature);
        };
      } else if(samplingMethod == "topk") {
        int topk = 10; // number of top-k values to sample from
        float temperature = 1.f;
        if(samplingOpts.size() > 1)
          topk = std::stoi(samplingOpts[1]);
        if(samplingOpts.size() > 2)
          temperature = std::stof(samplingOpts[2]);

        LOG_ONCE(info, "Output sampling via top-{} sampling with temperature {}", topk, temperature);

        samplingFn_ = [topk, temperature](Expr logits, bool normalize = false) {
          // top-k sampling is just gumbel trick with temperature 1 and top-k pruning
          return sampling::gumbelMaxTrick(sampling::topkPruning(logits, topk, normalize), temperature);
        };
      } else if(samplingMethod == "nucleus") {
        float threshold = 0.9f; // probability mass threshold of nucleus
        float temperature = 1.f;
        if(samplingOpts.size() > 1)
          threshold = std::stof(samplingOpts[1]);
        if(samplingOpts.size() > 2)
          temperature = std::stof(samplingOpts[2]);

        LOG_ONCE(info, "Output sampling via nucleus sampling with threshold {} temperature {}", threshold, temperature);

        samplingFn_ = [threshold, temperature](Expr logits, bool normalize = false) {
          // nucleus sampling is just gumbel trick with temperature 1 and nucleus pruning
          return sampling::gumbelMaxTrick(sampling::nucleusPruning(logits, threshold, normalize), temperature);
        };
      } else if(samplingMethod == "epsilon") {
        float eps = 0.02f; // mimimal probability of sampled token
        float temperature = 1.f;
        if(samplingOpts.size() > 1)
          eps = std::stof(samplingOpts[1]);
        if(samplingOpts.size() > 2)
          temperature = std::stof(samplingOpts[2]);

        LOG_ONCE(info, "Output sampling via epsilon sampling with eps {} and temperature {}", eps, temperature);

        samplingFn_ = [eps, temperature](Expr logits, bool normalize = false) {
          // epsilon sampling is just gumbel trick with temperature 1 and epsilon pruning
          return sampling::gumbelMaxTrick(sampling::epsilonPruning(logits, eps, normalize), temperature);
        };
      } else {
        ABORT("Unknown sampling method: {}", samplingMethod);
      }
    }
  }

  Expr force(Expr scores, int pos, int beamSize, std::vector<IndexType>& batchIndices) {
    // we check the last field of the batch for force-decoding content

    int dimTime = (int)batch_->back()->batchWidth();
    if(!forceDecode_ || pos >= dimTime) { // nothing to force-decode, just return original scores
      return scores;
    }

    LOG_ONCE(info, "Force-decoding with given prefixes");
    // if we get here, then we have to do force-decoding. We do this by "softly" modifying the scores and passing the
    // result to the normal top-k/beam search. "Softly" here means we add masking terms rather than making hard selections
    // which preserves the original tensor layout.
    // This allows for beam-search and batched force-decoding with different length prefixes in a batch
    // (way harder to do with actual index manipulation). We then return modified (masked) probabilities to the beam-search
    // which then continues as normal on the modified distribution.

    lazyCreateForceBatch();

    ABORT_IF(!forceBatch_, "forceBatch_ is undefined??");

    auto factoredVocab = batch_->front()->vocab()->tryAs<FactoredVocab>();
    ABORT_IF(factoredVocab, "Factored vocabularies are not supported for force-decoding");

    // if we remove batch entries during decoding (finished decoding) then adjust here
    if(forceBatch_->shape()[-2] != batchIndices.size())
      forceBatch_ = index_select(forceBatch_, -2, batchIndices);

    // get vocab index and probability for force-decoded tokens for the current time step
    Expr posIndices = slice(forceBatch_, /*axis=*/-3, pos);   // [1, 1, dimBatch, 1]

    Expr forceIndices = posIndices;
    if(shortlist_) {
      auto lsh = std::dynamic_pointer_cast<data::LSHShortlist>(shortlist_);
      ABORT_IF(!lsh, "Force-decoding not supported with shortlists other than LSH");
      // only get location for first beam slot, the other slots don't matter since we overwrite them later.
      lsh->setForcedIndices(forceIndices);
      forceIndices = lsh->tryForwardMap(forceIndices); // [1, 1, dimBatch, 1]
    }

    // select scores from first beam entry for force-decoding
    Expr b1stScores = slice(scores, /*axis=*/-4, 0); // [1, 1, dimBatch, dimVocab]
    Expr forceVals  = gather(b1stScores, /*axis=*/-1, forceIndices); // [1, 1, dimBatch, 1]

    // create dummy indices and values for beam entries other than the force-decoded value. This is required to ensure that the beam
    // does not collapse for hyps outside the forced hyps and can still do full beam-search once we finish force-decoding for a batch
    // entry. We initialize randomly (they are not going to be used anyway due to very low prob) and shift by 1 to have 0 at first postion.
    int dimVocab = scores->shape()[-1];
    auto graph = scores->graph();

    std::vector<IndexType> dummyIndicesVec(beamSize, 0);
    std::vector<float> dummyValsVec(beamSize, 0.f);
    for(int i = 1; i < beamSize; ++i) {
      // we use dimVocab - i - 1 to make sure that the dummy indices are different from the force-decoded index (last vocab indices)
      dummyIndicesVec[i] = dimVocab - i - 1;
      // we use invalidPathScore_ / (2.f + i) to make sure that the dummy values are very low and decrease with beam position
      dummyValsVec[i] = invalidPathScore_ / (2.f + i);
    }

    Expr dummyIndices = graph->constant({1, 1, 1, beamSize}, inits::fromVector(dummyIndicesVec), Type::uint32);
    Expr dummyVals    = graph->constant({1, 1, 1, beamSize}, inits::fromVector(dummyValsVec));

    // here we add the force-decoded entries back into the zeroed positions
    dummyIndices = cast(maximum(cast(dummyIndices, Type::float32), cast(forceIndices, Type::float32)), Type::uint32); // [1, 1, dimBatch, dimBeam]
    dummyVals    = dummyVals + forceVals; // [1, 1, dimBatch, dimBeam]

    // create a tensor of the same size as the original logits from the first beam entry, initialize with invalidPathScore and then scatter
    // the force-decoded and dummy values into the correct positions.
    Expr forcedScores = constant_like(b1stScores, inits::fromValue(invalidPathScore_)); // [1, 1, dimBatch, dimVocab]
    forcedScores = scatter(forcedScores, -1, dummyIndices, dummyVals); // [1, 1, dimBatch, dimVocab]

    // for entries that have finished force-decoding (the batch has eosId as vocab id) use the original logits for the whole batch entry
    // via interpolating by a selector. In marian eosId is used for padding, so this works everywhere and eos for unfinished hyps means
    // free decoding or sampling.
    WordIndex eosId = batch_->back()->vocab()->getEosId().toWordIndex();
    auto interpol = eq(cast(posIndices, scores->value_type()), (float)eosId);
    return interpol * scores + (1.f - interpol) * forcedScores;
  }

  Expr sample(Expr scores, bool normalize = false) {
    if(sampling_) {
      return samplingFn_(scores, normalize);
    } else { // no sampling
      return scores;
    }
  }
};

}