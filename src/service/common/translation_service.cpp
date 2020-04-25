#include "marian.h"
#include "common/logging.h"
#include "plaintext_translation.h"
#include "translation_service.h"
namespace marian {
namespace server {

// Auxiliary function for loading vocabs.
// This should actually go into vocab.*
// Also it should be merged with the loadOrCreate code in corpus_base.cpp
// and refactored as a separate function (that then goes into vocab.*).
std::vector<Ptr<const Vocab> >
loadVocabularies(Ptr<Options> options) {
  // @TODO: parallelize vocab loading for faster startup
  auto vfiles = options->get<std::vector<std::string>>("vocabs");
  // with the current setup, we need at least two vocabs: src and trg
  ABORT_IF(vfiles.size() < 2, "Insufficient number of vocabularies.");
  std::vector<Ptr<Vocab const> > vocabs(vfiles.size());
  std::unordered_map<std::string,Ptr<Vocab>> vmap;
  for (size_t i = 0; i < vocabs.size(); ++i) {
    auto m = vmap.emplace(std::make_pair(vfiles[i],Ptr<Vocab>()));
    if (m.second) { // new: load the vocab
      m.first->second = New<Vocab>(options, i);
      m.first->second->load(vfiles[i]);
    }
    vocabs[i] = m.first->second;
  }
  return vocabs;
}

TranslationService::TranslationService(Ptr<Options> options)
  : options_(options) {
  auto ssplit_prefix_file = options_->get<std::string>("ssplit-prefix-file","");
  if (ssplit_prefix_file.size()) {
    ssplit_prefix_file = cli::InterpolateEnvVars(ssplit_prefix_file);
    LOG(info, "Loading protected prefixes for sentence splitting from {}",
        ssplit_prefix_file);
    ssplit_.load(ssplit_prefix_file);
  }
  else {
    LOG(warn, "Missing list of protected prefixes for sentence splitting. "
        "Set with --ssplit-prefix-file.");
  }
  chooseDevice_(options);
}

TranslationService::~TranslationService() {
  stop();
}


// Callback from worker for each finished job
void TranslationService::callback_(Ptr<const History> h) {
  // This function is called by the workers once translations are available.

  JobEntry entry;
  { // remove the job / promise pair from the pool of scheduled jobs
    std::lock_guard<std::mutex> lock(lock_);
    auto m = scheduled_jobs_.find(h->getLineNum());
    if (m == scheduled_jobs_.end()) {
      // i.e., if job was cancelled (not yet implemented)
      return;
    }
    entry = std::move(m->second);
    scheduled_jobs_.erase(m);
  }

  // extract translations from history and fulfil the promise
  entry.first->finish(h, isRight2LeftDecoder(), *vocabs_.back());
  entry.first->callback(entry.first);
  entry.second.set_value(entry.first);
}

// When run in a docker container with nvidia docker, CUDA may or may
// not be available, depending on how the Docker container is run, i.e.,
// mapping host devices into the container via --gpus or not.
// chooseDevice() checks if CUDA is available and automatically switches
// to CPU mode if not. Without this check, Marian will crash unless
// --cpu-threads is set explicitly.
void TranslationService::chooseDevice_(Ptr<Options> options){
#ifdef __CUDA_ARCH__
  if (options->get<int>("cpu_threads",0) > 0)
    return; // nothing to worry about, user wants to use CPU anyway
  int ngpus;
  cudaError_t err = cudaGetDeviceCount(&ngpus);
  if (err != cudaSuccess){
    size_t nproc = std::thread::hardware_concurrency();
    size_t num_workers = options->get<size_t>("max_workers",nproc);
    LOG(warn, "NO GPU available, using CPU instead. "
        "Setting --cpu-threads to {}", num_workers);
    options->set("cpu_threads",num_workers);
  }
  if (options->get<int>("cpu_threads",0)){
    // use mini-batch size 1 if running on CPU
    options->set<int>("mini_batch",1);
  }
#endif
}


void TranslationService::stop() {
  for (auto& w: workers_) w->stop();
  for (auto& w: workers_) w->join();
}


std::pair<uint64_t, std::future<Ptr<const Job>>>
TranslationService::
push(uint64_t ejid,
     const std::string& input,
     const TranslationOptions* topts/*=NULL*/,
     size_t const priority/*=0*/, // priority handling currently not implemented
     std::function<void (Ptr<Job> j)> callback /* =[=](Ptr<Job> j){return;} */) {
  auto job = New<Job>(ejid, input, topts ? *topts : dflt_topts_, priority);
  if (input.empty()){ // return empty result immediately
    std::promise<Ptr<Job const>> prom;
    prom.set_value(job);
    return std::make_pair(job->unique_id,prom.get_future());
  }
  else if (!jq_->push(job)) {
    job->error = New<Error>("Could not push to Queue.");
    std::promise<Ptr<Job const>> prom;
    prom.set_value(job);
    return std::make_pair(job->unique_id,prom.get_future());
  }
  job->callback = callback;
  JobEntry* entry;
  {
    std::lock_guard<std::mutex> lock(lock_);
    entry = &scheduled_jobs_[job->unique_id];
  }
  entry->first = job;
  return std::make_pair(job->unique_id, entry->second.get_future());
}

Ptr<Vocab const> TranslationService::vocab(int i) const {
  if (i < 0) i += vocabs_.size();
  return vocabs_.at(i);
}

bool
TranslationService::isRight2LeftDecoder() const {
  return options_->get<bool>("right-left");
}

Ptr<PlainTextTranslation>
TranslationService::
translate(std::string const& input,
          ssplitmode const smode/* = ssplitmode::wrapped_text*/){
  // @TODO: old, needs revision or elimination
  Ptr<PlainTextTranslation> ret;
  ret.reset(new PlainTextTranslation(input, *this, dflt_topts_, smode));
  return ret;
}

ug::ssplit::SentenceStream
TranslationService::
createSentenceStream(std::string const& input,
                     ug::ssplit::SentenceStream::splitmode const& mode)
{
  return std::move(ug::ssplit::SentenceStream(input, this->ssplit_, mode));
}

ug::ssplit::SentenceStream::splitmode
string2splitmode(const std::string& m, bool throwOnError/*=false*/){
  typedef ug::ssplit::SentenceStream::splitmode splitmode;
  // @TODO: throw Exception on error
  if (m == "sentence" || m == "Sentence")
    return splitmode::one_sentence_per_line;
  if (m == "paragraph" || m == "Paragraph")
    return splitmode::one_paragraph_per_line;
  if (m != "wrapped_text" || m != "WrappedText" || m != "wrappedText") {
    LOG(warn,"Ignoring unknown text input format specification: {}.", m);
  }
  return splitmode::wrapped_text;
}

}} // end of namespace marian::server
