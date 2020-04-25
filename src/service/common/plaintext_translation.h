#pragma once
#include <future>
#include <vector>
#include <string>
#include <map>
#include "common/definitions.h"
#include "3rd_party/ssplit-cpp/src/ssplit/ssplit.h"
#include "translation_options.h"
namespace marian {
namespace server {

class TranslationService;
class Job;

class PlainTextTranslation {
public:
  typedef  ug::ssplit::SentenceStream::splitmode splitmode;
private:
  std::vector<std::future<Ptr<Job const>>> pending_jobs_;
  std::vector<Ptr<Job const>> finished_jobs_;
  splitmode smode_;
  bool ends_with_eol_char_{false};

public:
  // @TODO: implement a custom iterator that allows iteration
  // over partially finished PlainTextTranslation, to accommodate
  // translating large amounts of text. We can't just use an
  // iterator over finished_jobs_, because jobs may not have finished
  // by the time someone tries to access the results.

  PlainTextTranslation(std::string const& input,
                       TranslationService& service,
                       const TranslationOptions& topts,
                       splitmode const& smode);

  Ptr<const Job> await(const size_t i);

  size_t size() const;

  // void await() {
  //   for (size_t i = 0; i < finished_jobs.size(); ++i) {
  //     if (finished_jobs_[i] == NULL) {
  //       finished_jobs_[i] = pending_jobs_[i].get();
  //     }
  //   }
  // }

  std::string toString();
};
}} // end of namespace marian::server
