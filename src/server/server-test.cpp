// -*- mode: c++; indent-tabs-mode: nil; tab-width: 2 -*-
#include "marian.h"
#include "translator/beam_search.h"
#include "translator/output_printer.h"
#include "common/timer.h"
#include "common/utils.h"

#include "translation_service.h"
#include <sstream>
#include <mutex>

using namespace marian;

class Publisher {
  // OutputPrinter printer_;
  Ptr<Vocab const> target_vocab_;
  bool const r2l_decoded_;
  size_t const beam_size_;
  std::mutex mutex_;
public:
  Publisher(Ptr<Options const> opts, Ptr<Vocab const> vcb)
    : target_vocab_(vcb)
    , r2l_decoded_(opts->get<bool>("right-left"))
    , beam_size_(opts->get<size_t>("beam-size"))
  { }

public:
  void operator()(uint64_t jid, Ptr<History const> h) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto& nbl = h->NBest(beam_size_);
    size_t i = 0; // idx. of best non-empty hyp. (Marian issue #462)
    for (;i < nbl.size() && std::get<0>(nbl[i]).size() == 0; ++i);
    auto snt = std::get<0>(nbl[i%nbl.size()]);
    if (r2l_decoded_)
      std::reverse(snt.begin(),snt.end());
    std::string translation = target_vocab_->decode(snt);
    printf("[%zu] %s\n", jid, translation.c_str());
  }
};


int main(int argc, char* argv[])
{

  // Initialize translation task
  auto options = parseOptions(argc, argv, cli::mode::server, true);
  auto service = New<server::TranslationService<BeamSearch>>(options);
  service->start();
  std::string line;
  // auto publish = [](uint64_t ejid, Ptr<History const> h) {
  //   // ejid: "external" job id by the client
  //   std::ostringstream toptrans, nbest;

  //   std::cout << "Finished job " << ejid << std::endl;
  // };

  Publisher publish(options, service->vocab(-1));
  uint32_t linectr = 0;

  // publisher isn't copyable because of mutex, so we have to
  // wrap it in a lambda
  auto callback = [publish](uint64_t jid, Ptr<History const> h) { publish(jid,h); })
  while (getline(std::cin,line)) {
    // uint64_t ijid =
    service->push(++linectr, line, callback);
    // std::cout << "[" << ijid << "] " << line << std::endl;
  }

}
