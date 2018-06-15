// -*- mode: c++; tabs-indent: nil; tab-width: 2 -*-
#include "scheduler.h"
#include <fmt/format.h>
namespace marian {

void
Scheduler::
trnReportProgress() {
  std::string fmt = options_->get<std::string>("log-template");
  
  float words_per_sec = state_->wordsDisp / std::stof(timer.format(5,"%w"));
  boost::timer::cpu_times elapsed = timer.elapsed();
  // cpu_times::elapsed() reports in some integer type nanoseconds
  // -> convert to float representing seconds here:
  float wall_time = double(elapsed.wall)/1000000000;
  float user_time = double(elapsed.user)/1000000000;
  float system_time = double(elapsed.system)/1000000000;
  LOG(info,fmt,
      // list and initialize all recognized variables below:
      fmt::arg("EPOCH", state_->epochs),  
      fmt::arg("BATCH", state_->batches),
      fmt::arg("SAMPLE", state_->samples),
      fmt::arg("uTIME", user_time),
      fmt::arg("wTIME", wall_time),
      fmt::arg("sTIME", system_time),
      fmt::arg("WPS", words_per_sec),
      fmt::arg("WORDS_PER_SECOND", words_per_sec),
      fmt::arg("LRATE", state_->eta),
      // shortcuts:
      fmt::arg("E", state_->epochs),  
      fmt::arg("B", state_->batches),
      fmt::arg("S", state_->samples),
      fmt::arg("w", wall_time),
      fmt::arg("u", user_time),
      fmt::arg("s", system_time),
      fmt::arg("W", words_per_sec),
      fmt::arg("L", state_->eta));
}

