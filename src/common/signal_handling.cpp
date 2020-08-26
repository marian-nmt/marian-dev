#include "common/logging.h"
#include "signal_handling.h"

// We use signal() here instead of the usual strong recommendation for
// using sigaction, which apparently is not available for Windows (cf.
// https://stackoverflow.com/questions/231912/what-is-the-difference-between-sigaction-and-signal).

namespace marian{
volatile std::sig_atomic_t sigflags_{0};
volatile std::sig_atomic_t gracefulExitRequested_{0};

bool getSignalFlag(const int sig) {
  // sig_atomic_t has 32 bits. We don't accommodate signals beyond that.
  ABORT_IF(sig >= 32, "Signal out of range (must be < 32, is {}).", sig);
  return sigflags_ & (1<<sig);
}

void requestGracefulExit(int sig) {
  setSignalFlag(sig);         // keep track of triggering signal
  gracefulExitRequested_ = 1; // set flag to exit gracefully
}

bool gracefulExitRequested() {
  return gracefulExitRequested_;
}

void setSignalFlag(int sig) {
  // sig_atomic_t has 32 bits. We don't accommodate signals beyond that.
  if (sig < 32) sigflags_ |= (1<<sig);
}

}
