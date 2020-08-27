#include "common/logging.h"
#include "signal_handling.h"

// The simplest (and recommended) way to handle signals is to simply set a flag
// in the signal handler and check that flag later.
//
// We provide setSignalFlag as the most generic signal handler.
// This handler which uses a single sig_atomic_t as a bit field.
// On Linux, sig_atomic_t is equivalent to a signed int, theoretically
// providing 32 binary flags; in practice, most likely signals for which we may
// want to install signal handlers are
// - SIGTERM (15): which by default signals the request for a graceful exit
//   (see also: https://qph.fs.quoracdn.net/main-qimg-1180ef2465c309928b02481f02580c6a)
// - SIGUSR1,SIGUSR2 (10,12): signals specifically reserved for custom use
// - SIGINT (2): interrupt from the console
// Just to be safe, we accommodate signals up to signal No. 30.
constexpr int maxSignalForSetSetSignalFlag{30};

// Make sure sig_atomic_t is large enough as a bit field for our purposes.
// That said, I'm not aware of any platform where this would be a problem.
static_assert(SIG_ATOMIC_MAX > (1U<<maxSignalForSetSetSignalFlag),
              "sig_atomic_type is too small for signal flags on this platform.");

namespace marian{
volatile std::sig_atomic_t sigflags_{0};
volatile std::sig_atomic_t gracefulExitRequested_{0};

void setSignalFlag(int sig) {
  sigflags_ |= (1<<sig);
}

bool getSignalFlag(const int sig) {
  ABORT_IF(sig > maxSignalForSetSignalFlag,
           "Signal out of range (must be < {}, is {}).", maxSignalForSetSignalFlag, sig);
  return sigflags_ & (1<<sig);
}

void requestGracefulExit(int sig) {
  setSignalFlag(sig);         // keep track of triggering signal
  gracefulExitRequested_ = 1; // set flag to exit gracefully
}

bool gracefulExitRequested() {
  return gracefulExitRequested_ == 1;
}

}
