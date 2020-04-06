#pragma once
#include <csignal>

// SIGNAL HANDLING

// The Marian signal handlers set global flags that thread can
// consider when a signal is received. This can be used for a graceful
// shutdown instead of a hard abandonment, e.g.  after receiving
// SIGTERM during training.

// When SIGTERM is received, the global (static member) flag sigterm_
// (false by default) is set to true by signalHandler(). When sigterm_
// is true, keepGoing() returns false, and the current state of
// training models is saved prior to exiting.  This functionality is
// helpful when training on clusters with time limits on compute
// slots, e.g., on s clusters managed by slurm. Slurm can be asked to
// sending a (custom) warning signal to a process at a given point in
// time prior to the hard "time's up".
//
// Correspondingly, fetchBatches in the batch generator checks the flag
// frequently and quits after the overall process receives a SIGTERM.


namespace marian {
bool getSignalFlag(int sig); // return true if sig was received, false otherwise
void setSignalFlag(int sig); // set custom handler (set flag) for sig
}
