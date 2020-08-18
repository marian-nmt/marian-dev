#pragma once
#include <csignal>

// SIGNAL HANDLING

// The Marian signal handler setSignalFlag is a general purpose signal handler
// that sets a global flag upon receiving a signal (with SIGNAL No. < 32) in line 
// with the recommendations for signal handling in the SEI CERT C Coding Standard, specifically
// - SIG30-C: https://wiki.sei.cmu.edu/confluence/display/c/SIG30-C.+Call+only+asynchronous-safe+functions+within+signal+handlers
// - SIG31-C: https://wiki.sei.cmu.edu/confluence/display/c/SIG31-C.+Do+not+access+shared+objects+in+signal+handlers
// Usage: 
// - install the signal handler for a specific signal with signal(SIGNAL, setSignalFlag), 
//   e.g. signal(SIGTERM, setSignalFlag)
// - check the flag wherever appropriate with getSignalFlag(SIGNAL), 
//   e.g. getSignalFlag(SIGTERM)
// 
// This mechanism is currently used in marian training to ensure a graceful shutdown after receiving 
// SIGTERM, saving the current state of training before exiting. This behavior is particularly desirable
// when training on clusters with time limits on computeslots, e.g., on certain clusters managed by slurm. 
// Slurm can be asked to send a (custom) warning signal to a process at a certain time priopr to the 
// hard end of the time slot.

namespace marian {
bool getSignalFlag(int sig); // return true if sig was received, false otherwise
void setSignalFlag(int sig); // custom handler (set flag) for sig
} // end of namespace marian
