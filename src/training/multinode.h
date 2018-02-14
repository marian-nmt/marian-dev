/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#pragma once

#if MPI_FOUND
#include <mpi.h>
#endif

namespace marian {

struct Multinode {
  virtual ~Multinode() {}

  virtual void begin_forward() = 0;
  virtual void begin_backward() = 0;
  virtual void begin_update() = 0;
  virtual void end_iteration() = 0;

  virtual bool save() = 0;
  virtual void finished() = 0;
};

#if MPI_FOUND
class RMA : public Multinode {
  bool moved;

  float* val, * grad;
  MPI_Aint n;

  int me, ranks;
  MPI_Aint span;
  MPI_Win val_window, grad_window;
  bool mpi_rma_unified_memory_model;
  bool push;

  float* buf;

  void lock(int lock_type, int rank, int assert, MPI_Win window);
  void unlock(int rank, MPI_Win window);

  void fetch_unowned_parameters();
  void fetch_owned_gradients();
  void push_unowned_gradients();

  void begin_gradient_update();
  void end_gradient_update();

  void begin_parameter_update();
  void end_parameter_update();

  public:
  enum GradientAction { PULL, PUSH };

  RMA(const RMA&) = delete;
  RMA& operator=(const RMA&) = delete;

  RMA(float* val, float* grad, MPI_Aint n, GradientAction push = PULL);
  RMA(RMA&& o);
  ~RMA();

  RMA& operator=(RMA&& o);

  void begin_forward();
  void begin_backward();
  void begin_update();
  void end_iteration();

  bool save();
  void finished();
};
#endif

}
