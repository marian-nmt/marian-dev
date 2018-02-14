/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#include <algorithm>
#include <stdexcept>
#include "training/multinode.h"
#include "common/logging.h"

namespace marian {

#if MPI_FOUND
template <typename T>
static T* rma_malloc(MPI_Aint size) {
  void* p;
  MPI_Alloc_mem(size, MPI_INFO_NULL, &p);
  return reinterpret_cast<T*>(p);
}

static void rma_free(void* p) {
  MPI_Free_mem(p);
}

void RMA::lock(int lock_type, int rank, int assert, MPI_Win window) {
  if (rank != me || !mpi_rma_unified_memory_model) {
    MPI_Win_lock(lock_type, rank, assert, window);
  }
}

void RMA::unlock(int rank, MPI_Win window) {
  if (rank != me || !mpi_rma_unified_memory_model) {
    MPI_Win_unlock(rank, window);
  }
}

RMA::RMA(float* val, float* grad, MPI_Aint n, RMA::GradientAction push)
  : moved(false), val(val), grad(grad), n(n), push(push) {
  MPI_Init(nullptr, nullptr); // Alternatively, have a singleton manage this

  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  span = n / ranks;

  MPI_Win_create(val, n, sizeof *val, MPI_INFO_NULL, MPI_COMM_WORLD, &val_window);
  MPI_Win_create(grad, n, sizeof *grad, MPI_INFO_NULL, MPI_COMM_WORLD, &grad_window);

  /* The MPI RMA separate memory model distinguishes between a rank's view of
   * a local window, and the view of the same window observed by other ranks.
   * When using asynchronous 'passive target synchronisation', we must lock
   * and unlock to propagate changes in both directions. The unified memory
   * model does not draw this distinction, so we may omit local locks.
   */
  int* model;
  int success;
  MPI_Win_get_attr(val_window, MPI_WIN_MODEL, &model, &success);
  mpi_rma_unified_memory_model = success && *model == MPI_WIN_UNIFIED;

  LOG(multinode)->info("MPI RMA {} memory model, {} gradient at end of each iteration",
    mpi_rma_unified_memory_model ? "unified" : "separate", push ? "push" : "pull");

  buf = rma_malloc<float>((span + ranks-1) * sizeof *buf);
}

RMA::~RMA() {
  if (moved) {
    return;
  }

  MPI_Win_free(&val_window);
  MPI_Win_free(&grad_window);

  rma_free(buf);

  MPI_Finalize(); // Alternatively, have a singleton manage this
}

RMA::RMA(RMA&& o)
  : moved(o.moved), n(o.n), me(o.me), ranks(o.ranks), val_window(o.val_window), grad_window(o.grad_window), push(o.push), buf(o.buf) {
  o.moved = true;
}

RMA& RMA::operator=(RMA&& o) {
  moved = o.moved;
  n = o.n;
  me = o.me;
  ranks = o.ranks;
  val_window = o.val_window;
  grad_window = o.grad_window;
  buf = o.buf;

  o.moved = true;
  return *this;
}

static const char* message = "Tried to operate moved-from instance of class RMA";

void RMA::fetch_unowned_parameters() {
  if (moved) {
    throw std::runtime_error(message);
  }

  for (int target = 0; target < ranks; ++target) {
    if (target == me) {
      continue;
    }

    MPI_Aint begin = target * span;
    MPI_Aint end = target < ranks-1 ? begin + span : n;
    int length = end - begin;

    lock(MPI_LOCK_SHARED, target, 0, val_window);
    MPI_Get(buf, length, MPI_FLOAT, target, begin, length, MPI_FLOAT, val_window);
    unlock(target, val_window);

    std::copy(buf, buf + length, val + begin);
  }
}

void RMA::fetch_owned_gradients() {
  if (moved) {
    throw std::runtime_error(message);
  }

  MPI_Aint begin = me * span;
  MPI_Aint end = me < ranks-1 ? begin + span : n;
  int length = end - begin;

  for (int target = 0; target < ranks; ++target) {
    if (target == me) {
      continue;
    }

    lock(MPI_LOCK_SHARED, target, 0, grad_window);
    MPI_Get(buf, length, MPI_FLOAT, target, begin, length, MPI_FLOAT, grad_window);
    unlock(target, grad_window);

    for (int i = begin; i < end; ++i) {
      grad[i] += buf[i - begin];
    }
  }
}

void RMA::push_unowned_gradients() {
  if (moved) {
    throw std::runtime_error(message);
  }

  for (int target = 0; target < ranks; ++target) {
    if (target == me) {
      continue;
    }

    MPI_Aint begin = target * span;
    MPI_Aint end = target < ranks-1 ? begin + span : n;
    int length = end - begin;

    std::copy(grad + begin, grad + end, buf);

    lock(MPI_LOCK_EXCLUSIVE, target, 0, grad_window);
    MPI_Accumulate(buf, length, MPI_FLOAT, target, begin, length, MPI_FLOAT, MPI_SUM, grad_window);
    unlock(target, grad_window);
  }
}

void RMA::begin_gradient_update() {
  if (moved) {
    throw std::runtime_error(message);
  }

  lock(MPI_LOCK_EXCLUSIVE, me, 0, grad_window);
}

void RMA::end_gradient_update() {
  if (moved) {
    throw std::runtime_error(message);
  }

  unlock(me, grad_window);
}

void RMA::begin_parameter_update() {
  if (moved) {
    throw std::runtime_error(message);
  }

  lock(MPI_LOCK_EXCLUSIVE, me, 0, val_window);
  lock(MPI_LOCK_SHARED, me, 0, grad_window);
}

void RMA::end_parameter_update() {
  if (moved) {
    throw std::runtime_error(message);
  }

  unlock(me, grad_window);
  unlock(me, val_window);
}

void RMA::begin_forward() {
  fetch_unowned_parameters();
}

void RMA::begin_backward() {
  begin_gradient_update();
}

void RMA::begin_update() {
  end_gradient_update();

  if (push) {
    push_unowned_gradients();
  } else {
    fetch_owned_gradients();
  }

  begin_parameter_update();
}

void RMA::end_iteration() {
  end_parameter_update();
}

bool RMA::save() {
  return me == 0;
}

void RMA::finished() {
  MPI_Barrier(MPI_COMM_WORLD);
  if (save()) {
    fetch_unowned_parameters();
  }
}
#endif

}
