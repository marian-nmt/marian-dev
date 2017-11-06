#pragma once

#include "gpu/defs.h"
#include "functional/operands.h"
#include "functional/constants.h"
#include "gpu/primitives.h"

namespace marian {
  namespace functional {

    template <class X,
              class Accumulator = decltype(accumulator::plus),
              class Zero = decltype(zero)>
    struct ReduceRow {
      X x;
      Accumulator acc;
      Zero acc_zero;

      bool done{false};
      float cache{0};

      __HD__ ReduceRow() {}

      __HD__ ReduceRow(X x_,
                       Accumulator acc_ = accumulator::plus,
                       Zero acc_zero_ = zero)
      : x(x_), acc(acc), acc_zero(acc_zero_) {}


      template <typename T>
      __DI__ T operator()(gpu::Tensor<T> row) {
        if(!done) {
          cache = gpu::reduce_row(row.shape().back(), row, x, acc, acc_zero);
          done = true;
        }
        return cache;
      }

      std::string to_string() {
        return "ReduceRow<" + x.to_string()
          + "," + acc.to_string()
          + "," + acc_zero.to_string()
          + ">";
      }
    };

    template <class X,
              class Acc = decltype(accumulator::plus),
              class Zero = decltype(zero)>
    __DI__ ReduceRow<X, Acc, Zero> reduce_row(X x,
                                       Acc acc = accumulator::plus,
                                       Zero acc_zero = zero) {
      return ReduceRow<X, Acc, Zero>(x, acc, acc_zero);
    }

    template <class X>
    __DI__ auto sum_row(X x)->decltype(reduce_row(x)) {
      return reduce_row(x);
    }

    template <class X>
    __DI__ auto max_row(X x)->decltype(reduce_row(x, accumulator::max, first)) {
      return reduce_row(x, accumulator::max, first);
    }



    template <int N, typename T>
    __DI__ C<N>& reduce(C<N>& c, gpu::Tensor<T>& row) {
      return c;
    }

    template <int N, typename T>
    __DI__ Var<N>& reduce(Var<N>& var, gpu::Tensor<T>& row) {
      return var;
    }

    template <class X, class Acc, class Zero, typename T>
    __DI__ Capture reduce(ReduceRow<X, Acc, Zero>& r, gpu::Tensor<T>& row) {
      return Capture(r(row));
    }

    template <template <class> class F, class X, typename T>
    __DI__ auto reduce(F<X>& f, gpu::Tensor<T>& row)->decltype(F<decltype(reduce(f.x, row))>(f.x)) {
      return F<decltype(reduce(f.x, row))>(f.x);
    }

    template <template <class, class> class F, class X, class Y, typename T>
    __DI__ auto reduce(F<X, Y>& f, gpu::Tensor<T>& row)
    ->decltype(
      F<decltype(reduce(f.x, row)), decltype(reduce(f.y, row))>(
        reduce(f.x, row),
        reduce(f.y, row)
      )
    ) {
      return
        F<decltype(reduce(f.x, row)), decltype(reduce(f.y, row))>(
          reduce(f.x, row),
          reduce(f.y, row)
        );
    }

/******************************************************************************/

  }
}