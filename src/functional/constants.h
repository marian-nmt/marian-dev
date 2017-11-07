#pragma once

#include "functional/operands.h"
#include "functional/predicates.h"

namespace marian {
  namespace functional {

    template <int N>
    using var = Assignee<N>;

    __D__ static var<1> _1;
    __D__ static var<1> first;
    __D__ static var<1> same;

    __D__ static var<2> _2;
    __D__ static var<2> second;

    __D__ static var<3> _3;
    __D__ static var<3> third;

    __D__ static var<4> _4;
    __D__ static var<5> _5;
    __D__ static var<6> _6;
    __D__ static var<7> _7;
    __D__ static var<8> _8;
    __D__ static var<9> _9;

    __D__ static C<0> zero;
    __D__ static C<1> one;
    __D__ static C<2> two;

    namespace accumulator {

      __D__ static decltype(_1 += _2) plus;
      __D__ static decltype(_1 *= _2) mult;
      __D__ static decltype(_1 = max(_1, _2)) max;

    }
  }
}