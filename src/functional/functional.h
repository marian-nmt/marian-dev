#pragma once

#include "functional/operands.h"
#include "functional/predicates.h"

namespace marian {
  namespace functional {

    template <int N>
    using ref = Assignee<N>;

    __D__ static ref<1> _1;
    __D__ static ref<2> _2;
    __D__ static ref<3> _3;
    __D__ static ref<4> _4;
    __D__ static ref<5> _5;
    __D__ static ref<6> _6;
    __D__ static ref<7> _7;
    __D__ static ref<8> _8;
    __D__ static ref<9> _9;

    __D__ static C<0> _0c;
    __D__ static C<1> _1c;
    __D__ static C<2> _2c;
    __D__ static C<-1> _1cneg;
    __D__ static C<-2> _2cneg;
  }
}