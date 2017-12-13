#pragma once

#include <vector>
#include <ostream>

#include "common/utils.h"
#include "data/vocab.h"
#include "translator/history.h"

namespace marian {

void Printer(Ptr<Config> options,
             Ptr<Vocab> vocab,
             Ptr<History> history,
             std::ostream& best1,
             std::ostream& bestn);
}
