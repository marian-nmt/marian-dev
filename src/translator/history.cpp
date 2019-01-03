#include "history.h"

namespace marian {

History::History(size_t lineNo, float alpha, float wp, float xmlPenalty)
    : lineNo_(lineNo), alpha_(alpha), wp_(wp), xmlPenalty_(xmlPenalty) {}
}  // namespace marian
