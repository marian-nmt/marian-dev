#include "common/definitions.h"
#include "common/intrusive_ptr.h"
#include "graph/expression_graph.h"

#include <functional>

namespace marian {

// This class allows for simpler caching of Expr objects and automatic checking if the
// cached Expr needs to be updated/recreated.
class CachedExpr {
  private:
    ENABLE_INTRUSIVE_PTR(CachedExpr);

    Expr cachedKey_{nullptr};
    Expr cachedValue_{nullptr};

    typedef std::function<Expr(Expr)> ApplyFunT;
    typedef std::function<bool(Expr, Expr)> EqualFunT;

    UPtr<ApplyFunT> applyFun_; // function that creates the cached result
    UPtr<EqualFunT> equalFun_; // function that checks if the input changed. If yes,
                               // the `apply_` functions gets reapplied and the new result
                               // is cached.

  public:
    // No functors are given; they will have to supplied when calling `apply`.
    CachedExpr() {};

    // No apply functor is given; it will have to supplied when calling `apply`.
    CachedExpr(EqualFunT equalFun)
    : equalFun_(new EqualFunT(equalFun)) {};

    // Both functors are given, and will be used by default. They can however be overriden
    // if supplied directly in `apply`.
    CachedExpr(ApplyFunT applyFun, EqualFunT equalFun)
    : applyFun_(new ApplyFunT(applyFun)), equalFun_(new EqualFunT(equalFun)) {};

    // lazily executes the factory `applyFun` if no value is cached or `equalFun` indicates that the input has changed.
    Expr apply(Expr key, ApplyFunT applyFun, EqualFunT equalFun) {
      if(!cachedKey_ || !equalFun(cachedKey_, key)) {
        cachedKey_ = key;
        cachedValue_ = applyFun(key);
      }
      return cachedValue_;
    }

    // lazily executes the factory `applyFun` if a equality check that has been passed to the constructor
    // indicates that the input has changed.
    Expr apply(Expr key, ApplyFunT applyFun) {
      ABORT_IF(!equalFun_, "Equality check has not been passed to constructor");
      return apply(key, applyFun, *equalFun_);
    }

    // lazily executes a factory if a equality check indicates that the input has changed. Both,
    // the factory and the equality check have to have been passed to the constructor.
    Expr apply(Expr key) {
      ABORT_IF(!equalFun_, "Equality check has not been passed to constructor");
      ABORT_IF(!applyFun_, "Apply factory has not been passed to constructor");
      return apply(key, *applyFun_, *equalFun_);
    }

    // clears any cached values, calling apply after this will trigger recomputation.
    void clear() {
      cachedKey_   = nullptr;
      cachedValue_ = nullptr;
    }
};

}
