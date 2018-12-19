#include "graph/expression_graph.h"
#include <sstream>

#include "tensors/tensor_operators.h"

namespace marian {

ExpressionGraph::ExpressionGraph(bool inference, bool optimized)
  : inferenceOnly_(inference),
    optimized_(optimized),
    backend_(nullptr) {}

void ExpressionGraph::setDevice(DeviceId deviceId, Ptr<Device> device) {
  if(!backend_) {
    backend_ = BackendByDeviceId(deviceId, Config::seed);
    params_ = New<Parameters>();
    params_->init(backend_);
    if(device)
      tensors_ = New<Tensors>(backend_, device);
    else
      tensors_ = New<Tensors>(backend_);
  }
}

Expr ExpressionGraph::dropout(float prob, const Shape& shape, Type valueType) {
  return constant(shape, inits::dropout(prob), valueType);
}
Expr ExpressionGraph::dropout(float prob, const Shape& shape) {
  return constant(shape, inits::dropout(prob), parameterType_);
}

void ExpressionGraph::checkNan(Tensor t) {
  ABORT_IF(throwNaN_, "Not implemented"); t;
  // ABORT_IF(throwNaN_ && IsNan(t), "Tensor has NaN");
}

void ExpressionGraph::save(std::vector<io::Item>& ioItems) {
  for(auto p : params()->getMap()) {
    std::string pName = p.first;

    if(!namespace_.empty()) {
      if(pName.substr(0, namespace_.size() + 2) == namespace_ + "::")
        pName = pName.substr(namespace_.size() + 2);
    }

    Tensor val = p.second->val();
    auto mem = val->memory();
    std::vector<char> bytes(mem->size());
    copy(backend_,
         mem->data<char>(),
         mem->data<char>() + mem->size(),
         bytes.data());

    io::Item item;
    item.name = pName;
    item.shape = val->shape();

    if(val->type() == Type::float32) {
      item.type = val->type();
      // Use the actual memory as this will be aligned and padded.
      // When memory mapping this is required. Shape keeps track of
      // tensor size. Saving to *.npz will cut to size.
      item.bytes.swap(bytes);
    } else if(val->type() == Type::float16) {
      // converting to float
      item.type = Type::float32;
      item.bytes.resize(val->size() * sizeOf(item.type));

      auto beg16 = (float16*)bytes.data();
      auto end16 = beg16 + val->size();
      auto beg32 = (float*)item.bytes.data();

      std::copy(beg16, end16, beg32);
    } else {
      ABORT("Other types are currently not supported");
    }
    
    ioItems.emplace_back(std::move(item));
  }
}

}  // namespace marian
