#include "graph/expression_graph.h"
#include <sstream>

#include "tensors/tensor_operators.h"
#include "tensors/cpu/sharp/packed_gemm.h"
#include "tensors/cpu/backend.h"

namespace marian {

ExpressionGraph::ExpressionGraph(bool inference)
    : inferenceOnly_(inference), backend_(nullptr) {}

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

Expr ExpressionGraph::dropoutMask(float prob, const Shape& shape) {
  return constant(shape, inits::dropout(prob));
}

void ExpressionGraph::checkNan(Tensor t) {
  ABORT_IF(throwNaN_, "Not implemented"); t;
  // ABORT_IF(throwNaN_ && IsNan(t), "Tensor has NaN");
}

void ExpressionGraph::save(std::vector<io::Item>& ioItems, const bool packsave) {
  for(auto p : params()->getMap()) {
    std::string pName = p.first;

    if(!namespace_.empty()) {
      if(pName.substr(0, namespace_.size() + 2) == namespace_ + "::")
        pName = pName.substr(namespace_.size() + 2);
    }

    ABORT_IF(p.second->val()->type() != Type::float32,
             "Only float32 supported at the moment");

    Tensor val = p.second->val();

    // save as packed format
    //bool packsave = true;
    if(packsave && (pName.find("_W") == pName.length() - 3 || pName.find("_W") == pName.length() - 2)) {
      using namespace marian::cpu::variant;

      GemmType gemmType = getBackend()->getGemmType();
      io::Item item;
      auto allocator = New<TensorAllocator>(New<cpu::Backend>(CPU0, 1));

      Tensor packedTensor;
      if (gemmType == GemmType::FbFp16Packed) {
        // packing information
        int nrow;
        int ncol;
        int kernel_ncol_blocks;
        int brow;
        int bcol;
        int last_brow;
        int nbrow;
        int nbcol;
        uint64_t packsize;

        PackInfoFp16(val->shape(),
                    false,
                    nrow,
                    ncol,
                    kernel_ncol_blocks,
                    brow,
                    bcol,
                    last_brow,
                    nbrow,
                    nbcol,
                    packsize);

        allocator->allocate(packedTensor, {1, (int32_t)packsize}, Type::uint8);

        //PackFp32
        PackFp16(packedTensor,
                val,
                false,
                nrow,
                ncol,
                kernel_ncol_blocks,
                brow,
                bcol,
                last_brow,
                nbrow,
                nbcol,
                packsize);
        io::Item item;
        item.name = pName;
        item.shape = {(int32_t)packsize}; // val->shape();
        item.type = Type::uint8;
      } else if (gemmType == GemmType::FbInt8Packed) {
        // packing information
        int nrow;
        int ncol;
        int kernel_ncol_blocks;
        int brow;
        int bcol;
        int last_brow;
        int nbrow;
        int nbcol;
        uint64_t packsize;

        PackInfoFp16(val->shape(),
                    false,
                    nrow,
                    ncol,
                    kernel_ncol_blocks,
                    brow,
                    bcol,
                    last_brow,
                    nbrow,
                    nbcol,
                    packsize);

        auto allocator = New<TensorAllocator>(New<cpu::Backend>(CPU0, 1));

        allocator->allocate(packedTensor, {1, (int32_t)packsize}, Type::uint8);

        //PackFp32
        PackFp16(packedTensor,
                val,
                false,
                nrow,
                ncol,
                kernel_ncol_blocks,
                brow,
                bcol,
                last_brow,
                nbrow,
                nbcol,
                packsize);
        item.name = pName;
        item.shape = {(int32_t)packsize}; // val->shape();
        item.type = Type::uint8;
      } else {
        ABORT("Only int8 and fp16 is weights can be packed. {}", gemmType);
      }

      // Use the actual memory as this will be aligned and padded.
      // When memory mapping this is required. Shape keeps track of
      // tensor size. Saving to *.npz will cut to size.
      auto mem = packedTensor->memory();
      item.bytes.resize(mem->size());
      copy(backend_, mem->data<char>(), mem->data<char>() + mem->size(), item.bytes.data());

      ioItems.emplace_back(std::move(item));
    } else {
      io::Item item;
      item.name = pName;
      item.shape = val->shape();
      item.type = val->type();

      // Use the actual memory as this will be aligned and padded.
      // When memory mapping this is required. Shape keeps track of
      // tensor size. Saving to *.npz will cut to size.
      auto mem = val->memory();
      item.bytes.resize(mem->size());
      copy(backend_, mem->data<char>(), mem->data<char>() + mem->size(), item.bytes.data());

      ioItems.emplace_back(std::move(item));
    }
  }
}

}  // namespace marian
