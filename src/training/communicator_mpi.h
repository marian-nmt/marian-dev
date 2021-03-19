// Note: This must only be included if defined(CUDA_FOUND) && defined(USE_NCCL)
#include "training/communicator.h"
#include "3rd_party/threadpool.h"
#include "tensors/tensor_operators.h"

namespace marian {

class MpiCommunicator : public ICommunicator {
private:
  ShardingMode shardingMode_{ShardingMode::global};

  std::vector<int> devices_;       // For now there can only be one device
  Ptr<IMPIWrapper> mpi_;           // CAN NOT BE NULL
  mutable ThreadPool threadPool_;

  std::string mpiIdStr() const { // (for logging)
    return mpi_ ? mpi_->idStr() : "";
  }

  size_t numLocalRanks() const {
    return devices_.size();
  }

  size_t myLocalRank(size_t localDeviceIndex) const { // map local device index to a global rank
    return localDeviceIndex; // do nothing
  }

  size_t numRanks() const { // total number of devices across all MPI processes
      return mpi_->numMPIProcesses() * numLocalRanks();
  }

  size_t myRank(size_t localDeviceIndex) const { // map local device index to a global rank
      return mpi_->myMPIRank() * numLocalRanks() + myLocalRank(localDeviceIndex);
  }

  size_t dataSize() const { // total number of floats that comprise the concatenated parameter and gradient vector
    return graphs_[0]->params()->vals()->size();
  }

  // determine the (max) shard size
  // All shards except the last one have this size.
  // Presently, even all shards must have identical size, due to a limitation in NCCL we have not yet worked around.
  size_t shardSize() const {
    size_t numShards = shardingMode_ == ShardingMode::global ? numRanks() : numLocalRanks();
    size_t size = (dataSize() + numShards - 1) / numShards;
#if 1 // for now, all shards must have the same size, since NCCL does not allow a sub-slice for the last shard
    ABORT_IF(size * numShards != dataSize(), "presently, all shards must have the same size");
#endif
    return size;
  }

  // determine the index range (begin, end) of a shard
  std::pair<size_t, size_t> RankShardRange(size_t rank) const {
    size_t size = shardSize();
    size_t begin = rank * size;
    size_t end = begin + size;
    end = std::min(end, dataSize()); // clip last shard. Note: Presently this never happens, since shardSize() enforces uniform shard size.
    return{ begin, end };
  }

  // determine the index range (begin, end) of a shard
  std::pair<size_t, size_t> localShardRange(size_t localDeviceIndex) const {
    return RankShardRange(shardingMode_ == ShardingMode::global ? myRank(localDeviceIndex) : myLocalRank(localDeviceIndex));
  }

  void mpiBarrier() const {
    if (mpi_)
      mpi_->barrier();
  }

public:
  // This is mostly copied from NCCL communicator and edited. At the moment only global sharding mode
  // is supported with 1 CPU device per process. In the future we should have something like local sharding mode
  // using the default communicator, + a global sharding for gradient aggregation
  MpiCommunicator(const std::vector<Ptr<ExpressionGraph>>& graphs, ShardingMode shardingMode, Ptr<IMPIWrapper> mpi)
      : ICommunicator(graphs),
        shardingMode_(shardingMode),
        devices_(graphs.size()),
        mpi_(mpi),
        threadPool_(graphs.size(), graphs.size()) {
    mpiBarrier(); // barrier to group the multiple log messages from MPI processes
    LOG(info, "[comm] Using MPI communicator for CPU communication with {} processes.", mpi_->numMPIProcesses());
    mpiBarrier(); // (synchronize the log messages)

    // set up our local devices
    for(int i = 0; i < graphs_.size(); ++i) {
      auto device = graphs_[i]->getBackend()->getDeviceId();

      ABORT_IF(device.type == DeviceType::gpu, "MPI communicator can only be used with CPUs");
      ABORT_IF(graphs_.size() > 1, "MPI communicator can only be used with one thread per process for now.");
      devices_[i] = device.no;
    }
    ABORT_IF(!mpi_, "We require MPI for this implementation.");
    mpiBarrier();
    LOG(info, "[comm] Using {} sharding", shardingMode_ == ShardingMode::global ? "global" : "local");
    mpiBarrier();
    ABORT_IF(shardingMode_ != ShardingMode::global, "We only support global sharding mode for now.");

    mpiBarrier(); // (synchronize the log messages)
    LOG(info, "[comm] MPICommunicators constructed successfully");
    mpiBarrier(); // (synchronize the log messages)
  }

  ~MpiCommunicator() override {} // Empty destructor

  template <typename Ret>
  Ret foreachAcc(const ForeachFunc<Ret>& func, const AccFunc<Ret>& acc, Ret init, bool parallel = true) const {
    parallel &= graphs_.size() > 1;

    Ret retValue = init;
    std::vector<std::future<Ret>> threadResults(graphs_.size()); // [device index]
    for(size_t i = 0; i < graphs_.size(); ++i) {
      size_t begin, end; std::tie
      (begin, end) = localShardRange(i);
      if (parallel)
        threadResults[i] = threadPool_.enqueue(func, i, begin, end);
      else
        acc(retValue, func(i, begin, end));
    }
    if(parallel)
       for(auto& task : threadResults)
          acc(retValue, task.get());

    return retValue;
  }

  float foreach(const ForeachFunc<float>& func, AccFunc<float> acc, float init, bool parallel = true) const override {
    return foreachAcc(func, acc, init, parallel);
  }

  bool foreach(const ForeachFunc<bool>& func, bool parallel = true) const override {
    AccFunc<bool> allTrue = [](bool& x, bool y) { x = x && y; };
    return foreachAcc(func, allTrue, true, parallel);
  }

  void scatterReduceAndResetGrads() const override {
    for(int i = 0; i < graphs_.size(); ++i) {
      size_t begin, end; std::tie
      (begin, end) = localShardRange(i);

      auto grads = graphs_[i]->params()->grads();
      const auto* sendbuf = grads->data();
      //auto*       recvbuf = grads->subtensor(begin, end-begin)->data();
      size_t      bufsize = shardSize();
      ABORT_IF(grads->subtensor(begin, end-begin)->size() != bufsize, "unexpected subtensor size??");

      MPI_Datatype mpiFLoatType = MPI_FLOAT;
      if(grads->type() == Type::float16)
        ABORT("Half precision is datatype is not supported by MPI.");
      mpiBarrier(); // This barrier should be outside of the for loop probably.
      if(shardingMode_ == ShardingMode::global) {
        mpi_->reduceScatterBlock(sendbuf, (void *)sendbuf, bufsize, mpiFLoatType, MPI_SUM);
        // NCCL_CHECK(ncclReduceScatter(sendbuf, recvbuf, bufsize, ncclFloatType, ncclSum, globalComms_[i], streams_[i]));
      } else {
        ABORT("Local sharding mode reduceScatter not supported yet for mpi communicator.");
        //NCCL_CHECK(ncclReduceScatter(sendbuf, recvbuf, bufsize, ncclFloatType, ncclSum,  localComms_[i], streams_[i])); // reduceScatter locally
        //NCCL_CHECK(    ncclAllReduce(recvbuf, recvbuf, bufsize, ncclFloatType, ncclSum, globalComms_[i], streams_[i])); // then do tuple-wise allReduce across processes
      }
      mpiBarrier();
    }

    // reset gradients outside the shards we reduce in
    // In the future, we can keep quantization residuals here straight in the grads themselves.
    // @TODO: all the different places where gradients get reset are confusing
    auto resetGrads = [&](size_t i, size_t begin, size_t end) {
      auto grads = graphs_[i]->params()->grads();
      auto size = grads->size();
      // reset everything outside the shard that we reduce in
      if (begin > 0)
        grads->subtensor(0, begin)->set(0.f);
      if (end < size)
        grads->subtensor(end, size - end)->set(0.f);

      return true; // dummy success
    };
    foreach(resetGrads);
  }

  // This distributes all 64 model shards to all 64 GPUs.
  // @TODO: For unknown reasons, this takes longer than any other operation incl. scatterReduceAndResetGrads().
  //        But both should have the same number of data transfers of the same size.
  void allGatherParams() const override {
    for(int i = 0; i < graphs_.size(); ++i) {
      size_t begin, end; std::tie
      (begin, end) = localShardRange(i);

      auto vals = graphs_[i]->params()->vals();
      //const auto* sendbuf = vals->subtensor(begin, end-begin)->data();
      void*       recvbuf = vals->data();
      size_t      bufsize = shardSize();

      MPI_Datatype mpiFLoatType = MPI_FLOAT;
      if(vals->type() == Type::float16)
        ABORT("Half precision is datatype is not supported by MPI.");
      mpiBarrier(); // This barrier should be outside of the for loop probably.
      mpi_->Allgather(recvbuf, bufsize, mpiFLoatType, recvbuf, bufsize, mpiFLoatType);
      //the local version did it so:
      //auto& comms = shardingMode_ == ShardingMode::global ? globalComms_ : localComms_;
      //NCCL_CHECK(ncclAllGather(sendbuf, recvbuf, bufsize, ncclFloatType, comms[i], streams_[i]));
      mpiBarrier();
    }
  }

  void broadcastParams(bool average = false) const override {

    for(int i = 0; i < graphs_.size(); ++i) {
      auto vals = graphs_[i]->params()->vals();

      MPI_Datatype mpiFLoatType = MPI_FLOAT;
      if(vals->type() == Type::float16)
        ABORT("Half precision is datatype is not supported by MPI.");
      mpiBarrier(); // This barrier should be outside of the for loop probably.

      if(average)
        mpi_->allReduce(vals->data(), vals->data(), vals->size(), mpiFLoatType, MPI_SUM);
      else
        mpi_->bCast(vals->data(), vals->size(), mpiFLoatType, 0);
      mpiBarrier();
    }


    if(average) {
      auto avg = [&](size_t i, size_t /*begin*/, size_t /*end*/) {
        auto vals = graphs_[i]->params()->vals();
        using namespace functional;
        Element(_1 = _1 / (float)mpi_->numMPIProcesses(), vals);
        return true; // dummy success
      };
      foreach(avg);
    }
  }

  void broadcastShards(const std::vector<Ptr<OptimizerBase>>& opts, bool average = false) const override {
    if(shardingMode_ == ShardingMode::global)
      return; // nothing to do, shards are indepedent

    auto floatType = [](Tensor tensor) {
      MPI_Datatype mpiFLoatType = MPI_FLOAT;
      if(tensor->type() == Type::float16)
        ABORT("Half precision is datatype is not supported by MPI.");
      return mpiFLoatType;
    };

    // if we are here we use local mode and shards are process-wise copies
    // This is not yet supported for MPICommunicator, but it wouldn't hurt to have the code there
    ABORT("Local sharding mode reduceScatter not supported yet for mpi communicator.");
    mpiBarrier();
    for(int i = 0; i < opts.size(); ++i) {
      for(auto shard : opts[i]->getShards()) {
        if(shard) {
          if(average) {
            mpi_->allReduce(shard->data(), 
                            shard->data(), 
                            shard->size(), 
                            floatType(shard), 
                            MPI_SUM);
            using namespace functional;
            Element(_1 = _1 / (float)mpi_->numMPIProcesses(), shard);
          } else {
            mpi_->bCast(shard->data(), 
                        shard->size(), 
                        floatType(shard), 
                        0);
          }
        }
      }
    }
  }

  // Distribute a single CPU-side io::Item to shards across multiple devices and MPI processes.
  // This is used when restoring optimizer state, which is sharded.
  // It is assumed that all MPI processes get the same data() passed. Hence, no MPI transfers are needed here.
  void scatterState(const io::Item& data, const OptimizerBase::ScatterStateSetFunc& setFn) const override {
    size_t dataSize = data.size();
    size_t numShards = shardingMode_ == ShardingMode::global ? numRanks() : numLocalRanks(); // @TODO: numShards() function
    size_t shardSize = (dataSize + numShards - 1) / numShards;
    for(size_t localDeviceIndex = 0; localDeviceIndex < graphs_.size(); localDeviceIndex++) {
      // We only slice out data that is kept in our MPI process. Remember that all MPI processes receive the same, complete data.
      auto ncclRank = shardingMode_ == ShardingMode::global ? myRank(localDeviceIndex) : myLocalRank(localDeviceIndex);
      size_t begin = ncclRank * shardSize;
      size_t end   = std::min(begin + shardSize, dataSize);
      setFn(localDeviceIndex, data.bytes.data() + begin, data.bytes.data() + end);
    }
  }

  // Collect shards across multiple devices and MPI processes in the NCCL configuration into a single CPU-side io::Item.
  // This is used when persisting optimizer state which is sharded.
  io::Item gatherState(const OptimizerBase::GatherStateGetFunc& getFn) const override {
    // first, concatenate over all local devices
    io::Item localData = getFn(0);
    for(size_t localDeviceIndex = 1; localDeviceIndex < graphs_.size(); localDeviceIndex++) {
      localData.append(getFn(localDeviceIndex));
    }
    // localData now contains a concatentation of all local data

    // second, concatenate across MPI processes
    // Note that all local devices occupy consecutive ncclRanks in order.
    io::Item data;
    if (mpi_ && shardingMode_ == ShardingMode::global) {
      io::Item tmp = localData; // temp buffer used multiple times; assign localData for initialization
      // push one rank's data at a time using broadcast
      for(size_t mpiRank = 0; mpiRank < mpi_->numMPIProcesses(); mpiRank++) {
        // broadcast mpiRank's localData to all
        if(mpiRank == mpi_->myMPIRank()) {
          tmp = localData;
        }
        mpi_->bCast(tmp, /*rootRank=*/mpiRank);
        // now all ranks have the same slice: concatenate (we will end up with the same on all MPI processes)
        if(mpiRank == 0)
          data = tmp;
        else
          data.append(tmp);
      }
    }
    else { // no MPI: localData is the complete result already
      data = std::move(localData);
    }
    return data;
  }
};

}  // namespace marian
