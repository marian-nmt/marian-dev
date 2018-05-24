#include "training/graph_group_multinode_sync.h"
#include "functional/functional.h"
#include "tensors/tensor_operators.h"

#include "training/gradient_dropping/dropper.h"
#include "training/gradient_dropping/sparse_tensor.h"


#include "training/1bit_quantization/quantizer.h"
#include <chrono>

namespace marian {

void MultiNodeGraphGroupSync::updateMovingAverage(Tensor paramsAvg,
                                         Tensor params,
                                         size_t batches) {
  using namespace functional;
  float decay
      = std::max(mvDecay_, 1.f - (float)(batches + 1) / (float)(batches + 10));
  Element(_1 = ((1.f - decay) * _1) + (decay * _2), paramsAvg, params);
}


/**
 * Set given scheduler to register training observers on the shard optimizers.
 */
void MultiNodeGraphGroupSync::setScheduler(Ptr<Scheduler> scheduler) {
  scheduler_ = scheduler;
  // optimizer has to be registered last to see a change of learning rate
  scheduler_->registerTrainingObserver(scheduler_);

  scheduler_->registerTrainingObserver(syncOptimizer_);

  scheduler_->registerTrainingObserver(localOptimizer_);

}

/**
 * Allocate new tensor on given GPU and store allocator.
 */
Tensor MultiNodeGraphGroupSync::newTensor(int size, Ptr<Backend> backend) {
  Tensor t;
  Ptr<TensorAllocator> allocator = New<TensorAllocator>(backend);
  allocator->reserveExact(size * sizeof(float));
  allocator->allocate(t, {1, size});
  allocators_.push_back(allocator);
  return t;
}

/**
 * Setup training environment and launch server thread and (if enabled) client
 * communication overlap threads.
 * Includes setting up MPI, node and shard sizes, clients, server shards and
 * communication overlap stuff.
 */
void MultiNodeGraphGroupSync::init(Ptr<data::Batch> batch) {
  // Setup clients and shards
  setupClients(batch);
  int network_size = clientGraphs_[0]->params()->vals()->size();
  LOG(info, "model size = {} float params" , network_size);
  if (movingAvg_)
    paramsAvg_ = newTensor(network_size, clientGraphs_.back()->getBackend());

  // setup sync sgd storage, We keep the summed gradient on Node 0
  sumGradientBuffer = newTensor(network_size, clientGraphs_[0]->getBackend());
  accGradientsSync = newTensor(network_size, clientGraphs_[0]->getBackend());

  // quantized = newTensor(clientGraphs_[0]->params()->vals()->size() / 32, clientGraphs_[0]->getBackend());
}

/**
 * Initialize the CPU arrays, with pinned memory for faster CudaMemCpy operations.
 * Requires the graph to be initialized first so we know its size
 */
void MultiNodeGraphGroupSync::initCPUArrays() {
  int network_size = clientGraphs_[0]->params()->vals()->size();

  if (droping_rate == 0.0) {
    accGradientsSync_cpu = std::vector<float>(network_size);
    receiveBuffer_cpu = std::vector<float>(network_size);
  } else {
    sparseGrad_cpu = std::vector<float>(network_size * (1.0 - droping_rate));
    sparseIndices_cpu = std::vector<int>(network_size * (1.0 - droping_rate));


    gatherGrads_cpu = std::vector<float>(network_size * 
                                        (1.0 - droping_rate) * 
                                        mpi_comm_world_size_);

    gatherIndices_cpu = std::vector<int>(network_size * 
                                        (1.0 - droping_rate) * 
                                        mpi_comm_world_size_);
  }
}

/**
 * Setup MPI world size and rank of this node.
 */
void MultiNodeGraphGroupSync::setupMPI() {
#if MPI_FOUND
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_comm_world_size_);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_my_rank_);
#endif
}

/**
 * Setup clients that will compute gradients and communicate them with the
 * server shards.
 * There is one client per GPU.
 */
void MultiNodeGraphGroupSync::setupClients(Ptr<data::Batch> batch) {
  runBatchThroughClientGraphs(batch);
  initCPUArrays();

  clientThreadPool_ = new marian::ThreadPool(devices_.size(), devices_.size());
}

/**
 * Initialize the graphs (models) of all clients on this node with the given
 * batch.
 */
void MultiNodeGraphGroupSync::runBatchThroughClientGraphs(Ptr<data::Batch> batch) {
  for(int i = 0; i < devices_.size(); i++) {
    THREAD_GUARD(clientBuilders_[i]->build(clientGraphs_[i], batch);
                 clientGraphs_[i]->forward();
                 clientGraphs_[i]->getBackend()->synchronize(););
  }
}

/**
 * Initialize variables required for overlapping client computations and
 * communication.
 * Includes summed and committed word counts, buffer flags, mutexes and
 * condition variables.
 */
void MultiNodeGraphGroupSync::sumGRAD(Tensor gradient) {
  std::lock_guard<std::mutex> guard(sumGradientMutex_);
  sumGradientBuffer->copyFrom(gradient);
  using namespace functional; //@TODO makes more sense to do that on the CPU i think
  Element(_1 += _2, accGradientsSync, sumGradientBuffer);
}




/**
 * If it's rank 0, it's a local update, if it's rank one it's remote
 * send and receive. Make sure you only call from device 0.
 */
void MultiNodeGraphGroupSync::sendReceiveUpdateQuantized() {
  #if MPI_FOUND
  int network_size = clientGraphs_[0]->params()->vals()->size();
  static std::vector<float> quantized_cpu( network_size / 32);
  int quantized_size = quantized_cpu.size();

  static Quantizer quantizer = Quantizer(new QuantizerBase());
  static Quantizer quantizerFetch = Quantizer(new QuantizerBase());

  static std::vector<float> gatherQuantized_cpu(quantized_size * mpi_comm_world_size_);
  //LOG(info, "quantizing");
  float fetchAvg = 0;
  float avg = quantizer->quantize(accGradientsSync, quantized);
  float averages[mpi_comm_world_size_];
  // Tensor quantized now holds quantized version of a accGradientsSync
  
  // Copy the quantized gradient to cpu
  quantized->get(quantized_cpu);

  // Wait until all nodes are ready
  MPI_Barrier(MPI_COMM_WORLD);

  // Gather quantized gradients
  MPI_Gather(quantized_cpu.data(), quantized_size, MPI_FLOAT,
    gatherQuantized_cpu.data(), quantized_size, MPI_FLOAT, 0,
    MPI_Comm MPI_COMM_WORLD);
  //LOG(info, "gather avg");
  // Gather averages
  MPI_Gather(&avg, 1, MPI_FLOAT,
    averages, 1, MPI_FLOAT, 0,
    MPI_Comm MPI_COMM_WORLD);

  // Construct the gradients
  // TODO: not effective when nodes > 2
  // we will sum all the gradients here
  accGradientsSync->set(0);
  int pos = 0;
  for (int i=0;i < mpi_comm_world_size_; i++) {
    // copy from CPU to GPU
    quantized->set(gatherQuantized_cpu.data() + pos, 
                   gatherQuantized_cpu.data() + pos + quantized_size);

    // revert back to dense
    quantizer->dequantize(sumGradientBuffer, quantized, averages[i]);

    // accumulate the gradients
    using namespace functional;
    Element(_1 = _1 + _2, accGradientsSync, sumGradientBuffer);
    pos += quantized_size;
  }

  // copy gradient to last GPU

  clientGraphs_.back()->params()->grads()->copyFrom(accGradientsSync);

  // Perform optimizer step
  syncOptimizer_->update(clientGraphs_.back());
  if(movingAvg_)
          updateMovingAverage(
            paramsAvg_, clientGraphs_.back()->params()->vals(),
            scheduler_->numberOfBatches());

//Distribute the graph to the rest of the devices
  std::vector<std::thread> threads;
  for(int idx = 0; idx < devices_.size() - 1; idx++) {
    threads.emplace_back(std::thread(
        [=](int idx) {
          clientGraphs_[idx]->params()->vals()->copyFrom(
            clientGraphs_.back()->params()->vals());
        },
        idx));
  }
  for(auto&& t : threads) {
    t.join();
  }

  //set the accumulating buffers to zero;
  accGradientsSync->set(0);
  std::fill(accGradientsSync_cpu.begin(), accGradientsSync_cpu.end(), 0);
  std::fill(gatherQuantized_cpu.begin(), gatherQuantized_cpu.end(), 0);
  #endif
}




/**
 * If it's rank 0, it's a local update, if it's rank one it's remote
 * send and receive. Make sure you only call from device 0.
 */

void MultiNodeGraphGroupSync::sendReceiveUpdateSparse() {
  #if MPI_FOUND
  int network_size = accGradientsSync->size();
  int sparse_size = sparseGrad_cpu.size();

  if (!sparseGradient) {
    sparseGradient = SparseTensor(
          new SparseTensorBase(sparse_size * mpi_comm_world_size_ * 1.2,
                               accGradientsSync->getBackend()));
  }

  if (!dropper) {
    dropper = PrepareGradientDrop(accGradientsSync->getDevice());
  }

  // drop the gradient
  dropper->dropGraph(accGradientsSync,
                     sparseGradient,
                     droping_rate,
                     dropping_momentum);

  // Copy the gradient and indices to CPU
  sparseGradient->get(sparseGrad_cpu, sparseIndices_cpu);

  // Wait until all nodes are ready
  MPI_Barrier(MPI_COMM_WORLD);

  // Gather gradient
  MPI_Allgather(sparseGrad_cpu.data(), sparse_size, MPI_FLOAT,
    gatherGrads_cpu.data(), sparse_size, MPI_FLOAT,
    MPI_Comm MPI_COMM_WORLD);

  // Gather indices
  MPI_Allgather(sparseIndices_cpu.data(), sparse_size, MPI_INT,
    gatherIndices_cpu.data(), sparse_size, MPI_INT,
    MPI_Comm MPI_COMM_WORLD);

  // Update params
  // Do update with last GPU to distribute the memory
  static SparseTensor tmp = SparseTensor(
          new SparseTensorBase(sparse_size * mpi_comm_world_size_,
                               clientGraphs_.back()->getBackend()));

  // Copy the data back to the GPU and do optimizer update
  tmp->set(gatherGrads_cpu, gatherIndices_cpu);
  tmp->toDense(clientGraphs_.back()->params()->grads(), 0);

  // Perform optimizer step
  syncOptimizer_->update(clientGraphs_.back());

  if(movingAvg_)
    updateMovingAverage(
      paramsAvg_, clientGraphs_.back()->params()->vals(),
      scheduler_->numberOfBatches());

  //Distribute the graph to the rest of the devices
  std::vector<std::thread> threads;
  for(int idx = 0; idx < devices_.size() - 1; idx++) {
    threads.emplace_back(std::thread(
        [=](int idx) {
          clientGraphs_[idx]->params()->vals()->copyFrom(
            clientGraphs_.back()->params()->vals());
        },
        idx));
  }
  for(auto&& t : threads) {
    t.join();
  }

  //set the accumulating buffers to zero;
  accGradientsSync->set(0);
  std::fill(sparseGrad_cpu.begin(), sparseGrad_cpu.end(), 0);
  std::fill(sparseIndices_cpu.begin(), sparseIndices_cpu.end(), 0);
  #endif
}

/**
 * If it's rank 0, it's a local update, if it's rank one it's remote
 * send and receive. Make sure you only call from device 0.
 */
void MultiNodeGraphGroupSync::sendReceiveUpdateSync() {
  #if MPI_FOUND
  int network_size = accGradientsSync_cpu.size();

  // Copy the data to the CPU
  accGradientsSync->get(accGradientsSync_cpu);

  // Wait until all nodes are ready
  MPI_Barrier(MPI_COMM_WORLD);

  int reduce_result = MPI_Allreduce(accGradientsSync_cpu.data(), //CPU buffers
              receiveBuffer_cpu.data(),
              network_size,
              MPI_FLOAT,
              MPI_SUM,
              MPI_COMM_WORLD);

  // Copy the data back to the GPU and do optimizer update
  // Do update with last GPU to distribute the memory
  clientGraphs_.back()->params()->grads()->set(receiveBuffer_cpu);

  // Perform optimizer step
  syncOptimizer_->update(clientGraphs_.back());

  if(movingAvg_)
      updateMovingAverage(
        paramsAvg_, clientGraphs_.back()->params()->vals(),
        scheduler_->numberOfBatches());

  //Distribute the graph to the rest of the devices
  std::vector<std::thread> threads;
  for(int idx = 0; idx < devices_.size() - 1; idx++) {
    threads.emplace_back(std::thread(
        [=](int idx) {
          clientGraphs_[idx]->params()->vals()->copyFrom(
            clientGraphs_.back()->params()->vals());
        },
        idx));
  }
  for(auto&& t : threads) {
    t.join();
  }

  //set the accumulating buffers to zero;
  accGradientsSync->set(0);
  #endif
}


/**
 * Execute given batch on this node, pushing/pulling the resulting
 * gradients/parameters to/from the server shards
 * or -- if comm. overlap enabled -- to/from the communication buffers, summing
 * gradients locally if the communication thread is busy
 *
 * @param batch Batch on which to perform forward and backward passes.
 */
void MultiNodeGraphGroupSync::execute(Ptr<data::Batch> fullBatch) {
  if(!initialized_) {
    init(fullBatch);
    initialized_ = true;
  }
  static double avgBatch = 0;
  std::vector<Ptr<data::Batch>> batches = fullBatch->split(devices_.size());
  
  static int t = 0;

  static float cost = 0;
  static size_t num_seen_words = 0;
  static size_t num_seen_sentences = 0;

  {
    auto task = [this, batches](int my_id) {
      auto batch = batches[my_id];
      auto graph = clientGraphs_[my_id];
      auto builder = clientBuilders_[my_id];

      auto costNode = builder->build(graph, batch);

      if (t == 0) {
        if (my_id != 0)
          graph->params()->vals()->copyFrom(clientGraphs_[0]->params()->vals());
      }

      graph->forward();
      {
        std::lock_guard<std::mutex> guard(sumCostMutex_);
        cost += costNode->scalar();
        num_seen_words += batch->words();
        num_seen_sentences += batch->size();
      }
      graph->backward();

      graph->getBackend()->synchronize(); //@Alham do you know why we need this here?

      sumGRAD(graph->params()->grads());
    };

    ThreadPool pool(devices_.size(), devices_.size());
    for(int idx = 0; idx < devices_.size(); ++idx)
      pool.enqueue(task, idx);
  }

  // local optimizer
  // localOptimizer_->update(clientGraphs_[0]->params()->vals(),
  //                       accGradientsSync);

  if (t % tau_ == 0)
    if (droping_rate > 0.0)
      sendReceiveUpdateSparse();
    else
      sendReceiveUpdateSync();

  // Run scheduler (if enabled)
  if(t % tau_ == 0 && scheduler_) {
    if (options_->get<std::string>("cost-type") != "ce-sum")
      cost /= (tau_ * devices_.size());

    if (tau_ > 1) { 
      std::vector<size_t> fakeLength = {1, 1};
      auto fb = data::CorpusBatch::fakeBatch(fakeLength,
                                        num_seen_sentences,
                                        NULL);
      fb->front()->setWords(num_seen_words);
      scheduler_->update(cost, fb);
    } else {
      scheduler_->update(cost, fullBatch);
    }
    
    num_seen_words = 0;
    num_seen_sentences = 0;
    cost = 0;

    if((scheduler_->saving() || scheduler_->validating())) {
      #if MPI_FOUND
      //wait until other nodes are ready
      MPI_Barrier(MPI_COMM_WORLD);

      // TODO: Saving is broken
      //if(mpi_my_rank_ == 0 && scheduler_->saving())
      //  this->save(graph);

      if(mpi_my_rank_ == 0 && scheduler_->validating()) {
        // temporarily save current params
        if(movingAvg_)
          accGradientsSync->copyFrom(clientGraphs_[0]->params()->vals());

        if(movingAvg_)
          for(auto graph : clientGraphs_)
            graph->params()->vals()->copyFrom(paramsAvg_);

        scheduler_->validate(clientGraphs_);

        if(movingAvg_)
          for(auto graph : clientGraphs_)
            graph->params()->vals()->copyFrom(accGradientsSync);
      }

      // inform other nodes to continue
      MPI_Barrier(MPI_COMM_WORLD);
      #endif
    }
    }

}
}
