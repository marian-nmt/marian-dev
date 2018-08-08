#include "training/graph_group_multinode_sync.h"
#include "functional/functional.h"
#include "tensors/tensor_operators.h"

#include "training/gradient_dropping/dropper.h"
#include "training/gradient_dropping/sparse_tensor.h"


#include "training/1bit_quantization/quantizer.h"

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

  // init for dropping
  if (droping_rate > 0.0) {
    int sparse_size = std::max(network_size * 0.1 ,
                               network_size * (1.0 - droping_rate));
    sparseGradient = SparseTensor(
          new SparseTensorBase(sparse_size * 1.2,
                               accGradientsSync->getBackend()));
    sparseGather = SparseTensor(
          new SparseTensorBase(sparse_size * mpi_comm_world_size_,
                               clientGraphs_.back()->getBackend()));

    dropper = PrepareGradientDrop(accGradientsSync->getDevice());
  }
  // init for quantization
  if (quantize_bit < 32) {
    int quantized_size = network_size * quantize_bit / 32;
    quantized = newTensor(quantized_size, 
                          accGradientsSync->getBackend());

    quantizer = Quantizer(new QuantizerBase());

    // test
    if (mpi_my_rank_ == 0)
      quantizer->test(accGradientsSync->getBackend());
  }
}


/**
 * Initialize the CPU arrays, with pinned memory for faster CudaMemCpy operations.
 * Requires the graph to be initialized first so we know its size
 */
void MultiNodeGraphGroupSync::initCPUArrays() {
  int network_size = clientGraphs_[0]->params()->vals()->size();

  accGradientsSync_cpu = std::vector<float>(network_size);
  receiveBuffer_cpu = std::vector<float>(network_size);

  // inits for gradient dropping
  if (droping_rate > 0.0) {
    int sparse_size = std::max(network_size * 0.1 ,
                               network_size * (1.0 - droping_rate));
    sparseGrad_cpu = std::vector<float>(sparse_size);
    sparseIndices_cpu = std::vector<int>(sparse_size);


    gatherGrads_cpu = std::vector<float>(sparse_size * 
                                        mpi_comm_world_size_);

    gatherIndices_cpu = std::vector<int>(sparse_size * 
                                        mpi_comm_world_size_);
  }
  // init for quantization
  if (quantize_bit < 32) {
    int quantized_size = network_size * quantize_bit / 32;
    quantized_cpu =  std::vector<float>(quantized_size);
    gatherQuantized_cpu = std::vector<float>(quantized_size * 
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

void MultiNodeGraphGroupSync::performUpdate(){
  // Perform optimizer step
  syncOptimizer_->update(clientGraphs_.back());

  if(movingAvg_)
    updateMovingAverage(
      paramsAvg_, clientGraphs_.back()->params()->vals(),
      scheduler_->numberOfBatches());
}

void MultiNodeGraphGroupSync::nodeParamSync(){
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
}

/**
 * If it's rank 0, it's a local update, if it's rank one it's remote
 * send and receive. Make sure you only call from device 0.
 */
void MultiNodeGraphGroupSync::sendReceiveUpdateQuantized() {
  #if MPI_FOUND
  int network_size = clientGraphs_[0]->params()->vals()->size();
  int quantized_size = quantized->size();

  float fetchAvg = 0;
  float avg = quantizer->quantize(accGradientsSync, quantized, quantize_bit);
  float averages[mpi_comm_world_size_];
  // Tensor quantized now holds quantized version of a accGradientsSync

  // Copy the quantized gradient to cpu
  quantized->get(quantized_cpu);

  // Wait until all nodes are ready
  MPI_Barrier(MPI_COMM_WORLD);
  
  // Gather quantized gradients
  MPI_Allgather(quantized_cpu.data(), quantized_size, MPI_FLOAT,
    gatherQuantized_cpu.data(), quantized_size, MPI_FLOAT,
    MPI_Comm MPI_COMM_WORLD);

  // Gather averages
  MPI_Allgather(&avg, 1, MPI_FLOAT, averages, 1, MPI_FLOAT,
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
    quantizer->dequantize(sumGradientBuffer, quantized, averages[i], quantize_bit);
    // accumulate the gradients
    using namespace functional;
    Element(_1 = _1 + _2, accGradientsSync, sumGradientBuffer);
    pos += quantized_size;
  }
  // copy gradient to last GPU 
  clientGraphs_.back()->params()->grads()->copyFrom(accGradientsSync);

  performUpdate();
  nodeParamSync();

  //set the accumulating buffers to zero;
  accGradientsSync->set(0);
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
  int sparse_limit = sparseGrad_cpu.size();
  float curr_droping_rate = droping_rate;


  static bool ok = false;
  if (!ok) {
    std::vector<float> x{ 100, -200, 300, 400, -400, 400};
    auto t = newTensor(6, accGradientsSync->getBackend());
    t->set(x);
    auto sparse = SparseTensor(
          new SparseTensorBase(4,
                               accGradientsSync->getBackend()));
    GradientDrop dropper = PrepareGradientDrop(accGradientsSync->getDevice());
    dropper->dropGraph(t, sparse, 0.5, 0);
    
    t->get(x);
    for (int i=0;i<6;i++) std::cout<<x[i]<<" "; std::cout<<std::endl;

    float *dt = (float *)malloc(10*sizeof(float));
    int *idc = (int *)malloc(10*sizeof(int));

    cudaMemcpy(dt, sparse->data(), sparse->size()*sizeof(float) , cudaMemcpyDeviceToHost);
    cudaMemcpy(idc, sparse->indices(), sparse->size()*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i=0;i<sparse->size();i++) std::cout<<dt[i]<<" : "<<idc[i]<<std::endl;
    std::vector<float> y{ 100, 100, 100, 100, 100, 100};
    t->set(y);
    dropper->dropGraph(t, sparse, 0.5, 0);
    t->get(y);
    for (int i=0;i<6;i++) std::cout<<y[i]<<" "; std::cout<<std::endl;

    ok = true;
  }

  if (true && scheduler_->numberOfBatches() < 1000) {
    curr_droping_rate = std::pow(droping_rate, 1000.0 / (1.0 + scheduler_->numberOfBatches()));
    // LOG(info, "WARMUP DROPPING = {}", curr_droping_rate);
  }

  int sparse_size = network_size * (1.0 - curr_droping_rate) * 1.2;
  static float resize_steps[4] = {0.95, 0.98, 0.99, 1.0};
  static int step_cnt = 0;
  if (sparse_size < sparseGrad_cpu.size() && curr_droping_rate >= resize_steps[step_cnt]) {
    step_cnt++;
    LOG(info, "resizing to {}", sparse_size);
    sparseGrad_cpu.resize(sparse_size);
    sparseIndices_cpu.resize(sparse_size * mpi_comm_world_size_);
  }
  static int step = 0;

  dropper->dropGraph(accGradientsSync,
                     clientGraphs_[0]->params()->grads(),
                     sparseGradient,
                     curr_droping_rate,
                     dropping_momentum);
  if (sparse_size > sparse_limit) {
  //  LOG(info, "full sync");
    sendReceiveUpdateSync(clientGraphs_[0]->params()->grads());
    return;
  }
  // Copy the gradient and indices to CPU
  sparseGradient->get(sparseGrad_cpu, sparseIndices_cpu, sparse_size);

  static MPI_Request r1, r2;
  // Gather gradient
  MPI_Iallgather(sparseGrad_cpu.data(), sparse_size, MPI_FLOAT,
    gatherGrads_cpu.data(), sparse_size, MPI_FLOAT,
    MPI_Comm MPI_COMM_WORLD, &r1);

  // Gather indices
  MPI_Iallgather(sparseIndices_cpu.data(), sparse_size, MPI_INT,
    gatherIndices_cpu.data(), sparse_size, MPI_INT,
    MPI_Comm MPI_COMM_WORLD, &r2);

  //parallel while data transfer is happening:
  if (false) {
    //replace
    using namespace functional; //@TODO makes more sense to do that on the CPU i think
    Element(_1 -= _2, accGradientsSync, clientGraphs_[0]->params()->grads());

    //sum or replace
    clientGraphs_.back()->params()->grads()->copyFrom(accGradientsSync);
    // error-sum
    // clientGraphs_.back()->params()->grads()->copyFrom(dropper->error());
    // dropper->error()->set(0);
  }
  
  MPI_Wait(&r1, MPI_STATUS_IGNORE);
  MPI_Wait(&r2, MPI_STATUS_IGNORE);

  // Update params
  // Copy the data back to the GPU and do optimizer update
  sparseGather->set(gatherGrads_cpu, gatherIndices_cpu, sparse_size * mpi_comm_world_size_);
  // clientGraphs_.back()->params()->grads()->set(0);
  // sparseGather->scatterAdd(clientGraphs_.back()->params()->grads(), 0);
  sparseGather->toDense(clientGraphs_.back()->params()->grads(), 0);

  performUpdate();

  if (true && step++ % 500 == 0) {
    clientGraphs_.back()->params()->vals()->get(accGradientsSync_cpu);
    MPI_Allreduce(accGradientsSync_cpu.data(), //CPU buffers
              receiveBuffer_cpu.data(),
              network_size,
              MPI_FLOAT,
              MPI_SUM,
              MPI_COMM_WORLD);
    LOG(info, "SYNC BRO");
    clientGraphs_.back()->params()->vals()->set(receiveBuffer_cpu);
    using namespace functional; //@TODO makes more sense to do that on the CPU i think
    Element(_1 /= 4, clientGraphs_.back()->params()->vals());
  }

  nodeParamSync();
  
  accGradientsSync->set(0);
  std::fill(sparseGrad_cpu.begin(), sparseGrad_cpu.end(), 0);
  std::fill(sparseIndices_cpu.begin(), sparseIndices_cpu.end(), 0);

  #endif
}

/**
 * If it's rank 0, it's a local update, if it's rank one it's remote
 * send and receive. Make sure you only call from device 0.
 */
void MultiNodeGraphGroupSync::sendReceiveUpdateSync(Tensor grad) {
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

  performUpdate();
  nodeParamSync();

  //set the accumulating buffers to zero;
  accGradientsSync->set(0);
  std::fill(accGradientsSync_cpu.begin(), accGradientsSync_cpu.end(), 0);
  std::fill(receiveBuffer_cpu.begin(), receiveBuffer_cpu.end(), 0);
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
 std::vector<Ptr<data::Batch>> batches = fullBatch->split(devices_.size());
 
  if(!initialized_) {
    init(batches[0]);
    initialized_ = true;
  }
  
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

  // float clipNorm = options_->get<double>("clip-norm");
  // static Ptr<ClipperBase> clipper = Clipper<Norm>(clipNorm);
  // if(clipNorm > 0) {
  //   if (t < 10) LOG(info, "LOCAL CLIP");
  //   clipper->clip(accGradientsSync);
  // }

  if (t % tau_ == 0)
    if (quantize_bit < 32)
      sendReceiveUpdateQuantized();
    else if (droping_rate > 0.0)
      sendReceiveUpdateSparse();
    else
      sendReceiveUpdateSync(accGradientsSync);

  t++;

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
      accGradientsSync->set(0);
      // inform other nodes to continue
      MPI_Barrier(MPI_COMM_WORLD);
      #endif
    }
  }

}
}
