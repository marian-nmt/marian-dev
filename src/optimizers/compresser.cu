#include <curand.h>
#include <curand_kernel.h>
#include <cmath>

#include <thrust/inner_product.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>


#include "optimizers/compresser.h"
#include "tensors/tensor_operators.h"
#include "tensors/tensor_allocator.h"



namespace marian {

  __global__ void gClip(float* data,
                            int size,
                            float range) {
      int idx = blockDim.x * blockIdx.x + threadIdx.x;
      if(idx >= size)
        return;                       
    
      if (data[idx] < -range) data[idx] = -range;
      if (data[idx] > range) data[idx] = range;
    }


  __global__ void gQuantizeFix(float* data,
                               float* delta,
                               int size,
                               int num_centers,
                               float max) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= size)
      return;

    // get sign flag
    float dataTemp = data[idx];

    int  center =  round((data[idx] / max) * num_centers); 
    if (center < -num_centers) center = -num_centers;
    if (center > num_centers - 1) center = num_centers - 1;

    data[idx] = (float) (center * max) / num_centers;
  
    if (delta != NULL) {
      delta[idx] = data[idx] / max;

      // scaled delta
      // delta[idx] = (abs(data[idx]) / abs(dataTemp) - 1.0) * max;
      data[idx] = dataTemp;
    }
  }

  __global__ void gKmeansAssign(float* data, float* centroids, float* values, float* centers, int size, int num_centers) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= size)
      return;
    values[idx] = abs(data[idx]);
    float best_distance = 999.0;
    int best_cluster = 0;
    for (int cluster = 0; cluster < num_centers; ++cluster) { 
      const float distance = abs(values[idx] - centers[cluster]);
      if (distance < best_distance) {
        best_distance = distance;
        best_cluster = cluster;
      }
    }

    centroids[idx] = best_cluster;
  }

  __global__ void gKmeansCompress(float* data, float* centroids, float* centers, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= size)
      return;
    
    if (data[idx] < 0)
      data[idx] = -centers[(int) centroids[idx]];
    else
      data[idx] = centers[(int) centroids[idx]];
  }

  __global__ void gQuantize(float* data,
                            float* delta,
                            int size,
                            int num_centers,
                            float base,
                            float max) {
  
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= size)
      return;

    // get sign flag
    float dataTemp = data[idx];

    bool isNeg = false;
      if (data[idx] < 0) {
      isNeg = true;
      data[idx] *= -1;
    }

    // compute the log of the parameter
    data[idx] /= max;
    int center = floor(log(data[idx] * (2.0 * base)/(1.0 + base)) / log(base));
    
    // clip the center to [0, 2^(bit-1)-1]    
    if (center < -num_centers)
      center = -num_centers;
    if (center > 0)
      center = 0;

    // revert back to floating point representation
    data[idx] = std::pow(base, center) * max;
    if (isNeg)
      data[idx] *= -1;
   
    if (delta != NULL) {
      // normal delta
      delta[idx] = data[idx] / max;
      
      // scaled delta
      // delta[idx] = (abs(data[idx]) / abs(dataTemp) - 1.0) * max;
      data[idx] = dataTemp;
    }
  }

  
  void compressFix(Tensor t, int bit, float clipRange, int kMeanStep) {
    int id = t->getDeviceId().no;
    static Tensor delta[4];
    static Ptr<TensorAllocator> alloc_[4];
    if (!delta[id]) {
      int msize = 100000000;
      LOG(info, "INIT DELTA FIXED-POINTZ FOR {} : {}", t->getDeviceId().no, msize);
      alloc_[id] = New<TensorAllocator>(t->getBackend());

      int elements = (int)msize;
      alloc_[id]->reserveExact(msize *sizeof(float));
      alloc_[id]->allocate(delta[id], {1, elements});

    }

    cudaSetDevice(t->getDeviceId().no);
    int threads = 512;
    int blocksSample = 1 + t->size() / threads;

    // clip first
    if (clipRange > 0.0)
      gClip<<<blocksSample, threads>>>(t->data(), t->size(), clipRange);

    // get max element in Tensor
    int centers  = (1<<(bit - 1));
    thrust::device_ptr<float> d_ptr(t->data());
    float max = *(thrust::max_element(d_ptr, d_ptr + t->size())) * centers / (centers - 1);
    float min = *(thrust::min_element(d_ptr, d_ptr + t->size())) * -1;
    max = std::max(min, max);
    // max adjustment 
    static int s = 0;
    // max adjustment 
    for (int i=0;i< kMeanStep;i++) {
      if (id == 0 && s++ < 100) LOG(info, "max adjust step {} = {} ", i, max);
      gQuantizeFix<<<blocksSample, threads>>>(t->data(), delta[id]->data(), t->size(), (1<<(bit-1)), max);

      thrust::device_ptr<float> delta_ptr(delta[id]->data());
      float delta_top = thrust::inner_product(delta_ptr, delta_ptr + t->size(), d_ptr, 0.0f);
      float delta_btm = thrust::inner_product(delta_ptr, delta_ptr + t->size(), delta_ptr, 0.0f);
      max = delta_top / delta_btm;
    }

    gQuantizeFix<<<blocksSample, threads>>>(t->data(), NULL, t->size(), (1<<(bit-1)), max);
  }

template<typename T>
struct absolute_value
{
  __host__ __device__ T operator()(const T &lhs, const T &rhs) const
  {
	  return abs(lhs) + abs(rhs);
  }
};

template<typename T>
struct square
{
    __host__ __device__
        T operator()(const T& x) const { 
            return x * x;
        }
};


  void real_kmeans(Tensor t, int bit, float clipRange, int kMeanStep) {
    int id = t->getDeviceId().no;
    int N = (1<<(bit - 1));

    static Tensor centroid[4], values[4], means[4];
    static Ptr<TensorAllocator> alloc_centroid[4], alloc_values[4], alloc_means[4];

    if (!centroid[id]) {
      int msize = 100000000;
      LOG(info, "INIT DELTA FOR {} : {}", t->getDeviceId().no, msize);
      alloc_centroid[id] = New<TensorAllocator>(t->getBackend());
      alloc_values[id] = New<TensorAllocator>(t->getBackend());
      alloc_means[id] = New<TensorAllocator>(t->getBackend());

      int elements = (int)msize;
      alloc_centroid[id]->reserveExact(msize *sizeof(float));
      alloc_values[id]->reserveExact(msize *sizeof(float));
      alloc_means[id]->reserveExact(N *sizeof(float));

      alloc_centroid[id]->allocate(centroid[id], {1, elements});
      alloc_values[id]->allocate(values[id], {1, elements});
      alloc_means[id]->allocate(means[id], {1, N});
    }
    
    cudaSetDevice(t->getDeviceId().no);
    int threads = 512;
    int blocksSample = 1 + t->size() / threads;

    // clip first
    if (clipRange > 0.0)
      gClip<<<blocksSample, threads>>>(t->data(), t->size(), clipRange);

    // get max element in Tensor
   
    float max = 0;
     /*
     thrust::device_ptr<float> d_ptr(t->data());

    max = *(thrust::max_element(d_ptr, d_ptr + t->size()));
    float min = *(thrust::min_element(d_ptr, d_ptr + t->size())) * -1;
    max = std::max(max, min);
    */
    max = 1.0;
    // init centroids
    for (int i = 0;i < N;i++) {
      means[id]->set(i, max); 
      max /= 2.0;
    }
    
    thrust::device_ptr<float> centroid_ptr(centroid[id]->data());
    thrust::device_ptr<float> tmpvalue_ptr(values[id]->data());

    // kmean loop
    bool isImproved = true;
    for (int step = 0; step <= kMeanStep; step++) {
      // assign
      gKmeansAssign<<<blocksSample, threads>>>(t->data(), centroid[id]->data(), values[id]->data(), means[id]->data(), t->size(), N);

       // update the centers, unless it is the last step:
      if (!isImproved || step == kMeanStep)
        break;
      isImproved = false;

      // sort the key
      thrust::sort_by_key(centroid_ptr, centroid_ptr + t->size(), tmpvalue_ptr);
      
      // get the new centers
      // step 1: sum all the value for each centers
      thrust::device_vector<float> reduced_key(N), reduced_sum(N), reduced_count(N);
      thrust::reduce_by_key(centroid_ptr, centroid_ptr + t->size(), tmpvalue_ptr, reduced_key.begin(), reduced_sum.begin());  

      // step 2: count the member, then average
      thrust::reduce_by_key(centroid_ptr, centroid_ptr + t->size(), thrust::make_constant_iterator(1), reduced_key.begin(), reduced_count.begin());

      for (int i=0; i< N;i++) {
        if (reduced_count[i] == 0)
          continue;
        int cid = (int) reduced_key[i];
        float new_mean = reduced_sum[i] / reduced_count[i];
        if (abs(new_mean - means[id]->get(cid)) > 1e-4) isImproved = true;

        if (id == 0 && cid == 0) LOG_ONCE(info, "(step {}) center {} mean {} -> {}", step, cid, means[id]->get(cid), new_mean);
        means[id]->set(cid,new_mean);
      }
    }

    // compress
    if (id == 0) LOG_ONCE(info, "before {} {} {} ", t->get(0), t->get(10), t->get(100));
    gKmeansCompress<<<blocksSample, threads>>>(t->data(), centroid[id]->data(), means[id]->data(), t->size());
    if (id == 0) LOG_ONCE(info, "after  {} {} {} ", t->get(0), t->get(10), t->get(100));
  }


  void compressImpl(Tensor t, int bit, float base, float clipRange, bool isMax, int kMeanStep){
    int id = t->getDeviceId().no;
    static Tensor delta[4];
    static Ptr<TensorAllocator> alloc_[4];
    if (!delta[id]) {
      int msize = 100000000;
      LOG(info, "INIT DELTA FOR {} : {}", t->getDeviceId().no, msize);
      alloc_[id] = New<TensorAllocator>(t->getBackend());

      int elements = (int)msize;
      alloc_[id]->reserveExact(msize *sizeof(float));
      alloc_[id]->allocate(delta[id], {1, elements});
  
    }
    // LOG(info, "address {}, size {}", t, t->size());

    cudaSetDevice(t->getDeviceId().no);
    int threads = 512;
    int blocksSample = 1 + t->size() / threads;
 
    // clip first
    if (clipRange > 0.0)
      gClip<<<blocksSample, threads>>>(t->data(), t->size(), clipRange);

    // get max element in Tensor
    float max = 0;
    thrust::device_ptr<float> d_ptr(t->data());
    if (clipRange < 0.0) {
      // forced scale
      // gClip<<<blocksSample, threads>>>(t->data(), t->size(), -clipRange);
      max = -clipRange;
    } else if (isMax) {
      // get max element as scale
      max = *(thrust::max_element(d_ptr, d_ptr + t->size()));
      float min = *(thrust::min_element(d_ptr, d_ptr + t->size())) * -1;
      max = std::max(max, min);
    } else {
      // align mean as scale
      float data_avg = thrust::reduce(d_ptr,
                                    d_ptr + t->size(),
                                    (float) 0,
                                    absolute_value<float>());
      data_avg /= t->size();

      float centers_avg =  std::pow(base, -bit + 2);
      max = data_avg / centers_avg;

      // std. dev as scale
      float norm = thrust::transform_reduce(d_ptr, d_ptr + t->size(),
                                      square<float>(),
                                      (float) 0,
                                      thrust::plus<float>());
      max = 2.58 * std::sqrt(norm / t->size());
    }
   
    static int s = 0;
    // max adjustment 
    for (int i=0;i< kMeanStep;i++) {
      if (id == 0 && s++ < 100) LOG(info, "max adjust step {} = {} ", i, max);
 
      gQuantize<<<blocksSample, threads>>>(t->data(), delta[id]->data(), t->size(), (1<<(bit-1)) - 1, base, max);

      thrust::device_ptr<float> delta_ptr(delta[id]->data());
      float delta_top = thrust::inner_product(delta_ptr, delta_ptr + t->size(), d_ptr, 0.0f);
      float delta_btm = thrust::inner_product(delta_ptr, delta_ptr + t->size(), delta_ptr, 0.0f);
      max = delta_top / delta_btm;
    }

    // compress
    gQuantize<<<blocksSample, threads>>>(t->data(), NULL, t->size(), (1<<(bit-1)) - 1, base, max);
  }
}
