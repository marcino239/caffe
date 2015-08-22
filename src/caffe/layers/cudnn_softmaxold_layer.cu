#ifdef USE_CUDNN
#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNSoftmaxOldLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  CUDNN_CHECK(cudnnSoftmaxOldForward(handle_, CUDNN_SOFTMAXOLD_ACCURATE,
      CUDNN_SOFTMAXOLD_MODE_CHANNEL,
      bottom_desc_, bottom_data, top_desc_, top_data));
}

template <typename Dtype>
void CuDNNSoftmaxOldLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    CUDNN_CHECK(cudnnSoftmaxOldBackward(handle_, CUDNN_SOFTMAXOLD_ACCURATE,
        CUDNN_SOFTMAXOLD_MODE_CHANNEL,
        top_desc_, top_data, top_desc_, top_diff, bottom_desc_, bottom_diff));
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNSoftmaxOldLayer);

}  // namespace caffe
#endif
