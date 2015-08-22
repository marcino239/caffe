#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FixedBatchNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  moving_average_fraction_ = this->layer_param_.batch_norm_param().moving_average_fraction();
  use_global_stats_ = this->layer_param_.batch_norm_param().use_global_stats();
  channels_ = bottom[0]->channels();
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(3);
    this->blobs_[0].reset(new Blob<Dtype>(1, channels_, 1, 1));
    this->blobs_[1].reset(new Blob<Dtype>(1, channels_, 1, 1));
    this->blobs_[2].reset(new Blob<Dtype>(1, 1, 1, 1));
    for(int i = 0; i<3; ++i){
      caffe_gpu_set(this->blobs_[i]->count(),Dtype(0),this->blobs_[i]->mutable_gpu_data());
    }
  }
}

template <typename Dtype>
void FixedBatchNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
  CHECK_EQ(bottom[0]->channels(),channels_);
  mean_.Reshape(1, bottom[0]->channels(), 1, 1);
  variance_.Reshape(1, bottom[0]->channels(), 1, 1);
  temp_.ReshapeLike(*bottom[0]);
  num_sum_.Reshape(bottom[0]->num(),1,1,1);
  if(bottom[0]->width()!=sum_multiplier_.width()||bottom[0]->height()!=sum_multiplier_.height()){
    sum_multiplier_.Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
    Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
    caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
  }
  if(bottom[0]->num()!=num_by_chans_.num()||bottom[0]->channels()!=num_by_chans_.channels()){
    num_by_chans_.Reshape(bottom[0]->num(),bottom[0]->channels(),1,1);
    caffe_set(num_sum_.count(),Dtype(1),num_sum_.mutable_cpu_data());
  }
}

template <typename Dtype>
void FixedBatchNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  LOG(FATAL) << "Unimplemented";
}

template <typename Dtype>
void FixedBatchNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  LOG(FATAL) << "Unimplemented";
}


#ifdef CPU_ONLY
STUB_GPU(FixedBatchNormLayer);
#endif

INSTANTIATE_CLASS(FixedBatchNormLayer);
REGISTER_LAYER_CLASS(FixedBatchNorm);
}  // namespace caffe
