#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MLLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0))
      << "Inputs must have same num.";
  CHECK_EQ(bottom[0]->shape(1), 2*bottom[1]->shape(1))
      << "Inputs must have the 2x dimension.";
}

template <typename Dtype>
void MLLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* pred = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();

  Dtype loss = 0.0;
  int dim = bottom[1]->shape(1);
  for (int i = 0; i < dim; ++i) {
    Dtype mu = pred[i];
    Dtype sigma = pred[i+dim];
    CHECK_GT(sigma, 0) << "Sigma is " << sigma << ", must be > 0";
    Dtype x = target[i];
    loss += 0.5 * (pow(mu-x,2)/pow(sigma,2)) + log(sigma);
  }

  //Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void MLLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype* diff = bottom[0]->mutable_cpu_diff();
  const Dtype* pred = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();

  int dim = bottom[1]->shape(1);
  Dtype loss_weight = top[0]->cpu_diff()[0];

  for (int i = 0; i < dim; ++i) {
    Dtype mu = pred[i];
    Dtype sigma = pred[i+dim];
    Dtype x = target[i];

    diff[i] = (mu-x)/pow(sigma,2) * loss_weight;
    diff[i+dim] = (-pow(mu-x,2)/pow(sigma,3) + 1.0/sigma) * loss_weight;
  }

}

INSTANTIATE_CLASS(MLLossLayer);
REGISTER_LAYER_CLASS(MLLoss);

}  // namespace caffe
