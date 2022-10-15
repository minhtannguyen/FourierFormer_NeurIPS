#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

// CUDA forward declarations

torch::Tensor fourier_layer_cuda_forward(
		const torch::Tensor& head_q,
		const torch::Tensor& head_k,
		const torch::Tensor& paramR);

std::vector<torch::Tensor> fourier_layer_cuda_backward(
		const torch::Tensor& grad_Y,
		const torch::Tensor& head_q,
		const torch::Tensor& head_k,
		const torch::Tensor& paramR,
		const torch::Tensor& Y);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor fourier_layer_forward(
		const torch::Tensor& head_q,
		const torch::Tensor& head_k,
		const torch::Tensor& paramR) {
  //std::cout <<"start kernel:..."<<std::endl;
  
  CHECK_INPUT(head_q);
  CHECK_INPUT(head_k);
  CHECK_INPUT(paramR);
  //std::cout <<"done checking input."<<std::endl;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(head_q));
  return fourier_layer_cuda_forward(head_q,head_k,paramR);
}

std::vector<torch::Tensor> fourier_layer_backward(
		const torch::Tensor& grad_Y,
		const torch::Tensor& head_q,
		const torch::Tensor& head_k,
		const torch::Tensor& paramR,
		const torch::Tensor& Y)  {
  CHECK_INPUT(grad_Y);
  CHECK_INPUT(head_q);
  CHECK_INPUT(head_k);
  CHECK_INPUT(paramR);
  CHECK_INPUT(Y);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(head_q));
  return fourier_layer_cuda_backward(grad_Y, head_q, head_k, paramR, Y);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward" , &fourier_layer_forward , "FOURIER_LAYER forward  (CUDA)");
  m.def("backward", &fourier_layer_backward, "FOURIER_LAYER backward (CUDA)");
}