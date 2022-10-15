#include <torch/extension.h>
#include <torch/script.h>
#include <cmath>
#include <omp.h>
#include <iostream>
using namespace std;
#define _USE_MATH_DEFINES

const float eps = 1e-30f;

// head_q:	[n_head, bsz, qlen, d_head] -> [n_head, bsz, qlen, 1,    d_head]
// head_k:	[n_head, bsz, klen, d_head] -> [n_head, bsz, 1,    klen, d_head]
// head_q - head_k:                        [n_head, bsz, qlen, klen, d_head]
// sum on dim4                          -> [n_head, bsz, qlen, klen]

// QK_distance0 = (head_q.unsqueeze(3) - head_k.unsqueeze(2)) * self.paramR / pi 
// QK_distance0 = torch.sinc(QK_distance0) * self.paramR
// attn_prob = torch.prod(QK_distance0, dim=4)  
//        [n_head, bsz, qlen, klen]

torch::Tensor fourier_layer_forward(
		torch::Tensor head_q,
		torch::Tensor head_k,
		torch::Tensor paramR) 
{
	int n_head     = head_q.size(0);
	int bsz        = head_q.size(1);
	int qlen       = head_q.size(2);
	int klen       = head_k.size(2);
	int d_head     = head_k.size(3);
	//cout << n_head <<", " << bsz <<", " << qlen << ", " << klen << ", " << d_head<<endl;
  
	auto Y = torch::zeros( {n_head, bsz, qlen, klen}, torch::kF32 );
  
	float* p_Y       = Y.data<float>();         // [n_head, bsz, qlen, klen]
  	float* p_head_q  = head_q.data<float>();    // [n_head, bsz, qlen, d_head]
	float* p_head_k  = head_k.data<float>();    // [n_head, bsz, klen, d_head]
	float* p_paramR  = paramR.data<float>();    // 1
	
	#pragma omp parallel for default(shared) private(n,b)
	for(int n=0; n<n_head; n++){
		for(int b=0; b<bsz; b++){
			float* p_head_q_i = p_head_q + n*bsz*qlen*d_head + b*qlen*d_head;
			float* p_head_k_i = p_head_k + n*bsz*klen*d_head + b*klen*d_head;
			for(int q=0; q<qlen; q++){
				for(int k=0; k<klen;k++){
					float & result = p_Y[n*bsz*qlen*klen + b*qlen*klen + q*klen + k];
					result=1.0f;
					//sum on d
					for(int d=0; d<d_head; d++){	
						//float diff = ( p_head_q[n,b,q,d] - p_head_k[n,b,k,d] ) * 
						
						float diff = ( p_head_q_i[q*d_head + d] - 
						               p_head_k_i[k*d_head + d] ) * 
							           p_paramR[0];
						
						if(abs(diff)<eps) diff=1;
						else diff = sinf(diff)/diff;
						
						
						result = result * diff ;
					}
					result *= powf(p_paramR[0],d_head);
				}
			}
		}
	}
	return Y;
}


std::vector<torch::Tensor> fourier_layer_backward(
//torch::Tensor my_linear_backward(
		torch::Tensor grad_Y,
		torch::Tensor head_q,
		torch::Tensor head_k,
		torch::Tensor paramR,
		torch::Tensor Y)
{
	int n_head     = head_q.size(0);
	int bsz        = head_q.size(1);
	int qlen       = head_q.size(2);
	int klen       = head_k.size(2);
	int d_head     = head_k.size(3);
	
	auto grad_head_k = torch::zeros( {n_head, bsz, klen, d_head}, torch::kF32 );
	auto grad_head_q = torch::zeros( {n_head, bsz, qlen, d_head}, torch::kF32 );
	auto grad_paramR = torch::zeros( {1}, torch::kF32 );
  
	float* p_grad_Y      = grad_Y.data<float>();      //[n_head, bsz, qlen, klen]
	float* p_Y           = Y.data<float>();           //[n_head, bsz, qlen, klen]
  	float* p_head_q      = head_q.data<float>();      //[n_head, bsz, qlen, d_head]
	float* p_head_k      = head_k.data<float>();      //[n_head, bsz, klen, d_head]
	float* p_paramR      = paramR.data<float>();  // 1
	float& r_grad_paramR = grad_paramR.data<float>()[0];  // 1
    float* p_grad_head_q = grad_head_q.data<float>(); //[n_head, bsz, qlen, d_head]
	float* p_grad_head_k = grad_head_k.data<float>(); //[n_head, bsz, klen, d_head]
	
	//compute grad_head_q
	#pragma omp parallel for default(shared) private(n,b)
	for(int n=0; n<n_head; n++){
		for(int b=0; b<bsz; b++){
			float* p_head_q_i      = p_head_q + n*bsz*qlen*d_head + b*qlen*d_head;
			float* p_head_k_i      = p_head_k + n*bsz*klen*d_head + b*klen*d_head;
			float* p_grad_head_q_i = p_grad_head_q + n*bsz*qlen*d_head + b*qlen*d_head;
			float* p_grad_Y_i      = p_grad_Y + n*bsz*qlen*klen + b*qlen*klen;
			float* p_Y_i           = p_Y      + n*bsz*qlen*klen + b*qlen*klen;
			
			for(int q=0; q<qlen; q++){
				for(int d=0; d<d_head; d++){
					float & result = p_grad_head_q_i[q*d_head + d];
					result = 0;
					//sum on k
					for(int k=0; k<klen; k++){		
						float diff = ( p_head_q_i[q*d_head + d] - 
						               p_head_k_i[k*d_head + d] ) *p_paramR[0];
						
						if(abs(diff)<eps) diff=0;
						else diff =  1.0f/tanf(diff) - 1.0f/diff  ;

						result += diff * p_Y_i[q*klen + k] * p_grad_Y_i[q*klen + k];
					}
					result *= p_paramR[0];
				}
			}
			
		}
	}
	
	// compute grad_head_k
	#pragma omp parallel for default(shared) private(n,b)
	for(int n=0; n<n_head; n++){
		for(int b=0; b<bsz; b++){
			float* p_head_q_i      = p_head_q + n*bsz*qlen*d_head + b*qlen*d_head;
			float* p_head_k_i      = p_head_k + n*bsz*klen*d_head + b*klen*d_head;
			float* p_grad_head_k_i = p_grad_head_k + n*bsz*klen*d_head + b*klen*d_head;
			float* p_grad_Y_i      = p_grad_Y + n*bsz*qlen*klen + b*qlen*klen;
			float* p_Y_i           = p_Y      + n*bsz*qlen*klen + b*qlen*klen;
			
			for(int k=0; k<klen; k++){
				for(int d=0; d<d_head; d++){
					float & result = p_grad_head_k_i[k*d_head + d];
					result = 0;
					//product on q
					for(int q=0; q<qlen; q++){		
						float diff = ( p_head_q_i[q*d_head + d]   - 
						               p_head_k_i[k*d_head + d] ) * p_paramR[0] ;
						
						if(abs(diff)<eps) diff=1;
						else diff = 1.0f/tanf(diff) - 1.0f/diff  ;
						
						result -= diff * p_Y_i[q*klen + k] * p_grad_Y_i[q*klen + k];
					}
					result *= p_paramR[0];
				}
			}
			
		}
	}	
	
	//compute grad_p
	r_grad_paramR=0;
	#pragma omp parallel for default(shared) private(n,b)
	for(int n=0; n<n_head; n++){
		for(int b=0; b<bsz; b++){
			float* p_head_q_i    = p_head_q + n*bsz*qlen*d_head + b*qlen*d_head;
			float* p_head_k_i    = p_head_k + n*bsz*klen*d_head + b*klen*d_head;
			float* p_grad_Y_i    = p_grad_Y + n*bsz*qlen*klen + b*qlen*klen;
			float* p_Y_i         = p_Y      + n*bsz*qlen*klen + b*qlen*klen;
			
			for(int q=0; q<qlen; q++){	
				for(int k=0; k<klen; k++){
					//sum on d
					float temp = 0;
					for(int d=0; d<d_head; d++){
						float diff = ( p_head_q_i[q*d_head + d]   - 
						               p_head_k_i[k*d_head + d] )  ;
						
						if(abs(diff)<eps) temp += 1.0f/p_paramR[0];
						else              temp += diff / tanf(p_paramR[0]*diff) ;
					}
					r_grad_paramR += temp * p_grad_Y_i[q*klen + k] * p_Y_i[q*klen + k];
				}
			}
			
		}
	}	
	
	return {grad_head_q, grad_head_k, grad_paramR};
}
  
  /*
  auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));
  auto gates = gate_weights.chunk(3, 1);

  auto input_gate = torch::sigmoid(gates[0]);
  auto output_gate = torch::sigmoid(gates[1]);
  auto candidate_cell = torch::elu(gates[2], 1.0);
  auto new_cell = old_cell + candidate_cell * input_gate;
  auto new_h = torch::tanh(new_cell) * output_gate;
  return {new_h, new_cell, input_gate, output_gate, candidate_cell, X, gate_weights};
  */


/*
std::vector<torch::Tensor> rbf_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gate_weights,
    torch::Tensor weights) {

  auto d_output_gate = torch::tanh(new_cell) * grad_h;
  auto d_tanh_new_cell = output_gate * grad_h;
  auto d_new_cell = d_tanh(new_cell) * d_tanh_new_cell + grad_cell;

  auto d_old_cell = d_new_cell;
  auto d_candidate_cell = input_gate * d_new_cell;
  auto d_input_gate = candidate_cell * d_new_cell;

  auto gates = gate_weights.chunk(3, 1);
  d_input_gate *= d_sigmoid(gates[0]);
  d_output_gate *= d_sigmoid(gates[1]);
  d_candidate_cell *= d_elu(gates[2]);

  auto d_gates =
      torch::cat({d_input_gate, d_output_gate, d_candidate_cell},1);

  auto d_weights = d_gates.t().mm(X);
  auto d_bias = d_gates.sum(0, true);

  auto d_X = d_gates.mm(weights);
  const auto state_size = grad_h.size(1);
  auto d_old_h = d_X.slice(1, 0, state_size);
  auto d_input = d_X.slice(1, state_size);
  
  return {d_old_h, d_input, d_weights, d_bias, d_old_cell};
}
*/
  


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fourier_layer_forward, "FOURIER_LAYER forward");
  m.def("backward", &fourier_layer_backward, "FOURIER_LAYER backward");
  m.def("get_max_threads", &omp_get_max_threads, "Returns max number of threads");
  m.def("set_num_threads", &omp_set_num_threads, "Set number of threads");
  //m.def("sum_thread_ids", &sum_thread_ids, "Adds the id of threads");
}
