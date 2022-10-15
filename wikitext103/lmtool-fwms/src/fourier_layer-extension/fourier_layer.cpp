#include <torch/extension.h>
#include <torch/script.h>
#include <cmath>
#include <omp.h>
#include <iostream>
using namespace std;
#define _USE_MATH_DEFINES

const float eps = 1e-31f;

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
	// head_q: size =[qlen, bsz, n_head, d_head]
	// head_k: size =[klen, bsz, n_head, d_head]
	int n_head     = head_q.size(2);
	int bsz        = head_q.size(1);
	int qlen       = head_q.size(0);
	int klen       = head_k.size(0);
	int d_head     = head_k.size(3);
  
	auto Y = torch::zeros( {qlen, klen, bsz, n_head}, torch::kF32 );
  
	float* p_Y       = Y.data<float>();         // [qlen, klen, bsz, n_head]
  	float* p_head_q  = head_q.data<float>();    // [qlen, bsz, n_head, d_head]
	float* p_head_k  = head_k.data<float>();    // [klen, bsz, n_head, d_head]
	float* p_paramR  = paramR.data<float>();    // 1
	
	#pragma omp parallel for default(shared) private(n,b)
	for(int n=0; n<n_head; n++){
	  for(int b=0; b<bsz; b++){

		for(int q=0; q<qlen; q++){
		  for(int k=0; k<klen;k++){
			float& result = p_Y[q*klen*bsz*n_head + k*bsz*n_head + b*n_head + n];
			float* p_head_q_i = p_head_q + q*bsz*n_head*d_head + b*n_head*d_head + n*d_head;
			float* p_head_k_i = p_head_k + k*bsz*n_head*d_head + b*n_head*d_head + n*d_head ;
			result=1.0f;
			//sum on d
			for(int d=0; d<d_head; d++){	
			  //float diff = ( p_head_q[n,b,q,d] - p_head_k[n,b,k,d] ) 
						 
			  float diff = ( p_head_q_i[d] - p_head_k_i[d] ) * p_paramR[0];
						
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
	// head_q: size =[qlen, bsz, n_head, d_head]
	// head_k: size =[klen, bsz, n_head, d_head]
	int n_head     = head_q.size(2);
	int bsz        = head_q.size(1);
	int qlen       = head_q.size(0);
	int klen       = head_k.size(0);
	int d_head     = head_k.size(3);
	
	auto grad_head_k = torch::zeros( {klen, bsz, n_head, d_head}, torch::kF32 );
	auto grad_head_q = torch::zeros( {qlen, bsz, n_head, d_head}, torch::kF32 );
	auto grad_paramR = torch::zeros( {1}, torch::kF32 );
	
  
	float* p_grad_Y      = grad_Y.data<float>();      //[qlen, klen, bsz, n_head]
	float* p_Y           = Y.data<float>();           //[qlen, klen, bsz, n_head]
  	float* p_head_q      = head_q.data<float>();      //[qlen, bsz, n_head, d_head]
	float* p_head_k      = head_k.data<float>();      //[klen, bsz, n_head, d_head]
	float* p_paramR      = paramR.data<float>();  // 1
	float* p_grad_paramR = grad_paramR.data<float>();  // 1
    float* p_grad_head_q = grad_head_q.data<float>(); //[qlen, bsz, n_head, d_head]
	float* p_grad_head_k = grad_head_k.data<float>(); //[klen, bsz, n_head, d_head]
	
	//compute grad_head_q
	#pragma omp parallel for default(shared) private(n,b)
	for(int n=0; n<n_head; n++){
	  for(int b=0; b<bsz; b++){
/* 		float* p_head_q_i      = p_head_q + n*bsz*qlen*d_head + b*qlen*d_head;
		float* p_head_k_i      = p_head_k + n*bsz*klen*d_head + b*klen*d_head;
		float* p_grad_head_q_i = p_grad_head_q + n*bsz*qlen*d_head + b*qlen*d_head;
		float* p_grad_Y_i      = p_grad_Y + n*bsz*qlen*klen + b*qlen*klen;
		float* p_Y_i           = p_Y      + n*bsz*qlen*klen + b*qlen*klen; */
			
		for(int q=0; q<qlen; q++){
		  for(int d=0; d<d_head; d++){
			float& result = p_grad_head_q[q*bsz*n_head*d_head + b*n_head*d_head + n*d_head + d];
			float& p_head_q_i = p_head_q[q*bsz*n_head*d_head + b*n_head*d_head + n*d_head + d];
			float* p_head_k_i = p_head_k + b*n_head*d_head + n*d_head + d;
			result = 0;
			float* p_Y_i      = p_Y      + q*klen*bsz*n_head  + b*n_head + n;
			float* p_grad_Y_i = p_grad_Y + q*klen*bsz*n_head  + b*n_head + n;
			//sum on k
			for(int k=0; k<klen; k++){		
			  float diff = ( p_head_q_i - p_head_k_i[0] ) *p_paramR[0];
			  p_head_k_i += bsz*n_head*d_head;
						
			  if(abs(diff)<eps) diff=0;
			  else diff =  1.0f/tanf(diff) - 1.0f/diff  ;

			  result     += diff * p_Y_i[0] * p_grad_Y_i[0];
			  p_Y_i      += bsz*n_head;
			  p_grad_Y_i += bsz*n_head;
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
/* 		float* p_head_q_i      = p_head_q + n*bsz*qlen*d_head + b*qlen*d_head;
		float* p_head_k_i      = p_head_k + n*bsz*klen*d_head + b*klen*d_head;
		float* p_grad_head_k_i = p_grad_head_k + n*bsz*klen*d_head + b*klen*d_head;
		float* p_grad_Y_i      = p_grad_Y + n*bsz*qlen*klen + b*qlen*klen;
		float* p_Y_i           = p_Y      + n*bsz*qlen*klen + b*qlen*klen; */
			
		for(int k=0; k<klen; k++){
		  for(int d=0; d<d_head; d++){
			float& result = p_grad_head_k[k*bsz*n_head*d_head + b*n_head*d_head + n*d_head + d];
			float* p_head_q_i = p_head_q + b*n_head*d_head + n*d_head + d;
			float& p_head_k_i = p_head_k[k*bsz*n_head*d_head + b*n_head*d_head + n*d_head + d];
			float* p_Y_i      = p_Y      + k*bsz*n_head + b*n_head + n;
			float* p_grad_Y_i = p_grad_Y + k*bsz*n_head + b*n_head + n;
			result = 0;
			//product on q
			for(int q=0; q<qlen; q++){		
			  float diff = ( p_head_q_i[0] - p_head_k_i ) *p_paramR[0];
			  p_head_q_i += bsz*n_head*d_head;
						
			  if(abs(diff)<eps) diff=1;
			  else diff = 1.0f/tanf(diff) - 1.0f/diff  ;
				
			  result     -= diff * p_Y_i[0] * p_grad_Y_i[0];	
			  p_Y_i      += klen*bsz*n_head;
			  p_grad_Y_i += klen*bsz*n_head;
			}
			result *= p_paramR[0];
		  }
		}
			
	  }
	}	
	
	//compute grad_p
	p_grad_paramR[0]=0;
	#pragma omp parallel for default(shared) private(n,b)
	for(int n=0; n<n_head; n++){
	  for(int b=0; b<bsz; b++){
/* 		float* p_head_q_i    = p_head_q + n*bsz*qlen*d_head + b*qlen*d_head;
		float* p_head_k_i    = p_head_k + n*bsz*klen*d_head + b*klen*d_head;
		float* p_grad_Y_i    = p_grad_Y + n*bsz*qlen*klen + b*qlen*klen;
		float* p_Y_i         = p_Y      + n*bsz*qlen*klen + b*qlen*klen; */
			
		for(int q=0; q<qlen; q++){	
		  for(int k=0; k<klen; k++){
			//sum on d
			float* p_head_q_i = p_head_q + q*bsz*n_head*d_head + b*n_head*d_head + n*d_head;
			float* p_head_k_i = p_head_k + k*bsz*n_head*d_head + b*n_head*d_head + n*d_head;
			float temp = 0;
			for(int d=0; d<d_head; d++){
			  float diff = ( p_head_q_i[d] - p_head_k_i[d] );
						
			  if(abs(diff)<eps) temp += 1.0f/p_paramR[0];
			  else              temp += diff / tanf(p_paramR[0]*diff) ;
			}
	p_grad_paramR[0] += temp*p_grad_Y[q*klen*bsz*n_head + k*bsz*n_head + b*n_head + n] * 					     p_Y[q*klen*bsz*n_head + k*bsz*n_head + b*n_head + n];
		  }
		}
			
	  }
	}	
	
	return {grad_head_q, grad_head_k, grad_paramR};
}
 
 


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fourier_layer_forward, "FOURIER_LAYER forward");
  m.def("backward", &fourier_layer_backward, "FOURIER_LAYER backward");
  m.def("get_max_threads", &omp_get_max_threads, "Returns max number of threads");
  m.def("set_num_threads", &omp_set_num_threads, "Set number of threads");
  //m.def("sum_thread_ids", &sum_thread_ids, "Adds the id of threads");
}



/*
for(int n=0; n<n_head; n++){
  for(int b=0; b<bsz; b++){
	for(int q=0; q<qlen; q++){
	  for(int d=0; d<d_head; d++){
		float & result = p_grad_head_q[q*bsz*n_head*d_head + b*n_head*d_head + n*d_head + d];
		result = 0;
		//sum on k
		for(int k=0; k<klen; k++){		
float diff = ( p_head_q[q*bsz*n_head*d_head + b*n_head*d_head + n*d_head + d] - 
			   p_head_k[k*bsz*n_head*d_head + b*n_head*d_head + n*d_head + d] ) *p_paramR[0];
					
		  if(abs(diff)<eps) diff=0;
		  else diff =  1.0f/tanf(diff) - 1.0f/diff  ;

		  result += diff * p_Y[q*klen*bsz*n_head + k*bsz*n_head + b*n_head + n] *            p_grad_Y[q*klen*bsz*n_head + k*bsz*n_head + b*n_head + n];
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
		
	for(int k=0; k<klen; k++){
	  for(int d=0; d<d_head; d++){
		float & result = p_grad_head_k[k*bsz*n_head*d_head + b*n_head*d_head + n*d_head + d];
		result = 0;
		//product on q
		for(int q=0; q<qlen; q++){		
float diff = ( p_head_q[q*bsz*n_head*d_head + b*n_head*d_head + n*d_head + d] - 
			   p_head_k[k*bsz*n_head*d_head + b*n_head*d_head + n*d_head + d] ) *p_paramR[0];
					
		  if(abs(diff)<eps) diff=1;
		  else diff = 1.0f/tanf(diff) - 1.0f/diff  ;
			
		  result -= diff * p_Y[q*klen*bsz*n_head + k*bsz*n_head + b*n_head + n] *            p_grad_Y[q*klen*bsz*n_head + k*bsz*n_head + b*n_head + n];	
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
		
	for(int q=0; q<qlen; q++){	
	  for(int k=0; k<klen; k++){
		//sum on d
		float temp = 0;
		for(int d=0; d<d_head; d++){
float diff = ( p_head_q[q*bsz*n_head*d_head + b*n_head*d_head + n*d_head + d] - 
			   p_head_k[k*bsz*n_head*d_head + b*n_head*d_head + n*d_head + d] );
					
		  if(abs(diff)<eps) temp += 1.0f/p_paramR[0];
		  else              temp += diff / tanf(p_paramR[0]*diff) ;
		}
r_grad_paramR += temp * p_grad_Y[q*klen*bsz*n_head + k*bsz*n_head + b*n_head + n] * 					        p_Y[q*klen*bsz*n_head + k*bsz*n_head + b*n_head + n];
	  }
	}
		
  }
}	

*/