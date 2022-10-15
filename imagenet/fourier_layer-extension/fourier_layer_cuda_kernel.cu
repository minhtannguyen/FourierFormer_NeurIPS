#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cooperative_groups.h>
#define _USE_MATH_DEFINES

using namespace cooperative_groups;

#if __CUDA_ARCH__ < 600
template <typename T>
__device__ double atomicAdd(T* address, T val)
{
    unsigned long long int* address_as_ull =
    (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
        __double_as_longlong(val +
        __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

namespace{
//const float eps = 1e-31f;

template <typename scalar_t>
__global__ void fourier_layer_cuda_forward_kernel(
          scalar_t* p_Y,
	const scalar_t* __restrict__ p_head_q,
    const scalar_t* __restrict__ p_head_k,
    const scalar_t* __restrict__ p_paramR,
	const size_t n_head,
	const size_t bsz, 
	const size_t qlen, 
	const size_t klen,
	const size_t d_head,
	const size_t N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int b = blockIdx.y;

  if (i < N){
	  
	//extract n,b,q,k
	// size = [qlen, klen bsz, n_head]
	int q =   i / (klen*n_head);
	int k = ( i % (klen*n_head) ) / (n_head) ;
	int n =   i % n_head;
	
	scalar_t& result = p_Y[q*klen*bsz*n_head + k*bsz*n_head + b*n_head + n];
	//scalar_t& result = p_Y[i];
	
	const scalar_t* p_head_q_i = p_head_q + (q*bsz*n_head + b*n_head + n ) * d_head;
	const scalar_t* p_head_k_i = p_head_k + (k*bsz*n_head + b*n_head + n ) * d_head;
	result=1.0f;
	
	//sum on d
	scalar_t diff;
	for(int d=0; d<d_head; d++){	
	  //float diff = ( p_head_q[n,b,q,d] - p_head_k[n,b,k,d] ) 	 
	  diff = ( p_head_q_i[d] - p_head_k_i[d] ) * p_paramR[d] ;
				
	  if(abs(diff)<1e-30f) diff=1;
	  else diff = sinf(diff)/diff;
								
	  result = result * diff ;
	}
	
	//result *= __powf(p_paramR[0],d_head);
	
	
  }//end if	
  
}

/*
template <typename scalar_t>
__device__ scalar_t reduce_sum(thread_group g, float *temp, scalar_t val){
    int lane = g.thread_rank();
    for (int i = g.size()/2; i > 0; i /= 2)    {
        temp[lane] = val;
        g.sync(); // wait for all threads to store
        if(lane<i) val += temp[lane + i];
		//g.sync();
		if(lane==0 && i%2==1 && i>2) val+=(temp[i-1]+temp[2*i-1]);
        g.sync(); // wait for all threads to load
    }
    return val; // note: only thread 0 will return full sum
}
*/

template <typename scalar_t>
__device__ scalar_t reduce_sum(thread_group g, float *temp, scalar_t val){
    int lane = g.thread_rank();
	temp[lane] = val;
	g.sync();
    for (int i = g.size()/2; i > 0; i /= 2)    {
		if( lane==0 && i%2==1 && i>2) temp[lane]+=(temp[i-1]+temp[2*i-1]);
		if(lane<i) temp[lane] += temp[lane + i];
		
        // wait for all threads to load
		g.sync();
    }
    return temp[0]; // note: only thread 0 will return full sum
}

// compute grad_head_q
template <typename scalar_t>
__global__ void fourier_layer_cuda_backward_kernel_q(
    const scalar_t* p_grad_Y,
	const scalar_t* __restrict__ p_head_q,
    const scalar_t* __restrict__ p_head_k,
    const scalar_t* __restrict__ p_paramR,
	const scalar_t* __restrict__ p_Y,
	      scalar_t* __restrict__ p_grad_head_q,
		  scalar_t* __restrict__ p_grad_head_k,
		  scalar_t* __restrict__ p_grad_paramR,
	const size_t n_head,
	const size_t bsz, 
	const size_t qlen, 
	const size_t klen,
	const size_t qleni,
	const size_t kleni,
	const size_t d_head,
	const size_t N_q) 
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = blockIdx.y;
  extern __shared__ float temp_q[]; //size = qlen*klen*d_head*n_head

  if (i < N_q){
	// size(head_q) = [qlen, bsz, n_head, d_head]
	//int ki = threadIdx.x;
	//int dqi =  blockIdx.x ;
	//int qi =   dqi / (d_head);
	//int k =  (i % (klen*d_head) ) / d_head;
	//int d =   dqi %  d_head;
	int ki =   i / (qleni*d_head) ;
	int qi = ( i % (qleni*d_head) ) / d_head;
	int d  = i % d_head;
	

	
	scalar_t* grad_head_q = p_grad_head_q + 0*bsz*n_head*d_head + 0*n_head*d_head + n*d_head + d;
	scalar_t* grad_head_k = p_grad_head_k + 0*bsz*n_head*d_head + 0*n_head*d_head + n*d_head + d;
	
	const scalar_t* p_head_q_i = p_head_q + 0*bsz*n_head*d_head + 0*n_head*d_head + n*d_head + d;
	const scalar_t* p_head_k_i = p_head_k + 0*bsz*n_head*d_head + 0*n_head*d_head + n*d_head + d;
	const scalar_t* p_Y_i      = p_Y      + 0*klen*bsz*n_head + 0*bsz*n_head + 0*n_head + n;
	const scalar_t* p_grad_Y_i = p_grad_Y + 0*klen*bsz*n_head + 0*bsz*n_head + 0*n_head + n;
	//sum on k
	scalar_t diff, temp,  grad_paramR=0;
	//for(int n=0; n<n_head; n++){	
	for( int qii=qi; qii<qlen; qii+=qleni){
	  scalar_t sum_k = 0;
	  for( int kii=ki; kii<klen; kii+=kleni){
	
	    diff = ( p_head_q_i[qii*bsz*n_head*d_head] - p_head_k_i[kii*bsz*n_head*d_head] ) *p_paramR[d];
	    //p_head_k_i += bsz*n_head*d_head;
				
	    if(abs(diff)<1.0e-30f) temp=0;
	    else temp =  1.0f/tanf(diff) - 1.0f/diff  ;
	  
	    temp *= p_Y_i[qii*klen*bsz*n_head + kii*bsz*n_head] * p_grad_Y_i[qii*klen*bsz*n_head + kii*bsz*n_head] * p_paramR[d] ;

	    //auto g = this_thread_block();
        //scalar_t block_sum = reduce_sum(g, temp_q, temp);
	    //if (g.thread_rank() == 0) grad_head_q[qii*bsz*n_head*d_head + 0*d_head]= block_sum;//block_sum;
	    //atomicAdd( grad_head_q + qii*bsz*n_head*d_head ,  temp );
	    atomicAdd( grad_head_k + kii*bsz*n_head*d_head , -temp );
	  
	    sum_k += temp;
	  
	    //size = qlen*klen*d_head*n_head	  
	    //grad_head_q[qii*bsz*n_head*d_head] += temp;
	    //grad_head_k[kii*bsz*n_head*d_head] -= temp;
	    grad_paramR += temp*diff ;
	    //p_Y_i      += bsz*n_head;
	    //p_grad_Y_i += bsz*n_head;
	  
	  }
	  atomicAdd( grad_head_q + qii*bsz*n_head*d_head ,  sum_k );
	  //atomicAdd( &grad_head_k[0*d_head] , -temp_k );
	  //grad_head_k[0] -= temp;
	}
	
	atomicAdd( &p_grad_paramR[d], grad_paramR/(p_paramR[d]*p_paramR[d]) );
	//p_grad_paramR[0] += grad_paramR/(p_paramR[0]*p_paramR[0]) ;
  }
  
}


// compute grad_head_k
template <typename scalar_t>
__global__ void fourier_layer_cuda_backward_kernel_k(
    const scalar_t* p_grad_Y,
	const scalar_t* __restrict__ p_head_q,
    const scalar_t* __restrict__ p_head_k,
    const scalar_t* __restrict__ p_paramR,
	const scalar_t* __restrict__ p_Y,
	      scalar_t* __restrict__ p_grad_head_k,
	const size_t n_head,
	const size_t bsz, 
	const size_t qlen, 
	const size_t klen,
	const size_t d_head,
	const size_t N_k) 
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N_k){
	// size(head_k) = [klen, bsz, n_head, d_head]
	int k =   i / (bsz*n_head*d_head);
	int b = ( i % (bsz*n_head*d_head) ) / (n_head*d_head) ;
	int n = ( i % (n_head*d_head)     ) / d_head;
	int d =   i %  d_head;

	//float& result = p_grad_head_k[k*bsz*n_head*d_head + b*n_head*d_head + n*d_head + d];
	scalar_t& result = p_grad_head_k[i];
	const scalar_t* p_head_q_i = p_head_q + b*n_head*d_head + n*d_head + d;
	const scalar_t& p_head_k_i = p_head_k[k*bsz*n_head*d_head + b*n_head*d_head + n*d_head + d];
	const scalar_t* p_Y_i      = p_Y      + k*bsz*n_head + b*n_head + n;
	const scalar_t* p_grad_Y_i = p_grad_Y + k*bsz*n_head + b*n_head + n;
	result = 0;
	//product on q
	for(int q=0; q<qlen; q++){		
	  float diff = ( p_head_q_i[0] - p_head_k_i ) *p_paramR[d];
	  p_head_q_i += bsz*n_head*d_head;
				
	  if(abs(diff)<1.0e-30f) diff=1;
	  else diff = 1.0f/tanf(diff) - 1.0f/diff  ;
		
	  result     -= diff * p_Y_i[0] * p_grad_Y_i[0];	
	  p_Y_i      += klen*bsz*n_head;
	  p_grad_Y_i += klen*bsz*n_head;
	}
	result *= p_paramR[d];
			
  }
  
}



// compute grad_paramR
template <typename scalar_t>
__global__ void fourier_layer_cuda_backward_kernel_paramR(
    const scalar_t* p_grad_Y,
	const scalar_t* __restrict__ p_head_q,
    const scalar_t* __restrict__ p_head_k,
    const scalar_t* __restrict__ p_paramR,
	const scalar_t* __restrict__ p_Y,
	      scalar_t* __restrict__ p_grad_paramR,
	const size_t n_head,
	const size_t bsz, 
	const size_t qlen, 
	const size_t klen,
	const size_t d_head,
	const size_t N_paramR) 
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N_paramR){
	// s [qlen,klen, bsz, n_head ]
	int q =   i / (klen*bsz*n_head);
	int k = ( i % (klen*bsz*n_head) ) / (bsz*n_head) ;
	int b = ( i % (bsz*n_head)     ) / n_head;
	int n =   i %  n_head;

	const scalar_t* p_head_q_i = p_head_q + q*bsz*n_head*d_head + b*n_head*d_head + n*d_head;
	const scalar_t* p_head_k_i = p_head_k + k*bsz*n_head*d_head + b*n_head*d_head + n*d_head;
	scalar_t temp;
	for(int d=0; d<d_head; d++){
	  scalar_t diff = ( p_head_q_i[d] - p_head_k_i[d] ) ;
				
	  if(abs(diff)<1.0e-30f) temp = 0.0f;
	  else                   temp = (diff / tanf(p_paramR[d]*diff) - 1.0f/p_paramR[d] );
	  
	  atomicAdd( &p_grad_paramR[d] , temp * p_grad_Y[i] * p_Y[i] );
	}
	
			
  }
  
}


}//end namespace
  
  
// 1. forward  
//template <typename T>
torch::Tensor fourier_layer_cuda_forward(
		const torch::Tensor& head_q,
		const torch::Tensor& head_k,
		const torch::Tensor& paramR)
{ 
  const auto n_head     = head_q.size(2);
  const auto bsz        = head_q.size(1);
  const auto qlen       = head_q.size(0);
  const auto klen       = head_k.size(0);
  const auto d_head     = head_k.size(3);
  
  const auto N = qlen* klen  * n_head;
  
  const int threads = 1024;
  const dim3 blocks((N + threads - 1) / threads, bsz);
  //const int blocks =  (N + threads - 1) / threads ;
  auto dev = head_q.get_device();
  auto options = torch::TensorOptions().dtype(head_q.dtype())
                                       .layout(torch::kStrided)
                                       .device(torch::kCUDA, dev)
                                       .requires_grad(true);
	
  auto Y = torch::zeros( {qlen, klen, bsz, n_head}, options );

  //float* p_Y       = Y.data<float>();         // [qlen, klen, bsz, n_head]
  //float* p_head_q  = head_q.data<float>();    // [qlen, bsz, n_head, d_head]
  //float* p_head_k  = head_k.data<float>();    // [klen, bsz, n_head, d_head]
  //float* p_paramR  = paramR.data<float>();     
  
  
  AT_DISPATCH_FLOATING_TYPES(head_q.type(), "fourier_layer_forward_cuda", ([&] {
    fourier_layer_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
  Y.data<scalar_t>(), 
  head_q.data<scalar_t>(), 
  head_k.data<scalar_t>(), 
  paramR.data<scalar_t>(), 
  n_head, bsz, qlen, klen, d_head, N);
  }));
  //printf("Error in cuda: %s\n", cudaGetLastError());

  return Y;
}


std::vector<torch::Tensor> fourier_layer_cuda_backward(
//torch::Tensor my_linear_backward(
		const torch::Tensor& grad_Y,
		const torch::Tensor& head_q,
		const torch::Tensor& head_k,
		const torch::Tensor& paramR,
		const torch::Tensor& Y)
{
  int n_head     = head_q.size(2);
  int bsz        = head_q.size(1);
  int qlen       = head_q.size(0);
  int klen       = head_k.size(0);
  int d_head     = head_k.size(3);
  
  int stride = 16;
  int qleni = stride, kleni = stride;
  
  const int threads=klen;
  auto const N_q = qleni*d_head*kleni;
  auto const N_k = klen*bsz*n_head*d_head;
  auto const N_paramR = qlen*klen*bsz*n_head;
  const dim3 blocks_q ( (N_q + threads - 1)/threads, n_head );
  const int blocks_k = (N_k + threads - 1)/threads;
  const int blocks_paramR = (N_paramR + threads - 1)/threads;
	
  auto grad_head_q = torch::zeros_like(head_q); //[qlen,bsz,n_head,d_head]
  auto grad_head_k = torch::zeros_like(head_k); //[klen,bsz,n_head,d_head]
  auto grad_paramR = torch::zeros_like(paramR); //[1]
  
  /*
  scalar_t* p_grad_Y = grad_Y.data<scalar_t>();
  scalar_t* p_head_q = head_q.data<scalar_t>(); 
  scalar_t* p_head_k = head_k.data<scalar_t>(); 
  scalar_t* p_paramR = paramR.data<scalar_t>();
  scalar_t* p_Y      =      Y.data<scalar_t>();  
  scalar_t* p_grad_head_q = grad_head_q.data<scalar_t>();
  scalar_t* p_grad_head_k = grad_head_k.data<scalar_t>();
  scalar_t* p_grad_paramR = grad_paramR.data<scalar_t>();
  */
  //sizeof(scalar_t)*qlen*klen*d_head*n_head
  int size_shared = klen;
  for (int b=0; b<bsz; b++){
	//int b = bn/n_head;
	//int n = bn%n_head;
	AT_DISPATCH_FLOATING_TYPES(head_q.type(), "fourier_layer_cuda_backward_kernel_q", 
	([&] {fourier_layer_cuda_backward_kernel_q<scalar_t><<<blocks_q, threads, sizeof(float)*size_shared>>>(
	grad_Y.data<scalar_t>() + b*n_head + 0, 
	head_q.data<scalar_t>() + b*n_head*d_head + 0*d_head + 0, 
	head_k.data<scalar_t>() + b*n_head*d_head + 0*d_head + 0, 
	paramR.data<scalar_t>(),
	     Y.data<scalar_t>() + b*n_head + 0,  
	grad_head_q.data<scalar_t>() + b*n_head*d_head + 0*d_head + 0,
	grad_head_k.data<scalar_t>() + b*n_head*d_head + 0*d_head + 0,
	grad_paramR.data<scalar_t>(),
	n_head, bsz, qlen, klen, qleni, kleni, d_head, N_q);
	}));  
  }
  /*	
  AT_DISPATCH_FLOATING_TYPES(head_k.type(), "fourier_layer_cuda_backward_kernel_k", 
  ([&] {fourier_layer_cuda_backward_kernel_k<scalar_t><<<blocks_k, threads>>>(
  grad_Y.data<scalar_t>(), 
  head_q.data<scalar_t>(), 
  head_k.data<scalar_t>(), 
  paramR.data<scalar_t>(),
  Y.data<scalar_t>(),  
  grad_head_k.data<scalar_t>(),
  n_head, bsz, qlen, klen, d_head, N_k);
  }));  
  
  AT_DISPATCH_FLOATING_TYPES(head_k.type(), "fourier_layer_cuda_backward_kernel_paramR", 
  ([&] {fourier_layer_cuda_backward_kernel_paramR<scalar_t><<<blocks_paramR, threads>>>(
  grad_Y.data<scalar_t>(), 
  head_q.data<scalar_t>(), 
  head_k.data<scalar_t>(), 
  paramR.data<scalar_t>(),
  Y.data<scalar_t>(),  
  grad_paramR.data<scalar_t>(),
  n_head, bsz, qlen, klen, d_head, N_paramR);
  }));    
  */
  return {grad_head_q, grad_head_k, grad_paramR};
}