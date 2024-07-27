#include <stdio.h>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


// const size_t DSIZE = 16384;      // matrix side dimension
const size_t DSIZE = 6;      // matrix side dimension
const int block_size = 32;  // CUDA maximum is 1024
const float element_val = 100;


__global__ void softmax_max(float *A, size_t ds) {

  int idx = threadIdx.x;
  __shared__ float sdata[block_size];
  sdata[idx] = 0.0f;
  float val = 0.0f;

  // Total elements this block is supposed to handle
  int total_elements = ds * ds;
  int start_index = blockIdx.x * ds; // Start index for this block
  int end_index = start_index + ds;  // End index for this block

  __shared__ float max_val;
  max_val = 0.0f;

  // Find the maximum value in the block

  for (int index = start_index + idx; index < end_index; index += blockDim.x) {
    if (index < ds*ds) sdata[idx] = max(A[index], sdata[idx]);
  }

  for(int s = blockDim.x/2; s > 0; s/=2){
    __syncthreads();
    if (idx < s) sdata[idx] = max(sdata[idx], sdata[idx + s]);
  }
  __syncthreads();

  if (idx == 0) max_val = sdata[0];
  __syncthreads();

  sdata[idx] = 0.0f;

  // Process elements
  for (int index = start_index + idx; index < end_index; index += blockDim.x) {
    if (index < total_elements) {
      val = expf(A[index] - max_val);
      A[index] = val;
      atomicAdd(&sdata[idx], val);
    }
  }

  __syncthreads();

  // Sum reduction in shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (idx < s) {
      sdata[idx] += sdata[idx + s];
    }
    __syncthreads();
  }

  // Normalize the values
  for (int index = start_index + idx; index < end_index; index += blockDim.x) {
    if (index < total_elements) {
      A[index] /= sdata[0];
    }
  }
}



int main(){

  float *h_A, *d_A;
  h_A = new float[DSIZE*DSIZE];  

  for (int i = 0; i < DSIZE*DSIZE; i++) h_A[i] = element_val;

  cudaMalloc(&d_A, DSIZE*DSIZE*sizeof(float)); 
  cudaCheckErrors("cudaMalloc failure");

  cudaMemcpy(d_A, h_A, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");

  softmax_max<<<DSIZE, block_size>>>(d_A, DSIZE);
  cudaCheckErrors("kernel launch failure");

  cudaMemcpy(h_A, d_A, DSIZE*DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckErrors("cudaMemcpy D2H failure");

  for(int i = 0; i < DSIZE*DSIZE; i++){
    printf("h_A[%d]: %.8f\n", i, h_A[i]);
    if(abs(h_A[i] - 1/(float)DSIZE) > 0.00001
    ) {printf("results mismatch at %d, was: %.10f, should be: %.10f\n", i, h_A[i], 1/float(DSIZE)); return -1;}
  }
    printf("softmax correct!\n");
    
  return 0;
}
  
