#include <stdio.h>
#include "kernels.h"

const int block_size = 32;  // CUDA maximum is 1024 *total* threads in block  
const int block_size_vocab = 1024;

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

__global__ void layernorm(float *A, float *B, int dim, float *gamma, float *beta){

    int idx = threadIdx.x;

    __shared__ float sdata[block_size];
    __shared__ float sum;
    sdata[idx] = 0.0f;
    sum = 0.0f;
    float val = 0.0f;

    for(int i = 0; i < dim/blockDim.x; i++) sdata[idx] += A[dim*blockIdx.x + i*blockDim.x + idx];

    for(int s = blockDim.x/2; s > 0; s/=2){
    __syncthreads();
    if (idx < s) sdata[idx] += sdata[idx + s];
    }

    if(idx == 0) sum = sdata[0]/dim;

    __syncthreads();

    sdata[idx] = 0.0f;

    for(int i = 0; i < dim/blockDim.x; i++){
        val = (A[dim*blockIdx.x + i*blockDim.x + idx] - sum);
        sdata[idx] += val*val;
    }

    for(int s = blockDim.x/2; s > 0; s/=2){
        __syncthreads();
        if (idx < s) sdata[idx] += sdata[idx + s];
    }

    if (idx == 0) sdata[0] = 1/sqrt(sdata[0]/dim + 0.00001);

    __syncthreads();

    for(int i = 0; i < dim/blockDim.x; i++){
        B[dim*blockIdx.x + i*blockDim.x + idx] = (A[dim*blockIdx.x + i*blockDim.x + idx] - sum)*sdata[0]*gamma[i*blockDim.x + idx] + beta[i*blockDim.x + idx];
    }

}

__global__ void matmul(const float *A, const float *B, float *C, int height, int width, int dim) {

  // declare cache in shared memory
  __shared__ float As[block_size][block_size];
  __shared__ float Bs[block_size][block_size];

  int col = threadIdx.x+blockDim.x*blockIdx.x;
  int row = threadIdx.y+blockDim.y*blockIdx.y;

  if ((row < height) && (col < width)){
    float temp = 0;
    int iter = dim/block_size;
    for (int i = 0; i < iter; i++) {

      // Load data into shared memory
      As[threadIdx.y][threadIdx.x] = A[row*dim + (block_size*i + threadIdx.x)];
      Bs[threadIdx.y][threadIdx.x] = B[col + width*(block_size*i + threadIdx.y)];

      __syncthreads();

      // Keep track of the running sum
      // for (int k = 0; k < block_size; k++)

      for (int k = 0; k < block_size; k++)
        temp += As[threadIdx.y][k] * Bs[k][threadIdx.x]; // dot product of row and column

      __syncthreads();

    }

    C[row*width+col] = temp;
  }
}

__global__ void matmul_bias(const float *A, const float *B, float *C, float *bias, int height, int width, int dim, int N_tokens) {

  // declare cache in shared memory
  __shared__ float As[block_size][block_size];
  __shared__ float Bs[block_size][block_size];

  int row = threadIdx.y+blockDim.y*blockIdx.y;
  int col = threadIdx.x+blockDim.x*blockIdx.x;
  

  if ((row < height) && (col < width)){
    float temp = 0;
    for (int i = 0; i < dim/block_size; i++) {

      // Load data into shared memory
      As[threadIdx.y][threadIdx.x] = A[row*dim + (block_size*i + threadIdx.x)];
      Bs[threadIdx.y][threadIdx.x] = B[col + width*(block_size*i + threadIdx.y)];

      __syncthreads();

      // Keep track of the running sum
      for (int k = 0; k < block_size; k++)
      	temp += As[threadIdx.y][k] * Bs[k][threadIdx.x]; // dot product of row and column
      __syncthreads();
    }
  

    if(row < N_tokens) C[row*width+col] = temp + bias[col];
    else C[row*width+col] = temp;
  }
}


__global__ void QK_V(const float *QK, const float *V, float *C, int Dim, int N) {

  // declare cache in shared memory
  __shared__ float As[block_size][block_size];
  __shared__ float Bs[block_size][block_size];
  
  int col = threadIdx.x+blockDim.x*blockIdx.x; // create thread x index
  int row = threadIdx.y+blockDim.y*blockIdx.y; // create thread y index

  if ((row < N) && (col < Dim)){
    float temp = 0, val, sum = 0;

    for (int i = 0; i < N/block_size; i++) {

      // Load data into shared memory
      As[threadIdx.y][threadIdx.x] = expf(QK[row*N + (block_size*i + threadIdx.x)]);
      Bs[threadIdx.y][threadIdx.x] = V[col + Dim*(block_size*i + threadIdx.y)];

      __syncthreads();

      for (int k = 0; k < block_size; k++){
        val = As[threadIdx.y][k];
      	temp +=  val * Bs[k][threadIdx.x]; // dot product of row and column
        sum+=val;
      }

      __syncthreads();

    }

    C[row*Dim+col] = temp/sum;
  }
}

__global__ void gelu(float *A, int dim){

    int idx = threadIdx.x;
    float x;

    for(int i = 0; i < dim/blockDim.x; i++){
        x = A[dim*blockIdx.x + i*blockDim.x + idx];
        A[dim*blockIdx.x + i*blockDim.x + idx] = x*0.5*(1.0 + tanhf(0.7978845608*(x + 0.044715*x*x*x)));
    }
}

__global__ void add(float *A, float *B, int dim){

    int idx = threadIdx.x;

    for(int i = 0; i < dim/blockDim.x; i++){
        B[dim*blockIdx.x + i*blockDim.x + idx] = A[dim*blockIdx.x + i*blockDim.x + idx] + B[dim*blockIdx.x + i*blockDim.x + idx];
    }

}

// scale<<<N, block_size>>>(d_QK, head_dim, N, head_dim);

__global__ void scale(float *A, int N, int head_dim){

    int idx = threadIdx.x;

    for(int i = 0; i < N/blockDim.x; i++){
        A[N*blockIdx.x + i*blockDim.x + idx] = A[N*blockIdx.x + i*blockDim.x + idx]/sqrtf(head_dim);
    }

}

// set traingle values  and values outside N_tokens*N_tokens to -infinity
__global__ void set_inf(float *A, int dim, int N, int N_tokens){

    int idx = threadIdx.x;
    int NEG_INF = -50;
    if(blockIdx.x < N_tokens){
      for(int i = 0; i < dim/blockDim.x; i++){
          if (i*blockDim.x + idx < N_tokens &&  i*blockDim.x + idx < blockIdx.x+1) continue;
          A[dim*blockIdx.x + i*blockDim.x + idx] = NEG_INF;
        }
    }
    else{
      for(int i = 0; i < dim/blockDim.x; i++) A[dim*blockIdx.x + i*blockDim.x + idx] = NEG_INF;
    }
}

// set all values to -infinity except for N_tokens*N_tokens block
__global__ void set_zero(float *A, int dim, int N, int N_tokens){

    int idx = threadIdx.x;

    if(blockIdx.x < N_tokens){
      for(int i = 0; i < dim/blockDim.x; i++){
          if (i*blockDim.x + idx < N_tokens) continue;
          A[dim*blockIdx.x + i*blockDim.x + idx] = 0;
        }
    }
    else{
      for(int i = 0; i < dim/blockDim.x; i++) A[dim*blockIdx.x + i*blockDim.x + idx] = 0;
    }
}

__global__ void isnan_test(float *data, int width, int height){

  int idx = threadIdx.x+blockDim.x*blockIdx.x;

  while (idx < width){
    for (int i = 0; i < height; i++){
      if (isnan(data[(i*width) + idx]) || isinf(data[(i*width) + idx])){
        printf("NAN or INF at %d, %d\n", i, idx);
        return;
      }
    }
    idx += gridDim.x+blockDim.x;
    }
}

__global__ void matmul_mha_transpose(const float *A, const float *B, float *C, int height, int width, int dim, int head_dim, int head_num) {

  // declare cache in shared memory
  __shared__ float As[block_size][block_size];
  __shared__ float Bs[block_size][block_size];

  // if(threadIdx.x == 0 && threadIdx.y == 0) printf("head_num %d %d\n", blockIdx.x, blockIdx.y);

  int col = threadIdx.x+blockDim.x*blockIdx.x;
  int row = threadIdx.y+blockDim.y*blockIdx.y;
  // printf("%d %d\n", row, col);
  

  if ((row < height) && (col < width)){
    // if(row == 8 && col == 8) printf("final %d %d\n", row, col);
  
    float temp = 0;
    for (int i = 0; i < head_dim/block_size; i++) {

      // Load data into shared memory
      As[threadIdx.y][threadIdx.x] = A[row*dim + (block_size*i + threadIdx.x + head_dim*head_num)];
      Bs[threadIdx.y][threadIdx.x] = B[col*dim + (block_size*i + threadIdx.y + head_dim*head_num)];

      __syncthreads();

      // Keep track of the running sum
      for (int k = 0; k < block_size; k++)
      	temp += As[threadIdx.y][k] * Bs[k][threadIdx.x]; // dot product of row and column
    
      __syncthreads();

    }
    
    C[row*width+col] = temp;
  }
}


__global__ void matmul_mha(const float *A, const float *B, float *C, int height, int width, int dim, int head_dim, int head_num, int width_full ) {

// N, head_dim, N, head_dim, i, Dim

  // declare cache in shared memory
  __shared__ float As[block_size][block_size];
  __shared__ float Bs[block_size][block_size];

  // As[threadIdx.y][threadIdx.x] = 0.0f;
  // Bs[threadIdx.y][threadIdx.x] = 0.0f;

  int row = threadIdx.y+blockDim.y*blockIdx.y;
  int col = threadIdx.x+blockDim.x*blockIdx.x + head_dim*head_num;
  // if(threadIdx.x == 0 && threadIdx.y == 0) printf("headnum, %d %d\n", row, head_dim*head_num);
  
  if ((row < height) && (col < width_full)){

    float temp = 0;
    for (int i = 0; i < dim/block_size; i++) {

      // Load data into shared memory
      As[threadIdx.y][threadIdx.x] = A[row*dim + (block_size*i + threadIdx.x)];
      Bs[threadIdx.y][threadIdx.x] = B[col + width_full*(block_size*i + threadIdx.y)];

      __syncthreads();

      // Keep track of the running sum
      for (int k = 0; k < block_size; k++)
      	temp += As[threadIdx.y][k] * Bs[k][threadIdx.x]; // dot product of row and column
    
      __syncthreads();

    }
    // if(
    //   threadIdx.x == 0 && threadIdx.y == 0 )printf("final %d %d %d\n", row*width_full, width_full,  col);
    C[row*width_full+col] = temp;
  }
}

__global__ void max_index(float *A, size_t height, size_t width, int *max_idx) {

  int idx = threadIdx.x;
  __shared__ int sdata[block_size_vocab];

  sdata[idx] = 0;

  int start_index = blockIdx.x * width; // Start index for this block
  int end_index = start_index + width;  // End index for this block

  // Find the maximum value in the block

  for (int index = start_index + idx; index < end_index; index += blockDim.x) {
    // if(threadIdx.x == 0) printf("%d \n", index);
    if (index < height*width && A[index] > A[sdata[idx]]) sdata[idx] = index;
    
  }

  for(int s = blockDim.x/2; s > 0; s/=2){
    __syncthreads();
    if (idx < s &&  A[sdata[idx + s]] > A[sdata[idx]]) sdata[idx] = sdata[idx + s];
  }
  __syncthreads();

  if (idx == 0) *max_idx = sdata[0];
 
}

// set all values to -infinity except for N_tokens*N_tokens block
__global__ void set_new_embedding(float *A, int dim, int N_tokens, float *emb, int *new_token, float *pos){

    int idx = threadIdx.x;
    for(int i = 0; i < dim/blockDim.x; i++) 
    A[dim*N_tokens + i*blockDim.x + idx] = emb[new_token[0]*dim + i*blockDim.x + idx] + pos[N_tokens*dim + i*blockDim.x + idx];
    
}