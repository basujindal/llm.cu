#include <stdio.h>
using namespace std;

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


// const int N = 2048;      // matrix side dimension
const int Dim = 768;     // should be greater than 32
const int block_size = 32;  // CUDA maximum is 1024 *total* threads in block  
const int block_size_linear = 768;
const int block_size_vocab = 1024;

const int N_Layers = 12;
const int Vocab_OG = 50257;
const int Vocab = 50272;
const int num_heads = 12;
int num_new_tokens = 35;


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

    // Write to global memory
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

int MHA(float *d_input, float *d_Q, float *d_K, float *d_V, float *d_QK, float *d_act, float *d_act_wide,
      float *linear[4], float *bias[4], float *ln[], float *mlp1, float *mlp_bias1, float *mlp2, float *mlp_bias2,
      int Dim, int N, int N_tokens, float *h_output, float *h_test, float *d_act2){

    dim3 threads(block_size, block_size);
    dim3 grid((Dim + threads.y - 1)/block_size, (N + threads.x - 1)/block_size);


    // printf("Layer Normalization\n");
    layernorm<<<N_tokens, block_size>>>(d_input, d_act, Dim, ln[0], ln[1]);
    cudaCheckErrors("kernel launch failure");
    // isnan_test<<<1, 1>>>(d_act, Dim, N);
    cudaDeviceSynchronize();

    // calculate Q,K,V
    // printf("Q\n");
    matmul_bias<<<grid, threads>>>(d_act, linear[0], d_Q, bias[0], N, Dim, Dim, N_tokens);
    cudaCheckErrors("kernel launch failure");
    // isnan_test<<<1, 1>>>(d_Q, Dim, N);
    cudaDeviceSynchronize();


    // printf("K\n");
    matmul_bias<<<grid, threads>>>(d_act, linear[1], d_K, bias[1], N, Dim, Dim, N_tokens);
    cudaCheckErrors("kernel launch failure");
    // isnan_test<<<1, 1>>>(d_K, Dim, N);
    cudaDeviceSynchronize();    

    // printf("V\n");
    matmul_bias<<<grid, threads>>>(d_act, linear[2], d_V, bias[2], N, Dim, Dim, N_tokens);
    cudaCheckErrors("kernel launch failure");
    // isnan_test<<<1, 1>>>(d_V, Dim, N);
    cudaDeviceSynchronize();

    int head_dim = Dim/num_heads;
    dim3 grid_mha((Dim + threads.y - 1)/block_size, (N + threads.x - 1)/block_size);

    dim3 grid_mha_transpose((N + threads.y - 1)/block_size, (N + threads.x - 1)/block_size);

    for (int i = 0; i < num_heads; i++){
  
      // printf("QK\n"); 
      matmul_mha_transpose<<<grid_mha_transpose, threads>>>(d_Q, d_K, d_QK, N, N, Dim, head_dim, i);
      cudaCheckErrors("kernel launch failure");
      // isnan_test<<<1, 1>>>(d_QK, N, N);
      cudaDeviceSynchronize();
      
      // scale by sqrt(d_k)
      // printf("Scale\n");
      scale<<<N, block_size>>>(d_QK, N, head_dim);
      cudaCheckErrors("kernel launch failure");
      // isnan_test<<<1, 1>>>(d_QK, N, N);
      cudaDeviceSynchronize();

      // Set non tokens to -infinity
      // printf("Set non tokens to -infinity\n");
      set_inf<<<N, block_size>>>(d_QK, N, N, N_tokens);
      cudaCheckErrors("kernel launch failure");
      // isnan_test<<<1, 1>>>(d_QK, N, N);
      cudaDeviceSynchronize();

      // Softmax
      // printf("Softmax\n");
      softmax_max<<<N, block_size>>>(d_QK, N);
      cudaCheckErrors("kernel launch failure");
      // isnan_test<<<1, 1>>>(d_QK, N, N);
      cudaDeviceSynchronize();

      // Set non tokens to -infinity
      // printf("Set non tokens to -infinity\n");
      set_zero<<<N, block_size>>>(d_QK, N, N, N_tokens);
      cudaCheckErrors("kernel launch failure");
      // isnan_test<<<1, 1>>>(d_QK, N, N);
      cudaDeviceSynchronize();

      // printf("QK_V\n");
      matmul_mha<<<grid_mha, threads>>>(d_QK, d_V, d_act, N, head_dim, N, head_dim, i, Dim);
      cudaCheckErrors("kernel launch failure");
      // isnan_test<<<1, 1>>>(d_act, head_dim, N);
      cudaDeviceSynchronize();

    }
    cudaDeviceSynchronize();

    // printf("Final output\n");
    matmul_bias<<<grid, threads>>>(d_act, linear[3], d_act2, bias[3], N, Dim, Dim, N_tokens);
    cudaCheckErrors("kernel launch failure");
    // isnan_test<<<1, 1>>>(d_act, Dim, N);
    cudaDeviceSynchronize();


    // Residual connection
    // printf("Residual connection\n");
    add<<<N, block_size_linear>>>(d_act2, d_input, Dim);
    cudaCheckErrors("kernel launch failure");
    // isnan_test<<<1, 1>>>(d_input, Dim, N);
    cudaDeviceSynchronize();

    // Layer Normalization
    // printf("Layer Normalization\n");
    layernorm<<<N_tokens, block_size>>>(d_input, d_act, Dim, ln[2], ln[3]);
    cudaCheckErrors("kernel launch failure");
    // isnan_test<<<1, 1>>>(d_input, Dim, N);
    cudaDeviceSynchronize();

    dim3 grid_wide((4*Dim + threads.x - 1)/block_size, (N + threads.y - 1)/block_size);
    // Matmul
    // printf("Mlp1\n");
    matmul_bias<<<grid_wide, threads>>>(d_act, mlp1, d_act_wide, mlp_bias1, N, 4*Dim, Dim, N_tokens);
    cudaCheckErrors("kernel launch failure");
    // isnan_test<<<1, 1>>>(d_act_wide, Dim, N);
    cudaDeviceSynchronize();


    //gelu
    // printf("Gelu\n");
    gelu<<<N, block_size>>>(d_act_wide, 4*Dim);
    cudaCheckErrors("kernel launch failure");
    // isnan_test<<<1, 1>>>(d_act_wide, 4*Dim, N);
    cudaDeviceSynchronize();
    // cudaCheckErrors("kernel launch failure");

    // dim3 grid2((Dim + threads.y - 1)/block_size, (N + threads.x - 1)/block_size);

    // printf("mlp2\n");
    matmul_bias<<<grid, threads>>>(d_act_wide, mlp2, d_act, mlp_bias2, N, Dim, 4*Dim, N_tokens);
    cudaCheckErrors("kernel launch failure");
    // isnan_test<<<1, 1>>>(d_act, Dim, N);
    cudaDeviceSynchronize();

    // cudaMemcpy(h_test, d_act, N*Dim*sizeof(float), cudaMemcpyDeviceToHost);
    // cudaCheckErrors("cudaMemcpy D2H failure");
    // for(int k = 0; k < N_tokens; k++){
    //   printf("token %d ", k);
    //   for(int j = 0; j < 10; j++)printf("%.5f ", h_test[k*Dim + j]);
    //   printf("\n");
    // }
    // printf("\n\n");

    // Residual connection
    // printf("Residual connection\n");
    add<<<N, block_size_linear>>>(d_act, d_input, Dim);
    cudaCheckErrors("kernel launch failure");
    // isnan_test<<<1, 1>>>(d_input, Dim, N);
    cudaDeviceSynchronize();

    return 0;
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

int Transformer(float *d_input, float *d_Q, float *d_K, float *d_V, float *d_QK, float *d_act, float *d_act_wide,
      float *linear[N_Layers][4], float *bias[N_Layers][4], float *ln[N_Layers][4], float *mlp1[N_Layers],
      float *mlp_bias1[N_Layers], float *mlp2[N_Layers], float *mlp_bias2[N_Layers], float *ln_final[2],
      float *proj_linear, float *d_output, int Dim, int N, int N_tokens, float *h_output, 
      float *h_test, float *d_act2, int *d_max_idx, int *new_token){


      for(int i = 0; i < 12; i++){
        // printf("Layer %d\n", i);

        MHA(d_input, d_Q, d_K, d_V, d_QK, d_act, d_act_wide,
        linear[i], bias[i], ln[i], mlp1[i], mlp_bias1[i], mlp2[i], mlp_bias2[i],
         Dim, N, N_tokens, h_output, h_test, d_act2);
        cudaDeviceSynchronize();
        // printf("Layer %d done\n", i);
      }
      cudaDeviceSynchronize();

      // Layer Normalization
      layernorm<<<N_tokens, block_size>>>(d_input, d_input, Dim, ln_final[0], ln_final[1]);
      cudaCheckErrors("kernel launch failure");

      dim3 threads(block_size, block_size);
      dim3 grid((Vocab + block_size - 1)/block_size, (Dim + block_size - 1)/block_size);

      // Matmul
      matmul<<<grid, threads>>>(d_input, proj_linear, d_output, N, Vocab, Dim);
      cudaCheckErrors("kernel launch failure");

      max_index<<<1, block_size_vocab>>>(d_output + Vocab*(N_tokens-1), 1, Vocab_OG, d_max_idx);
      cudaCheckErrors("kernel launch failure");
      cudaDeviceSynchronize();

      return 0;

      }

int read_weight(float *arr, char *filename, int rows, int cols){

  // printf("Reading %s\n", filename);
  
  FILE *file = fopen(filename, "rb");
  if (file == NULL) {
      printf("Error opening file\n");
      return 1;
  }

  fread(arr, sizeof(float), rows * cols, file);
  fclose(file);

  return 0;
  
}


int main(){

    int N = 512;
    int N_tokens = 9;
    int text[N_tokens] = {15496,    11,   616,  1438,   318,  1757,    13,   314,  1101}; // 257
    // int N_tokens = 8;
    // int text[] = {15496,    11,   616,  1438,   318,  1757,    13,   314};

    // declare host memory pointers

    float *h_input, *h_output, *h_linear[N_Layers][4], *h_bias[N_Layers][4], *h_ln[N_Layers][4],
           *h_mlp1[N_Layers], *h_mlp_bias1[N_Layers], *h_mlp2[N_Layers], *h_mlp_bias2[N_Layers],
           *h_final_ln[2], *h_proj_linear, *h_ans, *h_pos, *h_emb, *h_input2;


    float *d_input, *d_output, *d_Q, *d_K, *d_QK, *d_V, *d_ACT, *d_ACT_wide,
        *d_linear[N_Layers][4], *d_bias[N_Layers][4], *d_ln[N_Layers][4], *d_mlp1[N_Layers], 
        *d_mlp_bias1[N_Layers], *d_mlp2[N_Layers], *d_mlp_bias2[N_Layers], *d_final_ln[2], 
        *d_proj_linear, *d_act2, *d_emb, *d_pos;
        int *d_max_idx;


    // allocate space for data in host memory

    h_input = new float[N*Dim];
    h_output = new float[N*Vocab];
    h_ans = new float[N*Vocab];
    h_pos = new float[N*Dim];
    h_emb = new float[Dim*Vocab_OG];
    h_input2 = new float[N*Dim];

    float* h_test = new float[4*N*Dim];

    for (int i = 0; i < N_Layers; i++){

      for (int j = 0; j < 4; j++){
        h_linear[i][j] = new float[Dim*Dim];
        h_bias[i][j] = new float[Dim];
        h_ln[i][j] = new float[Dim];
      }

      h_mlp1[i] = new float[Dim*4*Dim];
      h_mlp_bias1[i] = new float[4*Dim];
      h_mlp2[i] = new float[Dim*4*Dim];
      h_mlp_bias2[i] = new float[Dim];
    }

    for (int i = 0; i < 2; i++) h_final_ln[i] = new float[Dim];
    h_proj_linear = new float[Dim*Vocab];
    float *h_proj_linear_og = new float[Dim*Vocab_OG];

    for (int i = 0; i < Dim*Vocab; i++) h_proj_linear[i] = 0;

    for (int i = 0; i < N*Dim; i++) h_input[i] = 0;

    // initialize matrix in host memory
    char filename[256];

    // snprintf(filename, sizeof(filename), "gpt_weights/input.bin");
    // read_weight(h_input2, filename, N_tokens, Dim);

    snprintf(filename, sizeof(filename), "gpt_weights/wpe.weight.bin");
    read_weight(h_pos, filename, N, Dim);

    snprintf(filename, sizeof(filename), "gpt_weights/wte.weight.bin");
    read_weight(h_emb, filename, Vocab_OG, Dim);

    for (int i = 0; i < N_tokens; i++){
      for (int j = 0; j < Dim; j++){
        h_input[i*Dim + j] = h_emb[text[i]*Dim + j] + h_pos[i*Dim + j];
      }
    }

    for (int i = 0; i < N_Layers; i++){

      for(int j = 0; j < 2; j++){
        snprintf(filename, sizeof(filename), "gpt_weights/h.%d.ln_%d.weight.bin", i, j+1);
        read_weight(h_ln[i][j*2], filename, Dim, 1);
        snprintf(filename, sizeof(filename), "gpt_weights/h.%d.ln_%d.bias.bin", i, j+1);
        read_weight(h_ln[i][j*2+1], filename, Dim, 1);
      }
 
      snprintf(filename, sizeof(filename), "gpt_weights/h.%d.attn.c_attn.weight.q.bin", i);
      read_weight(h_linear[i][0], filename, Dim, Dim);

      snprintf(filename, sizeof(filename), "gpt_weights/h.%d.attn.c_attn.weight.k.bin", i);
      read_weight(h_linear[i][1], filename, Dim, Dim);

      snprintf(filename, sizeof(filename), "gpt_weights/h.%d.attn.c_attn.weight.v.bin", i);
      read_weight(h_linear[i][2], filename, Dim, Dim);

      snprintf(filename, sizeof(filename), "gpt_weights/h.%d.attn.c_proj.weight.bin", i);
      read_weight(h_linear[i][3], filename, Dim, Dim);

      snprintf(filename, sizeof(filename), "gpt_weights/h.%d.attn.c_attn.bias.q.bin", i);
      read_weight(h_bias[i][0], filename, Dim, 1);
      
      snprintf(filename, sizeof(filename), "gpt_weights/h.%d.attn.c_attn.bias.k.bin", i);
      read_weight(h_bias[i][1], filename, Dim, 1);

      snprintf(filename, sizeof(filename), "gpt_weights/h.%d.attn.c_attn.bias.v.bin", i);
      read_weight(h_bias[i][2], filename, Dim, 1);

      snprintf(filename, sizeof(filename), "gpt_weights/h.%d.attn.c_proj.bias.bin", i);
      read_weight(h_bias[i][3], filename, Dim, 1);

      snprintf(filename, sizeof(filename), "gpt_weights/h.%d.mlp.c_fc.weight.bin", i);
      read_weight(h_mlp1[i], filename, Dim*4*Dim, 1);

      snprintf(filename, sizeof(filename), "gpt_weights/h.%d.mlp.c_fc.bias.bin", i);
      read_weight(h_mlp_bias1[i], filename, 4*Dim, 1);

      snprintf(filename, sizeof(filename), "gpt_weights/h.%d.mlp.c_proj.weight.bin", i);
      read_weight(h_mlp2[i], filename, Dim*4*Dim, 1);

      snprintf(filename, sizeof(filename), "gpt_weights/h.%d.mlp.c_proj.bias.bin", i);
      read_weight(h_mlp_bias2[i], filename, Dim, 1);
    }


    snprintf(filename, sizeof(filename), "gpt_weights/ln_f.weight.bin");
    read_weight(h_final_ln[0], filename, Dim, 1);

    snprintf(filename, sizeof(filename), "gpt_weights/ln_f.bias.bin");
    read_weight(h_final_ln[1], filename, Dim, 1);

    snprintf(filename, sizeof(filename), "gpt_weights/etw.weight.bin");
    read_weight(h_proj_linear_og, filename, Dim, Vocab_OG);

    // copy from h_proj_linear_og to h_proj_linear
    for (int i = 0; i < Dim; i++){
      for (int j = 0; j < Vocab_OG; j++){
        h_proj_linear[i*Vocab + j] = h_proj_linear_og[i*Vocab_OG + j];
      }
    }

    snprintf(filename, sizeof(filename), "gpt_weights/output.bin");
    read_weight(h_ans, filename, N_tokens, Vocab_OG);

    // allocate device space

    cudaMalloc(&d_input, N*Dim*sizeof(float));
    cudaMalloc(&d_output, N*Vocab*sizeof(float));
    cudaMalloc(&d_Q, N*Dim*sizeof(float));
    cudaMalloc(&d_K, N*Dim*sizeof(float));
    cudaMalloc(&d_V, N*Dim*sizeof(float));  
    cudaMalloc(&d_QK, N*N*sizeof(float));
    cudaMalloc(&d_ACT, N*Dim*sizeof(float));
    cudaMalloc(&d_ACT_wide, 4*N*Dim*sizeof(float));
    cudaMalloc(&d_act2, N*Dim*sizeof(float));
    cudaMalloc(&d_emb, Dim*Vocab_OG*sizeof(float));
    cudaMalloc(&d_pos, N*Dim*sizeof(float));

    for (int i = 0; i < N_Layers; i++){

      for (int j = 0; j < 4; j++){
        cudaMalloc(&d_linear[i][j], Dim*Dim*sizeof(float));
        cudaMalloc(&d_bias[i][j], Dim*sizeof(float));
        cudaMalloc(&d_ln[i][j], Dim*sizeof(float));
        cudaCheckErrors("cudaMalloc failure"); // error checking
      }

      cudaMalloc(&d_mlp1[i], Dim*4*Dim*sizeof(float));
      cudaMalloc(&d_mlp_bias1[i], 4*Dim*sizeof(float));
      cudaMalloc(&d_mlp2[i], Dim*4*Dim*sizeof(float));
      cudaMalloc(&d_mlp_bias2[i], Dim*sizeof(float));
      cudaCheckErrors("cudaMalloc failure"); // error checking
    }

    for (int i = 0; i < 2; i++) cudaMalloc(&d_final_ln[i], Dim*sizeof(float));

    cudaMalloc(&d_proj_linear, Dim*Vocab*sizeof(float));
    cudaMalloc(&d_max_idx, sizeof(int));

    cudaDeviceSynchronize();

    // copy data to device
    cudaMemcpy(d_input, h_input, N*Dim*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    for (int i = 0; i < N_Layers; i++){

      for (int j = 0; j < 4; j++){
        cudaMemcpy(d_bias[i][j], h_bias[i][j], Dim*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_linear[i][j], h_linear[i][j], Dim*Dim*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ln[i][j], h_ln[i][j], Dim*sizeof(float), cudaMemcpyHostToDevice);
        cudaCheckErrors("cudaMemcpy H2D failure");
      }

      cudaMemcpy(d_mlp1[i], h_mlp1[i], Dim*4*Dim*sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_mlp_bias1[i], h_mlp_bias1[i], 4*Dim*sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_mlp2[i], h_mlp2[i], Dim*4*Dim*sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_mlp_bias2[i], h_mlp_bias2[i], Dim*sizeof(float), cudaMemcpyHostToDevice);
      cudaCheckErrors("cudaMemcpy H2D failure");
    }
    
    for (int i = 0; i < 2; i++){
      cudaMemcpy(d_final_ln[i], h_final_ln[i], Dim*sizeof(float), cudaMemcpyHostToDevice);
      cudaCheckErrors("cudaMemcpy H2D failure");
    }

    cudaMemcpy(d_proj_linear, h_proj_linear, Dim*Vocab*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    cudaMemcpy(d_emb, h_emb, Dim*Vocab_OG*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    cudaMemcpy(d_pos, h_pos, N*Dim*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    cudaDeviceSynchronize();

    int *new_token = new int;
    *new_token = 0;


    for (int z = 0; z < num_new_tokens; z++){


      // Launch kernel
      Transformer(d_input, d_Q, d_K, d_V, d_QK, d_ACT, d_ACT_wide, d_linear, d_bias,
      d_ln, d_mlp1, d_mlp_bias1, d_mlp2, d_mlp_bias2, d_final_ln, d_proj_linear, d_output, 
      Dim, N, N_tokens,h_output, h_test, d_act2, d_max_idx, new_token );
      cudaCheckErrors("kernel launch failure");

      // synchronize device
      cudaDeviceSynchronize();

      // // Copy results back to host
      cudaMemcpy(h_output, d_output, N*Vocab*sizeof(float), cudaMemcpyDeviceToHost);
      cudaCheckErrors("cudaMemcpy D2H failure");

      cudaMemcpy(new_token, d_max_idx, sizeof(int), cudaMemcpyDeviceToHost);
      cudaCheckErrors("cudaMemcpy D2H failure");
      cudaDeviceSynchronize();


    //   // find max value index
    //   float max = -INFINITY;
    //   int nnew_token = 0;
    //   for (int j = 0; j < Vocab_OG; j++){
    //     if (h_output[(N_tokens-1)*Vocab + j] > max){
    //       max = h_output[(N_tokens-1)*Vocab + j];
    //       nnew_token = j;
    //     }
    //   }
    printf("new token idx = %d\n", *new_token);
    cudaDeviceSynchronize();

    // add the new token to the input
    for (int j = 0; j < Dim; j++)
    h_input[N_tokens*Dim + j] = h_emb[new_token[0]*Dim + j] + h_pos[N_tokens*Dim + j];
    
    // copy new input to device
    cudaMemcpy(d_input, h_input, N*Dim*sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    N_tokens++;

    }

  
    return 0;
}
  