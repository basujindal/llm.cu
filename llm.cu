#include <stdio.h>
#include <time.h>
#include "load_weights.h"
#include "kernels.h"
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


const int Dim = 768;     // should be greater than 32
const int block_size = 32;  // CUDA maximum is 1024 *total* threads in block  
const int block_size_linear = 768;
const int block_size_vocab = 1024;
const int Vocab_OG = 50257;
const int Vocab = 50272;
const int num_heads = 12;
const int N_Layers = 12;

int MHA(float *d_input, float *d_Q, float *d_K, float *d_V, float *d_QK, float *d_act, float *d_act_wide,
      float *linear[4], float *bias[4], float *ln[], float *mlp1, float *mlp_bias1, float *mlp2, float *mlp_bias2,
      int Dim, int N, int N_tokens, float *h_output, float *d_act2, int N_compute){

    dim3 threads(block_size, block_size);
    dim3 grid((Dim + threads.y - 1)/block_size, (N_compute + threads.x - 1)/block_size);

    // Layer Normalization;
    layernorm<<<N_tokens, block_size>>>(d_input, d_act, Dim, ln[0], ln[1]);
    cudaCheckErrors("kernel launch failure");
    // isnan_test<<<1, 1>>>(d_act, Dim, N);

    // Q;
    matmul_bias<<<grid, threads>>>(d_act, linear[0], d_Q, bias[0], N_compute, Dim, Dim, N_tokens);
    cudaCheckErrors("kernel launch failure");
    // isnan_test<<<1, 1>>>(d_Q, Dim, N);

    // K;
    matmul_bias<<<grid, threads>>>(d_act, linear[1], d_K, bias[1], N_compute, Dim, Dim, N_tokens);
    cudaCheckErrors("kernel launch failure");
    // isnan_test<<<1, 1>>>(d_K, Dim, N);   

    // V;
    matmul_bias<<<grid, threads>>>(d_act, linear[2], d_V, bias[2], N_compute, Dim, Dim, N_tokens);
    cudaCheckErrors("kernel launch failure");
    // isnan_test<<<1, 1>>>(d_V, Dim, N);

    // // Start MHA

    int head_dim = Dim/num_heads;
    float scale_factor = sqrtf(head_dim);

    dim3 grid_mha_transpose((N_compute + threads.y - 1)/block_size, (N_compute + threads.x - 1)/block_size, num_heads);
    dim3 grid_softmax_mha(N_tokens, num_heads, 1);
    dim3 grid_mha((Dim + threads.y - 1)/block_size, (N_compute + threads.x - 1)/block_size, num_heads);
    dim3 grid_wide((4*Dim + threads.x - 1)/block_size, (N_compute + threads.y - 1)/block_size);
    dim3 threads_mha(block_size, block_size, 1);

    // QK; 
    matmul_mha_transpose_scale<<<grid_mha_transpose, threads_mha>>>(scale_factor, d_Q, d_K, d_QK, N_compute, N_compute, Dim, head_dim, N);
    cudaCheckErrors("kernel launch failure");

    // Set upper triangle to -inf
    set_inf_mha<<<grid_softmax_mha, block_size>>>(d_QK, N_compute, N, N_tokens);
    cudaCheckErrors("kernel launch failure");

    // Softmax
    softmax_max_mha<<<grid_softmax_mha, block_size>>>(d_QK, N_compute, N);
    cudaCheckErrors("kernel launch failure");
    // isnan_test<<<1, 1>>>(d_QK, N, N);

    // QKV
    matmul_mha<<<grid_mha, threads_mha>>>(d_QK, d_V, d_act, N_compute, head_dim, N_compute, head_dim, Dim, N);
    cudaCheckErrors("kernel launch failure");
    // isnan_test<<<1, 1>>>(d_act, head_dim, N);
    cudaDeviceSynchronize();

    // Final output;
    matmul_bias<<<grid, threads>>>(d_act, linear[3], d_act2, bias[3], N_compute, Dim, Dim, N_tokens);
    cudaCheckErrors("kernel launch failure");
    // isnan_test<<<1, 1>>>(d_act, Dim, N);

    // // End MHA

    // Residual connection
    add<<<N, block_size_linear>>>(d_act2, d_input, Dim);
    cudaCheckErrors("kernel launch failure");
    // isnan_test<<<1, 1>>>(d_input, Dim, N);

    // Layer Normalization
    layernorm<<<N_tokens, block_size>>>(d_input, d_act, Dim, ln[2], ln[3]);
    cudaCheckErrors("kernel launch failure");
    // isnan_test<<<1, 1>>>(d_input, Dim, N);

    // mlp1
    matmul_bias<<<grid_wide, threads>>>(d_act, mlp1, d_act_wide, mlp_bias1, N_compute, 4*Dim, Dim, N_tokens);
    cudaCheckErrors("kernel launch failure");
    // isnan_test<<<1, 1>>>(d_act_wide, Dim, N);

    //gelu
    gelu<<<N, block_size>>>(d_act_wide, 4*Dim);
    cudaCheckErrors("kernel launch failure");
    // isnan_test<<<1, 1>>>(d_act_wide, 4*Dim, N);
    cudaDeviceSynchronize();

    // mlp2;
    matmul_bias<<<grid, threads>>>(d_act_wide, mlp2, d_act, mlp_bias2, N_compute, Dim, 4*Dim, N_tokens);
    cudaCheckErrors("kernel launch failure");
    // isnan_test<<<1, 1>>>(d_act, Dim, N);
    cudaDeviceSynchronize();

    // Residual connection
    add<<<N_compute, block_size_linear>>>(d_act, d_input, Dim);
    cudaCheckErrors("kernel launch failure");
    // isnan_test<<<1, 1>>>(d_input, Dim, N);
    cudaDeviceSynchronize();

    return 0;
    }
  

int Transformer(float *d_input, float *d_Q, float *d_K, float *d_V, float *d_QK, float *d_act, float *d_act_wide,
      float *linear[N_Layers][4], float *bias[N_Layers][4], float *ln[N_Layers][4], float *mlp1[N_Layers],
      float *mlp_bias1[N_Layers], float *mlp2[N_Layers], float *mlp_bias2[N_Layers], float *ln_final[2],
      float *proj_linear, float *d_output, int Dim, int N, int N_tokens, float *h_output, 
      float *d_act2, int *d_max_idx, int *new_token, float *d_emb, float *d_pos, float *d_input2,
      int N_compute){

      cudaMemcpy(d_input2, d_input, N_tokens*Dim*sizeof(float), cudaMemcpyDeviceToDevice);

      for(int i = 0; i < 12; i++){

        MHA(d_input, d_Q, d_K, d_V, d_QK, d_act, d_act_wide,
        linear[i], bias[i], ln[i], mlp1[i], mlp_bias1[i], mlp2[i], mlp_bias2[i],
         Dim, N, N_tokens, h_output, d_act2, N_compute);
        cudaDeviceSynchronize();
      }
      cudaDeviceSynchronize();

      // Layer Normalization
      layernorm<<<N_tokens, block_size>>>(d_input, d_input, Dim, ln_final[0], ln_final[1]);
      cudaCheckErrors("kernel launch failure");
      cudaDeviceSynchronize();

      dim3 threads(block_size, block_size);
      dim3 grid((Vocab + block_size - 1)/block_size, (Dim + block_size - 1)/block_size);

      // Matmul
      matmul<<<grid, threads>>>(d_input, proj_linear, d_output, N, Vocab, Dim);
      cudaCheckErrors("kernel launch failure");
      cudaDeviceSynchronize();

      max_index<<<1, block_size_vocab>>>(d_output + Vocab*(N_tokens-1), 1, Vocab_OG, d_max_idx);
      cudaCheckErrors("kernel launch failure");
      cudaDeviceSynchronize();

      set_new_embedding<<<1, block_size_linear>>>(d_input, Dim, N_tokens, d_emb, d_max_idx, d_pos);
      cudaCheckErrors("kernel launch failure");
      cudaDeviceSynchronize();

      cudaMemcpy(d_input, d_input2, N_tokens*Dim*sizeof(float), cudaMemcpyDeviceToDevice);

      return 0;

      }

// struct GPT2Weights{
  
//       float *h_input;
//       float *h_output;
//       float ***h_linear;
//       float ***h_bias;
//       float ***h_ln;
//       float **h_mlp1;
//       float **h_mlp_bias1;
//       float **h_mlp2;
//       float **h_mlp_bias2;
//       float *h_final_ln[2];
//       float *h_proj_linear;
//       float *h_ans;
//       float *h_pos; 
//       float  *h_emb;
// };

int main(){

    printf("Running \n");
    clock_t t0, t1;
    double t1sum=0.0;
    t0 = clock();
    int N = 2048;

    // Test generation
    int N_tokens =  59;
    int num_new_tokens = 10;
    // Expected generation 1101, 257, 6260, 13, 314, 1101, 257, 6260, 13, 314
    printf("Expected generation \n 1101, 257, 6260, 13, 314, 1101, 257, 6260, 13, 314 \n");
    int text[N_tokens] = {15496,    11,   616,  1438,   318,  1757,    13,   314,  1101, 257, 6260, 11,
                          290, 314, 1101, 257, 6260, 13, 314, 1101, 257, 6260, 13, 314, 1101, 257, 6260,
                          13, 314, 1101, 257, 6260, 13, 314, 1101, 257, 6260, 13, 314, 1101, 257, 6260,
                          13, 314, 1101, 257, 6260, 13, 314, 1101, 257, 6260, 13, 314, 1101, 257, 6260,
                          13, 314}; 

    // // Test max generation time
    // int N_tokens =  2016;
    // int text[N_tokens];
    // int num_new_tokens = 32;
    // for(int i = 0; i < N_tokens; i++) text[i] = rand() %(Vocab_OG + 1);

    int N_compute = ((N_tokens + 32 - 1)/32)*32;

    float *d_input, *d_output, *d_Q, *d_K, *d_QK, *d_V, *d_ACT, *d_ACT_wide, *d_linear[N_Layers][4], 
      *d_bias[N_Layers][4], *d_ln[N_Layers][4], *d_mlp1[N_Layers],  *d_mlp_bias1[N_Layers],
      *d_mlp2[N_Layers], *d_mlp_bias2[N_Layers], *d_final_ln[2],  *d_proj_linear, *d_act2, *d_emb,
      *d_pos, *d_input2;
      
    int *d_max_idx;

    float *h_input, *h_output, ***h_linear, ***h_bias, ***h_ln,
           **h_mlp1, **h_mlp_bias1, **h_mlp2, **h_mlp_bias2,
           *h_final_ln[2], *h_proj_linear, *h_ans, *h_pos, *h_emb;


    h_input = new float[N*Dim];
    h_output = new float[N*Vocab];
    h_ans = new float[N*Vocab];
    h_pos = new float[N*Dim];
    h_emb = new float[Dim*Vocab_OG];
    h_linear = new float **[N_Layers];
    h_bias = new float **[N_Layers];
    h_ln = new float **[N_Layers];
    h_mlp1 = new float *[N_Layers];
    h_mlp_bias1 = new float *[N_Layers];
    h_mlp2 = new float *[N_Layers];
    h_mlp_bias2 = new float *[N_Layers];

    for (int i = 0; i < N_Layers; i++){

      h_linear[i] = new float *[4];
      h_bias[i] = new float *[4];
      h_ln[i] = new float *[4];

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

    read_gpt_weights(Dim, N,  Vocab_OG,  N_tokens,  N_Layers, h_input, h_linear, h_bias,
                    h_ln, h_mlp1, h_mlp_bias1, h_mlp2, h_mlp_bias2, h_final_ln, 
                    h_proj_linear, h_ans, h_pos, h_emb, h_proj_linear_og);
    

    for (int i = 0; i < N_tokens; i++){
      for (int j = 0; j < Dim; j++){
        h_input[i*Dim + j] = h_emb[text[i]*Dim + j] + h_pos[i*Dim + j];
      }
    }

    // copy from h_proj_linear_og to h_proj_linear
    for (int i = 0; i < Dim; i++){
      for (int j = 0; j < Vocab_OG; j++){
        h_proj_linear[i*Vocab + j] = h_proj_linear_og[i*Vocab_OG + j];
      }
    }

    // allocate device space
    cudaMalloc(&d_input, N*Dim*sizeof(float));
    cudaMalloc(&d_input2, N*Dim*sizeof(float));
    cudaMalloc(&d_output, N*Vocab*sizeof(float));
    cudaMalloc(&d_Q, N*Dim*sizeof(float));
    cudaMalloc(&d_K, N*Dim*sizeof(float));
    cudaMalloc(&d_V, N*Dim*sizeof(float));  
    cudaMalloc(&d_QK, num_heads*N*N*sizeof(float));
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

    int new_token[0];

    setbuf(stdout, NULL);
    // Initialization timing
    t1 = clock();
    t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
    printf("Init took %f seconds.  Begin compute\n", t1sum);


    for (int z = 0; z < num_new_tokens; z++){
    // while(1){

      t0 = clock();

      Transformer(d_input, d_Q, d_K, d_V, d_QK, d_ACT, d_ACT_wide, d_linear, d_bias,
      d_ln, d_mlp1, d_mlp_bias1, d_mlp2, d_mlp_bias2, d_final_ln, d_proj_linear, d_output, 
      Dim, N, N_tokens,h_output, d_act2, d_max_idx, new_token, d_emb, d_pos, d_input2, N_compute);
      cudaCheckErrors("kernel launch failure");

      cudaDeviceSynchronize();

      cudaMemcpy(new_token, d_max_idx, sizeof(int), cudaMemcpyDeviceToHost);
      cudaCheckErrors("cudaMemcpy D2H failure");
      cudaDeviceSynchronize();

      // printf("token num %d: %d\n", z, *new_token);
      // printf("%d, ", *new_token);
      cudaDeviceSynchronize();

      N_tokens++;
      N_compute = ((N_tokens + block_size - 1)/block_size)*block_size;
      if(z == 0) printf("N compute tokens = %d \n", N_compute);

      // Initialization timing
      t1sum = ((double)(clock() - t0))/CLOCKS_PER_SEC;
      // printf("Time per token = %f seconds\n", t1sum);
      // printf("%d, %f second\n ", *new_token, t1sum);
      printf("%d, ", *new_token);

    }

    printf("\nTime for %d tokens = %f seconds\n", num_new_tokens, ((double)(clock() - t1))/CLOCKS_PER_SEC);

    return 0;

}
  