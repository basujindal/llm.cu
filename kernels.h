#ifndef KERNELS_H
#define KERNELS_H

// Function declarations
__global__ void softmax_max(float *A, size_t ds);
__global__ void layernorm(float *A, float *B, int dim, float *gamma, float *beta);
__global__ void matmul(const float *A, const float *B, float *C, int height, int width, int dim) ;
__global__ void matmul_bias(const float *A, const float *B, float *C, float *bias, int height, int width, int dim, int N_tokens);
__global__ void QK_V(const float *QK, const float *V, float *C, int Dim, int N);;
__global__ void gelu(float *A, int dim);
__global__ void add(float *A, float *B, int dim);
__global__ void scale(float *A, int N, int head_dim);
__global__ void set_inf(float *A, int dim, int N, int N_tokens);
__global__ void set_zero(float *A, int dim, int N, int N_tokens);
__global__ void isnan_test(float *data, int width, int height);
__global__ void matmul_mha_transpose(const float *A, const float *B, float *C, int height, int width, int dim, int head_dim, int head_num);
__global__ void matmul_mha(const float *A, const float *B, float *C, int height, int width, int dim, int head_dim, int head_num, int width_full);
__global__ void max_index(float *A, size_t height, size_t width, int *max_idx);
__global__ void set_new_embedding(float *A, int dim, int N_tokens, float *emb, int *new_token, float *pos);

// Class declarations

#endif // KERNELS_H
