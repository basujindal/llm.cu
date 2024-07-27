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


const int blocks = 6;
const int dim = 2048;
const int block_size = 1024;  // CUDA maximum is 1024
int tokens = 6;


__global__ void layernorm(float *A, int dim, float *gamma, float *beta){

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
        A[dim*blockIdx.x + i*blockDim.x + idx] = (A[dim*blockIdx.x + i*blockDim.x + idx] - sum)*sdata[0]*gamma[i*blockDim.x + idx] + beta[i*blockDim.x + idx];
    }

}

__global__ void gelu(float *A, int dim, int tokens){

    int idx = threadIdx.x;
    float x;

    if(blockIdx.x < tokens){
        for(int i = 0; i < dim/blockDim.x; i++){
            x = A[dim*blockIdx.x + i*blockDim.x + idx];
            // A[dim*blockIdx.x + i*blockDim.x + idx] = x*0.5*(1.0 + tanhf(0.7978845608*(x + 0.044715*x*x*x)));
            A[dim*blockIdx.x + i*blockDim.x + idx] = x/(1 + expf(-1.702*x));

        __syncthreads();    
        }
    }

}

__global__ void add(float *A, float *B, int dim, int tokens){

    int idx = threadIdx.x;

    if(blockIdx.x < tokens){
        if (blockIdx.x < sizeof(A)/sizeof(float)/dim){
            for(int i = 0; i < dim/blockDim.x; i++){
                B[dim*blockIdx.x + i*blockDim.x + idx] = A[dim*blockIdx.x + i*blockDim.x + idx] + B[dim*blockIdx.x + i*blockDim.x + idx];
            }
        }
    }

}


bool validate_add(float *dataA, float *dataB, int dim, float *ans){

    for (int i = 0; i < tokens; i++){
        for (int j = 0; j < dim; j++){
            if (abs(ans[i*dim + j] - dataA[i*dim + j] - dataB[i*dim + j])  > 0.01) {
                printf("results mismatch at %d, was: %.10f, should be: %.10f\n", i*dim + j, ans[i*dim + j], dataA[i*dim + j] + dataB[i*dim + j]);
                return false;
            }
        }
    }

    return true;

}

bool validate_gelu(float *data, int dim, float* ans){

    for (int i = 0; i < tokens; i++){
        for (int j = 0; j < dim; j++){
            // if (abs(ans[i*dim + j] - data[i*dim + j]*0.5*(1.0 + tanhf(0.7978845608*(data[i*dim + j] + 0.044715*data[i*dim + j]*data[i*dim + j]*data[i*dim + j])))) > 0.01) {
            //     printf("results mismatch at %d, was: %.10f, should be: %.10f\n", i*dim + j, ans[i*dim + j], data[i*dim + j]*0.5*(1.0 + tanhf(0.7978845608*(data[i*dim + j] + 0.044715*data[i*dim + j]*data[i*dim + j]*data[i*dim + j]))));
            if (abs(ans[i*dim + j] - data[i*dim + j]/(1 + expf(-1.702*data[i*dim + j]))) > 0.01) {
                printf("results mismatch at %d, was: %.10f, should be: %.10f\n", i*dim + j, ans[i*dim + j], data[i*dim + j]/(1 + expf(-1.702*data[i*dim + j])));
                return false;
            }
        }
    }

    return true;
}

bool validate_ln(float *data, int dim, float *gamma, float *beta, float* ans){

    for (int i = 0; i < dim; i++){

        float var = 0;
        float sum = 0;
        
        for (int j = 0; j < dim; j++) sum += data[i*dim + j];
        sum = sum/dim;
        
        for (int j = 0; j < dim; j++) var += (data[i*dim + j] - sum)*(data[i*dim + j] - sum);
        
        var = 1/sqrt(var/dim + 0.00001);

        for (int j = 0; j < dim; j++){
            if (abs(ans[i*dim + j] - ((data[i*dim + j] - sum)*var*gamma[j] + beta[j])) > 0.01) {
                printf("results mismatch at %d, was: %.10f, should be: %.10f\n", i*dim + j, ans[i*dim + j], (data[i*dim + j] - sum)*var*gamma[j] + beta[j]);
                return false;
            }
        }
    }

    return true;
}

int main(){

    float *h_A, *d_A, *h_beta, *h_gamma, *d_beta, *d_gamma, *h_ans, *h_B, *d_B;

    h_A = new float[tokens*dim];
    h_B = new float[tokens*dim];

    h_beta = new float[dim];
    h_gamma = new float[dim];
    h_ans = new float[tokens*dim];

    for (int i = 0; i < tokens*dim; i++) h_A[i] =  rand()/float(RAND_MAX);
    for (int i = 0; i < dim; i++) h_beta[i] =  rand()/float(RAND_MAX);
    for (int i = 0; i < dim; i++) h_gamma[i] =  rand()/float(RAND_MAX);

    cudaMalloc(&d_A, tokens*dim*sizeof(float));
    cudaMalloc(&d_B, tokens*dim*sizeof(float));
    cudaMalloc(&d_beta, dim*sizeof(float));  
    cudaMalloc(&d_gamma, dim*sizeof(float));


    cudaCheckErrors("cudaMalloc failure"); // error checking
    cudaMemcpy(d_A, h_A, tokens*dim*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, tokens*dim*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, dim*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, dim*sizeof(float), cudaMemcpyHostToDevice);

    cudaCheckErrors("cudaMemcpy H2D failure");

    // layernorm<<<DSIZE, block_size>>>(d_A, dim, d_gamma, d_beta);
    // gelu<<<DSIZE, block_size>>>(d_A, dim);

    add<<<blocks, block_size>>>(d_A, d_B, dim, tokens);
    cudaCheckErrors("kernel launch failure");


    cudaMemcpy(h_ans, d_A, tokens*dim*sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");

    // if(validate_ln(h_A, dim, h_gamma, h_beta, h_ans)) printf("softmax correct!\n");
    // if(validate_gelu(h_A, dim, h_ans)) printf("gelu correct!\n");
    if (validate_add(h_A, h_B, dim, h_ans)) printf("add correct!\n");
    
  return 0;
}
  
