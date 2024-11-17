#ifndef KERNEL_H
#define KERNEL_H

int read_weight(float *arr, char *filename, int rows, int cols);

void read_gpt_weights(int Dim, int N, int Vocab_OG, int N_tokens, int N_Layers, 
                    float *h_input, float ***h_linear, float ***h_bias,
                    float ***h_ln, float **h_mlp1, float **h_mlp_bias1,
                    float **h_mlp2, float **h_mlp_bias2, float **h_final_ln, 
                    float *h_proj_linear, float *h_ans, float *h_pos, float *h_emb, 
                    float *h_proj_linear_og);

#endif