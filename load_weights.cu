#include<stdio.h>
#include "load_weights.h"
using namespace std;

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


void read_gpt_weights(int Dim, int N, int Vocab_OG, int N_tokens, int N_Layers, float *h_input,
                   float ***h_linear, float ***h_bias,
                    float ***h_ln, float **h_mlp1, float **h_mlp_bias1,
                    float **h_mlp2, float **h_mlp_bias2, float **h_final_ln, 
                    float *h_proj_linear, float *h_ans, float *h_pos, float *h_emb, float *h_proj_linear_og){
    // initialize matrix in host memory
    char filename[256];

    snprintf(filename, sizeof(filename), "gpt_weights/wpe.weight.bin");
    read_weight(h_pos, filename, N, Dim);

    snprintf(filename, sizeof(filename), "gpt_weights/wte.weight.bin");
    read_weight(h_emb, filename, Vocab_OG, Dim);

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

    snprintf(filename, sizeof(filename), "gpt_weights/output.bin");
    read_weight(h_ans, filename, N_tokens, Vocab_OG);
}