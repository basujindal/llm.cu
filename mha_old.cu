    // dim3 grid_attn((Dim + threads.y - 1)/block_size, (N_compute + threads.x - 1)/block_size);
    // dim3 grid_transpose((N_compute + threads.y - 1)/block_size, (N_compute + threads.x - 1)/block_size);

    // for (int i = 0; i < num_heads; i++){
      
    //   uint qk_offset = i*N*N;
    //   // printf("QK\n"); 
    //   matmul_attn_transpose<<<grid_transpose, threads>>>(d_Q, d_K, d_QK + qk_offset, N_compute, N_compute, Dim, head_dim, i);
    //   cudaCheckErrors("kernel launch failure");
    //   // isnan_test<<<1, 1>>>(d_QK, N, N);
      
    //   // scale by sqrt(d_k)
    //   // printf("Scale\n");
    //   scale<<<N, block_size>>>(d_QK + qk_offset, N_compute, head_dim);
    //   cudaCheckErrors("kernel launch failure");
    //   // isnan_test<<<1, 1>>>(d_QK, N, N);

    //   // Set non tokens to -infinity
    //   // printf("Set non tokens to -infinity\n");
    //   set_inf<<<N_tokens, block_size>>>(d_QK + qk_offset, N_compute, N_tokens);
    //   cudaCheckErrors("kernel launch failure");
    //   // isnan_test<<<1, 1>>>(d_QK, N, N);

    //   // Softmax
    //   // printf("Softmax\n");
    //   softmax_max<<<N_tokens, block_size>>>(d_QK + qk_offset, N_compute);
    //   cudaCheckErrors("kernel launch failure");
    //   // isnan_test<<<1, 1>>>(d_QK, N, N);

    //   set_zero<<<N, block_size>>>(d_QK + qk_offset, N_compute, N_compute, N_tokens);
    //   cudaCheckErrors("kernel launch failure");
    //   // isnan_test<<<1, 1>>>(d_QK, N, N);

    //   // printf("QK_V\n");
    //   matmul_attn<<<grid_attn, threads>>>(d_QK + qk_offset, d_V, d_act, N_compute, head_dim, N_compute, head_dim, i, Dim);
    //   cudaCheckErrors("kernel launch failure");
    //   // isnan_test<<<1, 1>>>(d_act, head_dim, N);
    //   cudaDeviceSynchronize();

    // }