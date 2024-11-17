# llm.cu

- GPT-2 in CUDA
- All the kernels are written from scratch
- Experiments are done on RTX 2060 mobile GPU

## Performance

Time refers to generating 32 tokens starting from the 2016 till 2048 tokens.

Iteration 1: ~= 49s
Iteration 2: 2

## Future Work

- [ ] Use 3D grid for the MultiHeadAttention
- [ ] KV cache
- [ ] Combine the kernels
- [ ] Add tests
- [ ] Use Flash Attention
- [ ] Use Tensor Cores
- [ ] Support for batch size > 1
- [ ] Support for fp16/int8
- [ ] Support for multi-GPU
- [ ] Remove weights from cpu



