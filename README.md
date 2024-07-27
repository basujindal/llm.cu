# llm.cu

- GPT-2 in CUDA
- All the kernels are written from scratch
- Experiments are done on RTX 2060

## Performance

Time refers to generating 32 tokens starting from the 2016 till 2048 tokens.

Iteration 1: 48.5s
Iteration 2: ?

## Future Work

- [ ] Use 3D grid for the MultiHeadAttention
- [ ] Combine the kernels
- [ ] Add tests
- [ ] Use Flash Attention
- [ ] KV cache
- [ ] Use Tensor Cores
- [ ] Support for batch size > 1
- [ ] Support for fp16/int8
- [ ] Support for multi-GPU

## Small improvements

- [ ] Remove weights from cpu

