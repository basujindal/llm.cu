# llm.cu

GPT-2 small model from scratch in CUDA. All the kernels are written from scratch.

Experiments are done on RTX 2060.


## Performance

Iterration 1: 

- Generation from initial lenght of 2016 tokens = ~1.5s


## Future Work

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

