

# llm.cu

- GPT-2 in CUDA
- All the kernels are written from scratch
- Experiments are done on RTX 2060 mobile GPU

## Download and conver weights to c readable

```
pip install -U "huggingface_hub[cli]"
huggingface-cli download gpt2 pytorch_model.bin --local-dir .
python convert_weights.py
```

## Run 

```
nvcc -arch=sm_75 --allow-unsupported-compiler llm.cu load_weights.cu kernels.cu -o test && ./test
```

## Performance

Time refers to generating 32 tokens starting from the 2016 till 2048 tokens.

Iteration 1: ~= 49s

## Future Work

### Performance

- [x] Use 3D grid for the MultiHeadAttention
- [ ] Use faster matmul algo
- [ ] Ask Claude/ChatGPT for suggestions on each kernel
- [ ] KV cache
- [ ] Combine the kernels
- [ ] Use Flash Attention
- [ ] Use Tensor Cores
- [ ] Support for batch size > 1
- [ ] Support for fp16/int8
- [ ] Support for multi-GPU

## Code quality

- [ ] Add tests
- [ ] Add performance benchmark
- [ ] Improve readability
- [ ] Use appropriate dtypes, like uint, size_t
- [ ] Remove weights from cpu



