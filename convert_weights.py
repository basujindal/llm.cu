
# save in a format readable by c
import numpy as np
import torch
import os

data = torch.load('pytorch_model.bin', map_location='cpu')

for i in data.keys():
    print(i, data[i].shape)

os.makedirs("gpt_weights", exist_ok=True)
for key in data.keys():
    if "attn.c_attn" in key:
        # split to q, k, v
        q, k, v = np.split(data[key].numpy(), 3, axis=-1)
        q.tofile(f"gpt_weights/{key}.q.bin")
        k.tofile(f"gpt_weights/{key}.k.bin")
        v.tofile(f"gpt_weights/{key}.v.bin")
    else:
        data[key].numpy().tofile(f"gpt_weights/{key}.bin")

# save transpose of wte.weight as etw.weight

etw = data['wte.weight'].numpy().T
etw.tofile("gpt_weights/etw.weight.bin")

