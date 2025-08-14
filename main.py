# Llama
# Phi
# Qwen
# Mistral
# Gemma
# GPT-OSS

# Possibly Vision Models
import struct
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype='auto')
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokens = tokenizer.encode("How far", return_tensors="pt")

decoder_model = model.model
lm_head = model.lm_head
embed_tokens = decoder_model.embed_tokens
layers = decoder_model.layers

# save the model weights to a .bin file
with open("model_weights.bin", "wb") as f:
    # write header
    header = np.zeros(256, dtype=np.int32)
    for name, param in model.named_parameters():
        breakpoint()

breakpoint()
model.generate(tokens)

breakpoint()

# import torch
# import torch.nn as nn

# class Branchy(nn.Module):
#     def forward(self, x):
#         # data-dependent if
#         if x.sum() > 0:
#             y = x * 2
#         else:
#             y = -x
#         # data-dependent loop
#         i = 0
#         while (y.abs().mean() > 0.1) and (i < 5):
#             y = y * 0.5
#             i += 1
#         return y

# m = Branchy()
# sm = torch.jit.script(m)   # scripting captures control flow
# print(sm.graph) 
# # print(torch.jit.script(model).graph)
# model.model(0)
# breakpoint()