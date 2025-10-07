from typing import Optional, Dict, Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import gymnasium as gym
import os
try:
    import wandb
except ImportError:
    print("wandb not installed. WandBLogger will not work.")
    wandb = None

device = "cuda" if torch.cuda.is_available() else "cpu"

from utils import *



model_name = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)


save_dir = f"logs/sparse_weights/{model_name.split('/')[-1]}"
num_samples = 128
sequence_length = 2048
sparsity_ratios = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
num_tokens = num_samples * sequence_length
calib_data = get_fineweb_edu(num_tokens, sequence_length, tokenizer, train=True)
_, test_data = get_w2_data(num_samples, sequence_length, tokenizer)


# S = 0.5
# b = 0.000


# for i in range(100):
#     s = []
#     to_stop = False
#     for j in range(1, 13):
#         v = (S - (b*(12-1))/2 + b*(j-1))
#         if v < 0 or v > 1:
#             to_stop = True
#             break
#         s.append(S - (b*(12-1))/2 + b*(j-1))
#     if to_stop:
#         print("Stopping early")
#         break
#     print(s)
#     prune_default(model, calib_data, s, theta1=0, theta2=0, theta3=1, is_sparsegpt=False, device=torch.device("cuda"))
#     ppl = eval_ppl(model, test_data, sequence_length, device="cuda")
#     print(f"Iteration {i}, ppl: {ppl}")
#     b += 0.002
#     model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="cpu")
# breakpoint()

