import struct
import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype='auto')

script_dir = os.path.dirname(__file__)
weights_dir = f"{script_dir}/../weights"
os.makedirs(weights_dir, exist_ok=True)

config = model.config
# save the model weights to a .bin file
with open(f"{weights_dir}/llama2_7b_weights_fp32.bin", "wb") as f:
    # write header
    header = np.zeros(256, dtype=np.uint32)
    header[0] = 1 # id
    header[1] = 1 # version
    header[2] = config.vocab_size
    header[3] = config.hidden_size
    header[4] = config.intermediate_size
    header[5] = config.num_attention_heads
    header[6] = config.num_key_value_heads
    header[7] = config.max_position_embeddings
    header[8] = config.head_dim
    header[9] = config.num_hidden_layers
    f.write(header[:10].tobytes())

    # Pack as float 32
    f.write(struct.pack("<f", float(config.rms_norm_eps)))
    f.write(struct.pack("<f", float(config.rope_theta)))
    f.write(struct.pack("<f", float(1.0)))  # partial_rotary_factor

    # remaining header entries
    rem = 256 - 13
    f.write(np.zeros(rem, dtype=np.uint32).tobytes())

    for name, param in model.named_parameters():
        f.write(param.detach().to(torch.float32).numpy().tobytes())
print(f"Model weights saved to {weights_dir}/llama2_7b_weights_fp32.bin")

model.to(torch.float32)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokens = tokenizer.encode("How far", return_tensors="pt")
position_ids = torch.arange(3).unsqueeze(0)

res = model(tokens)
with open(f"{weights_dir}/llama2_7b_logits.bin", "wb") as f:
    f.write(res.logits.detach().to(torch.float32).numpy().tobytes())
print(f"Logits saved to {weights_dir}/llama2_7b_logits.bin")