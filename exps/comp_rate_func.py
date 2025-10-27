import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


model_id = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="float16", device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id)

num_blocks = len(model.model.decoder.layers)
print(f"Model has {num_blocks} blocks.")

module_names = []
module_size = []
for i in range(num_blocks):
    block = model.model.decoder.layers[i]
    print(f"Modules in block {i}:")
    for name, module in block.named_modules():
        if isinstance(module, nn.Linear):
            full_name = f"model.decoder.layers.{i}.{name}"
            module_names.append(full_name)
            module_size.append(module.weight.numel())
            print(f" - {full_name}")

sparsity = 0.5
module_sparsity = [sparsity for _ in module_names]
print(f"Target sparsity: {sparsity}")
print(f"Total number of linear modules to prune: {len(module_names)}")
print(f"Obtain the performance of the compressed model for different sparsity profiles")

def obtain_sparsity_profile(module_names, target_sparsity):
    module_sparsity = [random.random() for _ in module_names]
    