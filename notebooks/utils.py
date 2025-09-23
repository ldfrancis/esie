import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import random
import math
import gc
torch.cuda.is_available()



#===================================================================================================================================================================================================================
# CALIBRATION AND EVALUATION DATA
#===================================================================================================================================================================================================================

# WIKITEXT train and test tokens
def get_w2_data(num_samples, seq_len, tokenizer):
    w2_train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    w2_val_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    train_tokens = tokenizer("\n\n".join(w2_train_dataset["text"]), return_tensors="pt", add_special_tokens=False).input_ids
    test_tokens = tokenizer("\n\n".join(w2_val_dataset["text"]), return_tensors="pt", add_special_tokens=False).input_ids
    
    num_tokens = train_tokens.size(1)
    idxes = np.random.choice(num_tokens-seq_len, num_samples, replace=False).tolist()
    test_idxes = range(test_tokens.size(1)//seq_len)
    
    train_data = list(map(lambda idx: train_tokens[:, idx:idx+seq_len], idxes))
    test_data = list(map(lambda idx: test_tokens[:, idx*seq_len:(idx+1)*seq_len], test_idxes))

    return train_data, test_data

# C4 train and test tokens
def get_c4_data(num_samples, seq_len, tokenizer):
    c4_train_dataset = load_dataset("allenai/c4", "default", data_files={"train": "en/c4-train.00000-of-01024.json.gz"},split="train", revision="607bd4c8450a42878aa9ddc051a65a055450ef87")
    c4_val_dataset = load_dataset("allenai/c4", "default", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"}, split="validation[:1100]", revision="607bd4c8450a42878aa9ddc051a65a055450ef87")

    train_tokens = tokenizer("\n\n".join(c4_train_dataset["text"][:seq_len+10]), return_tensors="pt", add_special_tokens=False).input_ids
    test_tokens = tokenizer("\n\n".join(c4_val_dataset["text"]), return_tensors="pt", add_special_tokens=False).input_ids
    
    num_tokens = train_tokens.size(1)
    idxes = np.random.choice(num_tokens-seq_len, num_samples, replace=False).tolist()
    test_idxes = range(test_tokens.size(1)//seq_len)
    
    train_data = list(map(lambda idx: train_tokens[:, idx:idx+seq_len], idxes))
    test_data = list(map(lambda idx: test_tokens[:, idx*seq_len:(idx+1)*seq_len], test_idxes))

    return train_data, test_data

# Fineweb:
# Source: https://github.com/IST-DASLab/EvoPress/blob/main/src/data_utils.py
def get_fineweb_edu(num_tokens, sequence_length, tokenizer, train = True):
    print("Loading FineWeb-Edu v2")
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train")
    tokens_to_load = num_tokens
    if train:
        dataset = dataset.select(range(dataset.num_rows//2))
    else:
        dataset = dataset.select(range(dataset.num_rows//2, dataset.num_rows))
    dataset = dataset.shuffle(seed=0)
    data_iter = iter(dataset)
    data = []
    while tokens_to_load > 0:
        sample = next(data_iter)
        tokenized_sample = tokenizer(sample["text"], return_tensors="pt", add_special_tokens=False).input_ids
        tokenized_sample = tokenized_sample[:, :min(tokenized_sample.shape[1], tokens_to_load)]
        # Split the sequence into multiple samples if it is too long
        # Just throwing away extra tokens would introduce bias to the dataset
        while tokenized_sample.shape[1] > sequence_length:
            data.append(tokenized_sample[:, :sequence_length])
            tokenized_sample = tokenized_sample[:, sequence_length:]
            tokens_to_load -= sequence_length
        if tokenized_sample.shape[1] == sequence_length:
            data.append(tokenized_sample)
            tokens_to_load -= tokenized_sample.shape[1]
    print(f"Total tokens loaded: {sum([sample.shape[1] for sample in data])}")
    return data



#===================================================================================================================================================================================================================
# WANDA PRUNING IMPLEMENTATION
#===================================================================================================================================================================================================================

# Wanda pruning implementation - https://github.com/locuslab/wanda/blob/main/lib/eval.py

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, theta1=0.42, theta2=0.51, theta3=0.38, is_sparsegpt=False):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3

        self.is_sparsegpt = is_sparsegpt
        if is_sparsegpt: 
            self.H = torch.zeros((self.columns, self.columns), device=self.dev)

        w = (self.layer.weight.data.clone().float()).abs()
        ws = w**2
        wrow = 1/ws.sum(dim=0, keepdims=True).sqrt()
        wcol = 1/ws.sum(dim=1, keepdims=True).sqrt()
        self.weight_imp = 0.5*w * (wrow**(self.theta1) + wcol**(self.theta2))

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        a = self.nsamples / (self.nsamples+tmp)
        self.scaler_row *= a
        if self.is_sparsegpt: self.H *= a
        self.nsamples += tmp

        b = 1 / self.nsamples
        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  * b
        inp = math.sqrt(2*b)*inp
        if self.is_sparsegpt: self.H += inp.matmul(inp.t())

    def get_metric(self):
        return self.weight_imp * torch.sqrt(self.scaler_row.reshape((1, -1)))**self.theta3

    def prune(self, sparsity_ratio, prune_n=0, prune_m=0):
        if self.is_sparsegpt:
            W = self.layer.weight.data.clone()
            W = W.float()
            H = self.H
            del self.H
            dead = torch.diag(H) == 0
            H[dead, dead] = 1
            W[:, dead] = 0
            Losses = torch.zeros(self.rows, device=self.dev)
            percdamp = 0.01
            blocksize = 128
            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(self.columns, device=self.dev)
            H[diag, diag] += damp
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
            Hinv = H
            mask = None
            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1
                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[i1:i2, i1:i2]
                if prune_n == 0:
                    if mask is not None:
                        mask1 = mask[:, i1:i2]
                    else:
                        tmp = W1**2 / (torch.diag(Hinv1).reshape((1, -1)))**2
                        thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity_ratio)]
                        mask1 = tmp <= thresh
                else:
                    mask1 = torch.zeros_like(W1) == 1
                for i in range(count):
                    w = W1[:,  i]
                    d = Hinv1[i, i]
                    if prune_n != 0 and i%prune_m == 0:
                        tmp = W1[:, i:(i+prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i+prune_m)].reshape((1, -1)))**2
                        mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
                    q = w.clone()
                    q[mask1[:, i]] = 0
                    Q1[:, i] = q
                    Losses1[:, i] = (w-q)**2 / d**2
                    err1 = (w-q)/d
                    W1[:,i:] -= err1.unsqueeze(1).matmul(Hinv1[i,i:].unsqueeze(0))
                    Err1[:, i] = err1
                W[:, i1:i2] = Q1
                Losses += torch.sum(Losses1, 1)/2
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
            torch.cuda.synchronize()
            self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.dtype)
        else:
            W_metric = self.get_metric()
            W_mask = (torch.zeros_like(W_metric) == 1)  
            sort_res = torch.sort(W_metric, dim=-1, stable=True)
            indices = sort_res[1][:,:int(W_metric.shape[1]*sparsity_ratio)]
            W_mask.scatter_(1, indices, True)
            self.layer.weight.data[W_mask] = 0 

    def clean(self):
        del self.scaler_row
        del self.weight_imp
        gc.collect()
        torch.cuda.empty_cache()

@torch.no_grad()
def prepare_calibration_input(model, calib_data, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    NUM_SAMPLES = len(calib_data)
    SEQUENCE_LEN = calib_data[0].shape[1]

    if hasattr(model, "hf_device_map") and "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((NUM_SAMPLES, SEQUENCE_LEN, model.config.hidden_size), dtype=dtype, device="cpu")
    inps.requires_grad = False
    cache = {'i': 0, 'kwargs':{}}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['kwargs'].update(kwargs)
            raise ValueError
    layers[0] = Catcher(layers[0])
    model.model.embed_tokens.to(device)
    for sample in calib_data:
        try:
            model(sample.to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    model.config.use_cache = use_cache

    return inps, outs, cache['kwargs']

@torch.no_grad()
def prune_default(model, calib_data, sparsity_ratio, theta1=0.42, theta2=0.51, theta3=0.38, is_sparsegpt=False, device=torch.device("cuda:0")):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    reconstruction_errors = []
    NUM_SAMPLES = len(calib_data)

    print("loading calibration data")
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, kwargs = prepare_calibration_input(model, calib_data, device)

    layers = model.model.layers
    for i in range(len(layers)):
        layer_recon_error = 0
        layer = layers[i]
        # layer.to(device)
        subset = find_layers(layer)

        if hasattr(model, "hf_device_map") and f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs = inps.to(dev), outs.to(dev)
            n_kwargs = {}
            for k in kwargs:
                n_kwargs[k] = kwargs[k].to(dev)
            kwargs = n_kwargs

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name], theta1=theta1, theta2=theta2, theta3=theta3, is_sparsegpt=is_sparsegpt)

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(NUM_SAMPLES):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0).to(device), **kwargs)[0].detach().cpu()
        for h in handles:
            h.remove()

        for name in subset:
            wrapped_layers[name].prune(sparsity_ratio)
            wrapped_layers[name].clean()

        # outs_cache = outs.clone()
        for j in range(NUM_SAMPLES):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0).to(device), **kwargs)[0].detach().cpu()

        # layer_recon_error = ((outs_cache.float() - outs.float())**2).sum().item()
        
        # reconstruction_errors += [layer_recon_error]
        # print(f"Layer {i} reconstruction error: {layer_recon_error}")
        inps, outs = outs, inps
        # layer.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache 

    return reconstruction_errors

@torch.no_grad()
def eval_ppl(model, test_data, seqlen,  bs=1, device=None):
    model.to(device)
    # Get input IDs
    testenc = test_data

    # Calculate number of samples
    nsamples = len(test_data)

    # List to store negative log likelihoods
    nlls = []
    # print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        # if i % 50 == 0:
        #     print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        # inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        # inputs = inputs.reshape(j-i, model.seqlen)
        inputs = torch.cat(testenc[i:j], dim=0).to(device)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()

