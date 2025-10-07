# EvoPress
import os
import time
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy
from tqdm import trange

from utils import *

try:
    import wandb

    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False


def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def compute_kl_div(model, data, target_logits, batch_size: int = 1):
    num_samples = len(data)
    device = next(model.parameters()).device
    # Running estimate of negative log-likelihood
    kl_div_running = 0
    # Number of tokens processed to far
    tokens_processed = 0
    # Loop through each batch
    for i in trange(0, num_samples, batch_size, desc="Computing KL Divergence", leave=False):
        torch.cuda.empty_cache()
        j = min(i + batch_size, num_samples)
       
        inputs = torch.cat(data[i:j]).to(device)
        targets = torch.cat(target_logits[i:j]).to(device)
        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Don't predict last token (not required, can be removed)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_targets = targets[:, :-1, :]
        
      
        #Squeeze on GPU 
        torch.cuda.empty_cache()
        for i in range(0, shift_logits.shape[1], 1024):
            j = min(i + 1024, shift_logits.shape[1])
            shift_logits_batch = shift_logits[:, i:j, :]
            shift_targets_batch = shift_targets[:, i:j, :]
            loss_batch = F.kl_div(
                shift_logits_batch.reshape(-1, shift_logits_batch.size(-1)).log_softmax(dim=-1),
                shift_targets_batch.reshape(-1, shift_targets_batch.size(-1)).log_softmax(dim=-1),
                log_target=True,
                reduction="batchmean",
            )
            # Calculate negative log likelihood
            a = shift_targets_batch.numel() / (tokens_processed + shift_targets_batch.numel())
            b = tokens_processed / (tokens_processed + shift_targets_batch.numel())
            kl_div_running = a * loss_batch + b * kl_div_running
            # Update number of processed tokens
            tokens_processed += shift_targets_batch.numel()
            del shift_logits_batch, shift_targets_batch, loss_batch
            torch.cuda.empty_cache()      
        
 
    return kl_div_running.item()

def load_layers(model: AutoModelForCausalLM, layer_names: List[str], new_state: List[int], sparse_weights_path: str):
    assert hasattr(model, "state")
    for layer_name, new_level, old_level in zip(layer_names, new_state, model.state):
        if new_level != old_level:
            layer = model.get_submodule(layer_name)
            layer.weight.data = torch.load(
                os.path.join(sparse_weights_path, layer_name, f"{new_level}.pth"), map_location=layer.weight.device
            ).to(layer.weight.dtype)
    # Update model state
    model.state = new_state

def compute_fitness(model, data, fitness_fn, target_logits: Optional[torch.Tensor] = None) -> float:
    if fitness_fn == "ppl":
        return eval_ppl(model, data, data[0].shape[-1], device="cuda")
    else:
        return compute_kl_div(model, data, target_logits)



def selection(
    model,
    layer_names,
    sparse_weights_path: str,
    candidates,
    num_survive: int,
    calibration_data,
    num_tokens: int,
    fitness_fn: str = "ppl",
    target_logits: Optional[List[torch.Tensor]] = None,
):
    calibration_minibatch = []
    minibatch_ids = []
    target_logits_minibatch = []
    tokens_used = 0
    while tokens_used < num_tokens:  # generate minibatch with exactly num_tokens tokens
        minibatch_id = random.randint(0, len(calibration_data) - 1)
        if minibatch_id in minibatch_ids:  # avoid duplicates
            continue
        minibatch_ids.append(minibatch_id)
        if tokens_used + calibration_data[minibatch_id].shape[1] > num_tokens:
            calibration_minibatch.append(calibration_data[minibatch_id][:, : num_tokens - tokens_used])
            if fitness_fn == "kl":
                target_logits_minibatch.append(target_logits[minibatch_id][:, : num_tokens - tokens_used])
            tokens_used = num_tokens
        else:
            calibration_minibatch.append(calibration_data[minibatch_id])
            if fitness_fn == "kl":
                target_logits_minibatch.append(target_logits[minibatch_id])
            tokens_used += calibration_data[minibatch_id].shape[1]

    if len(target_logits_minibatch) == 0:
        target_logits_minibatch = None

    fitnesses = []
    for candidate in candidates:
        load_layers(model, layer_names, candidate, sparse_weights_path)
        fitness = compute_fitness(model, calibration_minibatch, fitness_fn, target_logits_minibatch)
        fitnesses.append(fitness)
    # Keep only best
    best_ids = np.argsort(fitnesses)[:num_survive]
    return [candidates[i] for i in best_ids], [fitnesses[i] for i in best_ids]



def evopress(model, calibration_data, eval_datasets):
    args = {
        "log_wandb": True,
        "seed": 0,
        "dtype": "float16",
        "model_name_or_path": "facebook/opt-125m",
        "tokenizer_name": "facebook/opt-125m",
        "memory_efficient": True,
        "attn_implementation": "flash_attention_2",
        "use_fast_tokenizer": True,
        "calibration_data": "fineweb-edu",
        "calibration_tokens": 128 * 2048,
        "calibration_sequence_length": 2048,
        "eval_datasets": ["wikitext", "c4"],
        "eval_tokens": 128 * 2048,
        "eval_sequence_length": 2048,
        "fitness_fn": "kl",  # "ppl" or "kl"
        "sparse_weights_path": "/home/user/esie/logs/sparse_weights/opt-125m/sparsegpt",
        "generations": 400,
        "offspring": 64,
        "survivors_per_selection": [8, 2, 1],  # number of survivors after each selection step
        "tokens_per_selection": [2048, 16284, 65536],  # number of tokens used to evaluate fitness at each selection step
        "eval_every": 10,  # evaluate perplexity on eval datasets every N generations
        "max_level": 99999,  # maximum absolute sparsity level per layer
        "max_total_deviation": 99999,  # maximum total sparsity deviation (sum of absolute levels)


    }
    # Fix seed
    fix_seed(args["seed"])
    # Init W&B logger
    if args["log_wandb"]:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        wandb.init(config=args)
    # init device
    device = f"cuda"
    if args["dtype"] != "auto":
        args["dtype"] = getattr(torch, args["dtype"])
    
    model.config.use_cache = False  # do not use cache
  
    target_logits = []
    if args["fitness_fn"] == "kl":
        # Compute target logits (calibration)
        for i in trange(0, len(calibration_data), desc="Computing target logits (calib)", leave=False):
            with torch.no_grad():
                target_logits.append(model(calibration_data[i].to(device)).logits.cpu())

    # Prepare layers and initial state
    layer_names = []
    for layer_name in sorted(os.listdir(args["sparse_weights_path"])):
        if os.path.isdir(os.path.join(args["sparse_weights_path"], layer_name)):
            layer_names.append(layer_name)
    parent = [0 for _ in layer_names]
    model.state = [None] * len(layer_names)

    train_fitness = float("inf")
    log_dict = {}
    for generation in range(args["generations"]):
        
        print(f"Generation {generation + 1}/{args['generations']}")
        print(f"Current search point: {parent}")
        print(f"Train fitness: {train_fitness:.2e}")

        load_layers(model, layer_names, parent, args["sparse_weights_path"])

        # Evaluate current search point
        if generation % args["eval_every"] == 0:
            for eval_dataset_name, eval_dataset in zip(["W2"], eval_datasets):
                sequence_length = eval_dataset[0].shape[-1]
                ppl_eval = eval_ppl(model, eval_dataset, sequence_length, device="cuda")#compute_perplexity(model, eval_dataset)
                print(f"{eval_dataset_name}: {ppl_eval:.2f}")
                log_dict[f"ppl_eval/{eval_dataset_name}"] = ppl_eval
            ppl_train = eval_ppl(model, calibration_data, calibration_data[0].shape[-1], device="cuda")#compute_perplexity(model, calibration_data)
            print(f"ppl_train: {ppl_train:.2f}")
            log_dict["ppl_train"] = ppl_train
        if args["log_wandb"]:
            wandb.log(log_dict)
            # log_dict = {}

        offspring_list = []

        while len(offspring_list) < args["offspring"]:
            offspring = copy.deepcopy(parent)
            # mutate offspring
            num_flips = min(random.randint(1, 3), random.randint(1, 3))  # bias towards lower values
            for _ in range(num_flips):
                # positions where sparsity can be decreased
                while True:
                    decr_id = random.randint(0, len(offspring) - 1)
                    layer_name = layer_names[decr_id]
                    level = offspring[decr_id]
                    if abs(level - 1) > args["max_level"]:
                        continue
                    if os.path.exists(os.path.join(args["sparse_weights_path"], layer_name, f"{level - 1}.pth")):
                        break
                # positions where sparsity can be increased
                while True:
                    incr_id = random.randint(0, len(offspring) - 1)
                    layer_name = layer_names[incr_id]
                    level = offspring[incr_id]
                    if abs(level + 1) > args["max_level"]:
                        continue
                    if os.path.exists(os.path.join(args["sparse_weights_path"], layer_name, f"{level + 1}.pth")):
                        break
                offspring[decr_id] -= 1
                offspring[incr_id] += 1
            # avoid duplicates
            if offspring in offspring_list:
                continue
            # skip if total deviation exceeds specified threshold
            if sum(map(abs, offspring)) > args["max_total_deviation"]:
                continue
            offspring_list.append(offspring)

        for num_survive, num_tokens in zip(args["survivors_per_selection"], args["tokens_per_selection"]):
            if num_survive == args["survivors_per_selection"][-1]:
                if parent not in offspring_list:  # Elitist EA
                    offspring_list.append(parent)

            offspring_list, train_fitnesses = selection(
                model=model,
                layer_names=layer_names,
                sparse_weights_path=args["sparse_weights_path"],
                candidates=offspring_list,
                num_survive=num_survive,
                calibration_data=calibration_data,
                num_tokens=num_tokens,
                fitness_fn=args["fitness_fn"],
                target_logits=target_logits,
            )
        # In the end we have lists with a single element (only 1 survivor in last selection step)
        train_fitness = train_fitnesses[0]
        parent = offspring_list[0]
        print(f"Train fitnesses: {train_fitness:.2e}")
        log_dict["train_fitness"] = train_fitness

    # Save final configuration
    with open(os.path.join(args["sparse_weights_path"], args["configuration_name"]), "w") as f:
        f.write("\n".join([f"{layer_name}: {level}" for layer_name, level in zip(layer_names, parent)]))
    # Log final configuration
    print("Final configuration:")
    print(parent)
    # Final evaluation
    for eval_dataset_name, eval_dataset in zip(["W2"], eval_datasets):
        ppl_eval = eval_ppl(model, eval_dataset, eval_dataset[0].shape[-1], device="cuda")#compute_perplexity(model, eval_dataset)
        print(f"{eval_dataset_name}: {ppl_eval:.2f}")
        log_dict[f"ppl_eval/{eval_dataset_name}"] = ppl_eval
    ppl_train = eval_ppl(model, calibration_data, calibration_data[0].shape[-1], device="cuda")#compute_perplexity(model, calibration_data)
    print(f"ppl_train: {ppl_train:.2f}")
    log_dict["ppl_train"] = ppl_train
    if args["log_wandb"]:
        wandb.log(log_dict)

######################################################################################################################################
# RLPress
device = "cuda"
N = 19
S = 3
class Environment:
    def __init__(self, model_name:str, num_samples:int, sequence_length:int, target_sparsity:float=0.5)->None:
        self.model_name = model_name
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.target_sparsity = target_sparsity
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.possible_sparsities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # initialize state
        self.load_calibration_data()
        self.reset()

    def load_calibration_data(self):
        # caliberation data
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        num_tokens = self.num_samples * self.sequence_length
        self.calib_data = get_fineweb_edu(num_tokens, self.sequence_length, tokenizer, train=True)
        # self.test_data = get_fineweb_edu(num_tokens, self.sequence_length, tokenizer, train=False)
        _, self.test_data = get_w2_data(self.num_samples, self.sequence_length, tokenizer)

    @torch.no_grad()
    def init(self) -> None:
        # create model, tokenizer, and calibration data.
        # model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.model = model
        self.tokenizer = tokenizer

        # caliberation data
        test_data = self.test_data

        # env attributes
        self.action_mask = torch.ones(N)
        self.layers = model.model.decoder.layers
        self.num_layers = len(self.layers)
        self.current_layer = 0
        self.global_sparsity = 0.0
        self.layer_sparsities = [0.0] * self.num_layers
        self.pruning_info = {}

        # buffers
        self.inps = torch.zeros((self.num_samples, self.sequence_length, model.config.hidden_size), dtype=torch.float16, device=self.device)
        self.outs = torch.zeros_like(self.inps)
        self.inp_kwargs = {}

        # obtain input into the first decoder layer
        cache = model.config.use_cache
        model.config.use_cache = False
        inps = self.inps
        inp_kwargs = self.inp_kwargs
        class catch_inps(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
                self.num_inps = 0
            def forward(self, inp, **kwargs):
                nonlocal inps, inp_kwargs
                inps[self.num_inps] = inp
                inp_kwargs.update(kwargs)
                self.num_inps += 1
                raise Exception("caught inps. Stopping forward pass.")
        self.layers[0] = catch_inps(self.layers[0])
        for sample in self.calib_data:
            try:
                model(sample.to(self.device))
            except Exception as e:
                pass
        self.layers[0] = self.layers[0].module
        self.inps = inps
        self.inp_kwargs = inp_kwargs

        # save the log targets to a file for computing the KL divergence later
        # i_batches = 0
        # os.makedirs(f"logs/kl/{self.model_name}", exist_ok=True)
        # batch_size = 4
        # log_probs = []
        # for j in range(self.num_samples):
        #     if os.path.exists(f"logs/kl/{self.model_name}/log_targets_{(j//batch_size)}_{batch_size}.pt"):
        #         i_batches = j // batch_size
        #         continue
        #     sample = test_data[j]
        #     logits = model(sample.to(self.device)).logits
        #     log_probs.append(F.log_softmax(logits.float(), dim=-1).reshape(-1, model.config.vocab_size).cpu())
        #     if j % batch_size == batch_size-1:
        #         log_probs = torch.cat(log_probs, dim=0).cpu()
        #         torch.save(log_probs, f"logs/kl/{self.model_name}/log_targets_{i_batches}_{batch_size}.pt")
        #         print(f"Saved logs/kl/{self.model_name}/log_targets_{i_batches}_{batch_size}.pt")
        #         log_probs = []
        #     elif j == self.num_samples - 1 and len(log_probs) > 0:
        #         log_probs = torch.cat(log_probs, dim=0).cpu()
        #         torch.save(log_probs, f"logs/kl/{self.model_name}/log_targets_{i_batches}_{batch_size}.pt")
        #         print(f"Saved logs/kl/{self.model_name}/log_targets_{i_batches}_{batch_size}.pt")
        #     i_batches = j // batch_size
            
        # # create a dataloader for computing KL divergence later
        # model_name = self.model_name
        # class KLDataset(torch.utils.data.Dataset):
        #     def __init__(self):
        #         self.path_format = f"logs/kl/{model_name}"+"/log_targets_{}_{}.pt"
        #     def __len__(self):
        #         return i_batches + 1
        #     def __getitem__(self, idx):
        #         nonlocal batch_size
        #         samples = torch.cat(test_data[idx*batch_size:(idx+1)*batch_size], dim=0)
        #         log_probs = torch.load(self.path_format.format(idx, batch_size))
        #         return samples, log_probs
        # self.kl_dataloader = torch.utils.data.DataLoader(KLDataset(), batch_size=1, shuffle=False)
        # print(f"KL dataloader with {len(self.kl_dataloader)} batches created.")
        model.config.use_cache = cache

    def prune_layer(self, layer_idx:int, sparsity:float)->None:
        if layer_idx in self.pruning_info:
            raise Exception(f"Layer {layer_idx} already pruned. Skipping.")
        
        layer = self.layers[layer_idx]
        sublayers = {name: module for name, module in layer.named_modules() if isinstance(module, nn.Linear)}
        wrapped_layers = {}
        for name, sublayer in sublayers.items():
            wrapped_layers[name] = WrappedGPT(sublayer)

        # obtain the input activations to each sublayer, computing the feature-wise norms
        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in wrapped_layers:
            handles.append(sublayers[name].register_forward_hook(add_batch(name)))
        for j in range(self.num_samples):
            self.outs[j] = layer(self.inps[j].unsqueeze(0), **self.inp_kwargs)[0]
        for h in handles:
            h.remove()
        
        for name in sublayers:
            wrapped_layers[name].prune(sparsity, modify=True)
            wrapped_layers[name].clean()

        # outputs after pruning
        for j in range(self.num_samples):
            with torch.no_grad():
                self.outs[j] = layer(self.inps[j].unsqueeze(0), **self.inp_kwargs)[0]

        # the output from this layer should be the input to the next layer
        self.inps, self.outs = self.outs, self.inps

        # done pruning this layer. Prepare some info about this layer's pruning
        obtained_sparsity = np.mean([l.weight.data.eq(0).float().mean().item() for l in sublayers.values()]).item()
        info = {
            "layer": layer_idx,
            "layer_target_sparsity": sparsity,
            "layer_obtained_sparsity": obtained_sparsity,
        }
        self.pruning_info[layer_idx] = info

    def reset(self) -> Dict[str, torch.Tensor]:
        if hasattr(self, "inps"):
            del self.inps, self.outs, self.inp_kwargs
            # del self.kl_dataloader
            del self.model, self.tokenizer
        torch.cuda.empty_cache()
        self.init()
        return self.get_state(), {}

    def get_state(self) -> Dict[str, torch.Tensor]:
        s = [self.global_sparsity, self.target_sparsity, self.current_layer / self.num_layers]
        if self.current_layer == 0:
            mask = [1] * len(self.possible_sparsities)
        else:
            mask = [1 if (sum(self.layer_sparsities[:self.current_layer]) + s) / self.current_layer <= self.target_sparsity else 0 for s in self.possible_sparsities]
        state = {
            "state": torch.tensor(s, dtype=torch.float32),
            "action_mask": torch.tensor(mask, dtype=torch.float32)
        }
        return state

    @torch.no_grad()
    def step(self, action:int)->Tuple[Dict[str, torch.Tensor], float, bool, Dict[str, object]]:
        sparsity = self.possible_sparsities[action]
        self.prune_layer(self.current_layer, sparsity)
        # update global sparsity
        self.layer_sparsities[self.current_layer] = sparsity
        self.current_layer += 1
        self.global_sparsity = np.mean(self.layer_sparsities[:self.current_layer])
        # compute reward
        reward = 0
        done = self.current_layer == self.num_layers
        if done:
            # compute KL divergence between the pruned and unpruned model.
            # the logits have been saved to a file during initialization.
            running_kl = 0.0
            total_logprobs = 0
            # for batch in self.kl_dataloader:
            #     inps, target_log_probs = [batch[0].squeeze(0), batch[1].squeeze(0)]
            #     logits = self.model(inps.to(self.device)).logits.reshape(-1, self.model.config.vocab_size)
            #     log_probs = F.log_softmax(logits.float(), dim=-1)
            #     kl = F.kl_div(log_probs, target_log_probs.to(self.device), reduction="batchmean", log_target=True).item()
            #     running_kl *= (total_logprobs / (total_logprobs + target_log_probs.numel()))
            #     running_kl += (target_log_probs.numel() / (total_logprobs + target_log_probs.numel())) * kl
            #     total_logprobs += target_log_probs.numel()
            #     del target_log_probs, logits, kl
            #     torch.cuda.empty_cache()
            # reward = -running_kl
            ppl = eval_ppl(self.model, self.test_data, self.sequence_length, device=self.device)
            reward = -ppl
            self.ppl = ppl

        return self.get_state(), reward, done, False, {}
    
class Policy(nn.Module):
    def __init__(self, state_size:int, action_size:int, device:str="cuda"):
        super(Policy, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.base = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.head = nn.Linear(32, action_size)
        self.uniform_init()

    def to(self, device:Optional[str]=None):
        if device is None:
            device = self.device
        self.device = device
        return super().to(device)

    def forward(self, state:Dict[str, torch.Tensor]) -> torch.Tensor:
        large_neg = torch.finfo(state.dtype).min
        action_mask = state[:, -self.action_size:]

        x = self.base(state)
        logits = self.head(x)
        logits = torch.where(action_mask.to(self.device) == 1, logits, large_neg)
        
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        return dist

    def uniform_init(self):
        bias = self.head.bias.data.detach().clone()
        bias = torch.ones_like(bias)*(1/self.action_size)
        self.head.bias.data.copy_(bias)

    @torch.no_grad()
    def act(self, state:Dict[str, torch.Tensor], deterministic=False) -> tuple[torch.Tensor, torch.Tensor]:
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        dist = self(state)
        action = dist.sample() if not deterministic else dist.mode
        log_prob = dist.log_prob(action)
        return action, log_prob


class Value(nn.Module):
    def __init__(self, state_size:int, device:str):
        super(Value, self).__init__()
        self.state_size = state_size
        self.model = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.device = device

    def to(self, device:Optional[str]=None):
        if device is None:
            device = self.device
        self.device = device
        return super().to(device)

    def forward(self, state:torch.Tensor) -> torch.Tensor:
        return self.model(state)


class PolicyValue:
    def __init__(self, policy_model: nn.Module, value_model: nn.Module):
        self.policy_model = policy_model
        self.value_model = value_model
        self.device = policy_model.device

    def to(self, device:Optional[str]=None):
        if device is None:
            device = self.device
        self.device = device
        self.policy_model.to(device)
        self.value_model.to(device)
        return self

    def get_action_and_value(self, x, action=None):
        dist = self.policy_model(x)
        value = self.value_model(x)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def get_value(self, x):
        return self.value_model(x)
    
    def get_dist(self, x):
        return self.policy_model(x)
    

@torch.no_grad()
def process_trajectory(trajectory, gamma, lam, device):
    lastgaelam = 0
    steps = len(trajectory)
    advantages = torch.zeros(steps).to(device)

    states, actions, rewards, log_probs, values = [], [], [], [], []
    for trans in trajectory:
        state, action, reward, log_prob, value = trans
        states.append(torch.tensor(state).to(device))
        actions.append(torch.tensor(action).to(device))
        rewards.append(torch.tensor(reward).to(device))
        log_probs.append(torch.tensor(log_prob).to(device))
        values.append(torch.tensor(value).to(device))

    values = torch.cat(values)
    states = torch.stack(states, dim=0)
    actions = torch.cat(actions)
    log_probs = torch.cat(log_probs)
    rewards = torch.stack(rewards)

    for t in reversed(range(steps)):
        if t == steps - 1:
            nextnonterminal = 0.0
            nextvalue = 0.0
        else:
            nextnonterminal = 1.0
            nextvalue = values[t+1]
        delta = rewards[t] + gamma * nextvalue * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        
    returns = advantages + values.squeeze()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return states, actions, log_probs, values, returns, advantages


def process_trajectories(trajectories, gamma, lam, device):
    states, actions, log_probs, values, returns, advantages = [], [], [], [], [], []
    for trajectory in trajectories:
        s, a, lp, v, r, adv = process_trajectory(trajectory, gamma, lam, device)
        states.append(s)
        actions.append(a)
        log_probs.append(lp)
        values.append(v)
        returns.append(r)
        advantages.append(adv)
    states = torch.cat(states, dim=0)
    actions = torch.cat(actions, dim=0)
    log_probs = torch.cat(log_probs, dim=0)
    values = torch.cat(values, dim=0)
    returns = torch.cat(returns, dim=0)
    advantages = torch.cat(advantages, dim=0)
    # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return states, actions, log_probs, values, returns, advantages


class Logger:
    def __init__(self):
        self.step = 0

    def log(self, metrics:Dict[str, float], step:Optional[int]=None):
        raise NotImplementedError
    
    def term(self, *args):
        print(*args)


class WandBLogger(Logger):
    def __init__(self, entity:str="ldfrancis", project_name:str="RLPress"):
        super().__init__()
        wandb.init(project=project_name, entity=entity)

    def log(self, metrics:Dict[str, float], step:Optional[int]=None):
        if step is None:
            step = self.step
            self.step += 1
        wandb.log(metrics, step=step)


class TerminalLogger(Logger):
    def __init__(self):
        super().__init__()

    def log(self, metrics:Dict[str, float], step:Optional[int]=None):
        if step is None:
            step = self.step
            self.step += 1
        print(f"Step {step}:")
        for k, v in metrics.items():
            print(f"\t{k} : {v}")
        print("\n")


class RLLearner:
    def __init__(self, policy_n_value:PolicyValue, gamma: float = 0.99, lam: float = 0.95, lr=1e-4, device: str = "cuda"):
        self.policy_n_value = policy_n_value
        self.gamma = gamma
        self.lam = lam
        self.device = device
        self.policy_optimizer = torch.optim.Adam(self.policy_n_value.policy_model.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.policy_n_value.value_model.parameters(), lr=1e-5)
        self.global_step = 0

    def __call__(self, trajectories, epochs:int=5):
        states, actions, log_probs, values, returns, advantages = process_trajectories(trajectories, self.gamma, lam=0.95, device=self.device)
        max_grad_norm = 0.5
        target_kl = 0.01
        self.global_step += len(states)
       
        bs = 32
        inds = np.arange(0, len(states))
        clip_coef = 0.2
        clipfracs = []
        
        policy_losses = []
        value_losses = []
        entropy_losses = []
        approx_kls = []
        old_approx_kls = []
        grad_steps = 0

        for epoch in range(epochs):
            stop_updates = False
            np.random.shuffle(inds)
            # self.policy_optimizer.zero_grad(); self.value_optimizer.zero_grad()
            for start in range(0, len(states), bs):
                end = min(start+bs, len(states))
                b_inds = inds[start:end]

                x = states[b_inds].to(self.device)
                a = actions[b_inds].to(self.device)

                dist = self.policy_n_value.get_dist(x)
                newlogprob = dist.log_prob(a)
                entropy = dist.entropy()
                newvalue = self.policy_n_value.get_value(x)
                
                logratio = newlogprob - log_probs[b_inds].to(self.device)
                ratio = logratio.exp()

                # Policy loss
                pg_obj1 = advantages[b_inds] * ratio
                pg_obj2 = advantages[b_inds] * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = -torch.min(pg_obj1, pg_obj2).mean()

                # Value loss
                v_loss = 0.5 * ((newvalue - returns[b_inds])**2).mean()

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Combined loss
                loss = pg_loss + 0.1 * entropy_loss

                self.policy_optimizer.zero_grad(); self.value_optimizer.zero_grad()
                loss.backward()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.policy_n_value.value_model.parameters(), max_grad_norm)
                nn.utils.clip_grad_norm_(self.policy_n_value.policy_model.parameters(), max_grad_norm)
                self.policy_optimizer.step(); self.value_optimizer.step()

                # Approx kl
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = (ratio - 1 - logratio).mean()
                    clipfracs += [((ratio > (1 + clip_coef)) | (ratio < (1 - clip_coef))).float().mean().item()]

                grad_steps += 1
                policy_losses += [pg_loss.item()]
                value_losses += [v_loss.item()]
                entropy_losses += [entropy_loss.item()]
                approx_kls += [approx_kl.item()]
                old_approx_kls += [old_approx_kl.item()]

                # if approx_kl > target_kl:
                #     stop_updates = True
                #     break
            # self.policy_optimizer.step(); self.value_optimizer.step()
            if stop_updates:
                break

        learner_results = {
            "learner/losses/policy_loss": np.mean(policy_losses) if policy_losses else 0,
            "learner/losses/value_loss": np.mean(value_losses) if value_losses else 0,
            "learner/losses/entropy_loss": np.mean(entropy_losses) if entropy_losses else 0,
            "learner/losses/approx_kls": np.mean(approx_kls) if approx_kls else 0,
            "learner/losses/old_approx_kls": np.mean(old_approx_kls) if old_approx_kls else 0,
            "learner/losses/clipfrac": np.mean(clipfracs),
            "global_step": self.global_step,
        }

        return learner_results
        

class PolicyValueRollout:
    def __init__(self, env: Environment, policy_n_value: PolicyValue):
        self.env = env
        self.policy_n_value = policy_n_value
        self.policy_model = policy_n_value.policy_model
        self.value_model = policy_n_value.value_model

    @torch.no_grad()
    def __call__(self, deterministic=False):
        state, _ = self.env.reset()
        done = False
        trajectory = []
        step = 0
        while not done:
            # import pdb; pdb.set_trace()
            state = torch.cat([state["state"], state["action_mask"]], dim=0).float().to(self.policy_n_value.device)
            # state = state["state"].to(self.policy_n_value.device)
            if not deterministic:
                action, log_prob, _, value = self.policy_n_value.get_action_and_value(state.unsqueeze(0))
            else:
                action, log_prob = self.policy_model.act(state, deterministic=True)
                value = self.value_model(state.unsqueeze(0))
            next_state, reward, done, truncated, info = self.env.step(action.item())
            done = done or truncated
            trajectory.append((state, action, reward, log_prob, value))
            state = next_state
        return trajectory


class Trainer:
    def __init__(self, env: Environment, policy_n_value: PolicyValue, gamma: float = 0.99, lam: float = 0.95, lr: float = 1e-4):
        self.env = env
        self.policy_n_value = policy_n_value
        self.learner = RLLearner(policy_n_value, gamma=gamma, lam=lam, lr=lr, device=policy_n_value.device)
        self.rollout = PolicyValueRollout(env, policy_n_value)
        self.best_score = -1e20

    def __call__(self, num_iters:int=100):
        for iter in range(num_iters):
            start_time = time.time()
            trajectories = [self.rollout() for _ in range(10)]
            learner_results = self.learner(trajectories)
            with torch.no_grad():
                trj =  self.rollout(deterministic=True)
                # trj = trajectories[0]
                rew = sum(tran[2] for tran in trj)
                if rew > self.best_score:
                    self.best_score = rew
                    torch.save(self.policy_n_value.policy_model.state_dict(), "best_policy.pt")
                    print(f"New best model saved with score {self.best_score}")
            end_time = time.time()
            loss = learner_results["learner/losses/policy_loss"]
            # self.logger.log({**learner_results, "Score": rew}, step=learner_results["global_step"])
            data = {
                **learner_results,
                "Score": rew,
                "Best Score": self.best_score,
                "Iteration Time": end_time - start_time,
                "Iteration": iter + 1,
                "ppl_eval/W2": self.env.ppl,
                "obtained_sparsity": self.env.global_sparsity,
                # "levels_sum": self.env.levels_sum
            }
            wandb.log(data)
            print(f"Iteration {iter+1}/{num_iters}, Loss: {loss:.4f}, Rew: {rew:.2f}, Global Step: {learner_results['global_step']}, Time: {end_time - start_time:.2f}s")
            del trajectories, trj
            torch.cuda.empty_cache()
            

def rlpress(model, calibration_data, eval_datasets):
    args = {
        "log_wandb": True,
        "seed": 0,
        "dtype": "float16",
        "model_name_or_path": "facebook/opt-125m",
        "tokenizer_name": "facebook/opt-125m",
        "memory_efficient": True,
        "attn_implementation": "flash_attention_2",
        "use_fast_tokenizer": True,
        "calibration_data": "fineweb-edu",
        "calibration_tokens": 128 * 2048,
        "calibration_sequence_length": 2048,
        "eval_datasets": ["wikitext", "c4"],
        "eval_tokens": 128 * 2048,
        "eval_sequence_length": 2048,
        "fitness_fn": "kl",  # "ppl" or "kl"
        "sparse_weights_path": "/home/user/esie/logs/sparse_weights/opt-125m/sparsegpt",
        "generations": 400,
        "offspring": 64,
        "survivors_per_selection": [8, 2, 1],  # number of survivors after each selection step
        "tokens_per_selection": [2048, 16284, 65536],  # number of tokens used to evaluate fitness at each selection step
        "eval_every": 10,  # evaluate perplexity on eval datasets every N generations
        "max_level": 99999,  # maximum absolute sparsity level per layer
        "max_total_deviation": 99999,  # maximum total sparsity deviation (sum of absolute levels)
        "self_competition": True,


    }
    device = "cuda"
    num_layers_in_block = 6
    num_levels = 3
    N = 10 # 2 * num_levels + 1
    S = 3
    # env = Environment(model, eval_datasets[0], num_layers=12, num_layers_in_block=num_layers_in_block, num_levels=num_levels, weights_path=args["sparse_weights_path"])
    env = Environment(model_name=args["model_name_or_path"], num_samples=128, sequence_length=2048, target_sparsity=0.5)

    policy_model = Policy(state_size=S+N, action_size=N, device=device)
    value_model = Value(state_size=S+N, device=device)

    policy_n_value = PolicyValue(policy_model, value_model)
    policy_n_value.to(device)

    if args["log_wandb"]:
        wandb.init(config=args)
    trainer = Trainer(env, policy_n_value, gamma=0.99, lam=0.95, lr=1e-3)
    trainer(400)


model_name = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)


save_dir = f"logs/sparse_weights/{model_name.split('/')[-1]}"
num_samples = 128
sequence_length = 2048
num_tokens = num_samples * sequence_length
calib_data = get_fineweb_edu(num_tokens, sequence_length, tokenizer, train=True)
_, test_data = get_w2_data(num_samples, sequence_length, tokenizer)
eval_datasets = [test_data]

# evopress(model, calib_data, eval_datasets)
rlpress(model, calib_data, eval_datasets)