# EvoPress
import os
import time
from typing import Dict, List, Optional
import torch
import torch.nn.functional as F
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy
from tqdm import trange

from .utils import *

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



def evopress(model, calibration_data, eval_datasets, args):
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


