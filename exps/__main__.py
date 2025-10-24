import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from .evopress import evopress
from .rl import rl_run
from .create_level_db import prune_levels
from .utils import *
parser = argparse.ArgumentParser()


# EvoPress arguments
parser.add_argument("--log_wandb", action="store_true", help="Whether to log to Weights & Biases")
parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
parser.add_argument("--dtype", type=str, default="float16", help="Data type for model weights")
parser.add_argument("--model_name_or_path", type=str, default="facebook/opt-125m", help="Model name or path")
parser.add_argument("--tokenizer_name", type=str, default="facebook/opt-125m", help="Tokenizer name or path")
parser.add_argument("--memory_efficient", action="store_true", help="Whether to use memory efficient attention")
parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", help="Attention implementation to use")
parser.add_argument("--use_fast_tokenizer", action="store_true", help="Whether to use the fast tokenizer")
parser.add_argument("--calibration_data", type=str, default="fineweb-edu", help="Calibration dataset name")
parser.add_argument("--calibration_tokens", type=int, default=128*2048, help="Number of tokens for calibration")
parser.add_argument("--calibration_sequence_length", type=int, default=2048, help="Sequence length for calibration")
parser.add_argument("--eval_datasets", nargs="+", type=str, default=["wikitext", "c4"], help="Evaluation dataset names")
parser.add_argument("--eval_tokens", type=int, default=128*2048, help="Number of tokens for evaluation")
parser.add_argument("--eval_sequence_length", type=int, default=2048, help="Sequence length for evaluation")
parser.add_argument("--fitness_fn", type=str, default="kl", choices=["ppl", "kl"], help="Fitness function to use")
parser.add_argument("--sparse_weights_path", type=str, default="./logs/sparse_weights/opt-125m/sparsegpt", help="Path to sparse weights")
parser.add_argument("--generations", type=int, default=400, help="Number of generations for EvoPress")
parser.add_argument("--offspring", type=int, default=64, help="Number of offspring per generation")
parser.add_argument("--survivors_per_selection", nargs="+", type=int, default=[8, 2, 1], help="Number of survivors after each selection step")
parser.add_argument("--tokens_per_selection", nargs="+", type=int, default=[2048, 16284, 65536], help="Number of tokens used to evaluate fitness at each selection step")
parser.add_argument("--eval_every", type=int, default=10, help="Evaluate perplexity on eval datasets every N generations")
parser.add_argument("--max_level", type=int, default=99999, help="Maximum absolute sparsity level per layer")
parser.add_argument("--max_total_deviation", type=int, default=99999, help="Maximum total sparsity deviation (sum of absolute levels)")
parser.add_argument("--run_evopress", action="store_true", help="Whether to run EvoPress")

# For level database creation
parser.add_argument("--create_level_db", action="store_true", help="Whether to create the level database before running EvoPress")
parser.add_argument("--sparsity", type=float, default=0.5, help="Overall sparsity for level db creation")
parser.add_argument("--weights_diff", type=int, default=20_000, help="Weights difference for level db creation")
parser.add_argument("--num_levels", type=int, default=8, help="Number of sparsity levels for level db creation")

# For RL
parser.add_argument("--rl_run", action="store_true", help="Whether to run the RL-based sparsity level selection")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for RL training")
args = parser.parse_args()

# For Debugging
override_evopress_args = False # True - Use hardcoded evopress_args; False - Use argparse args
evopress_args = {
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
    "eval_datasets": ["wikitext"],
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

if __name__ == "__main__":
    # ensure that level db has been generated before running evopress
    db_dir = f"{args.sparse_weights_path}"
    if not os.path.exists(db_dir) and not args.create_level_db:
        print(f"Level database not found at {db_dir}. Please create the level database before running EvoPress.")
        exit(1)

    args = vars(args)
    if override_evopress_args: # only use argparse args if override_evopress_args is False
        for k, v in evopress_args.items():
            args[k] = v

    # Load model, calibration data, and evaluation datasets
    model = AutoModelForCausalLM.from_pretrained(
        args["model_name_or_path"], 
        dtype=getattr(torch, args["dtype"]), 
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args["tokenizer_name"], use_fast=args["use_fast_tokenizer"])
    num_tokens = args["calibration_tokens"]
    calib_data = get_fineweb_edu(num_tokens, args["calibration_sequence_length"], tokenizer, train=True)
    eval_datasets = []
    num_eval_samples = args["eval_tokens"] // args["eval_sequence_length"]
    sequence_length = args["eval_sequence_length"]
    for name in args["eval_datasets"]:
        if name == "wikitext":
            _, test_data = get_w2_data(num_eval_samples, sequence_length, tokenizer)
            eval_datasets.append(test_data)

    # Run level database creation
    if args["create_level_db"]:
        prune_levels(
            model, 
            calib_data, 
            args["sparsity"], 
            args["weights_diff"], 
            args["num_levels"],
            is_sparsegpt=True, 
            device=torch.device("cuda"), 
            save_dir=db_dir
        )

    # Run EvoPress
    if args["run_evopress"]:
        evopress(model, calib_data, eval_datasets, args)

    model_blocks_attr = "model.decoder.layers"
    modules_in_block = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj", "fc1", "fc2"]
    num_levels = 8
    num_blocks = model.config.num_hidden_layers
    test_data = eval_datasets[0]
    action_size = 2*num_levels + 1
    state_size = 2 + action_size

    # Run RL
    rl_args = {
        "model_blocks_attr": "model.decoder.layers",
        "modules_in_block": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj", "fc1", "fc2"],
        "num_levels": args["num_levels"],
        "num_blocks": model.config.num_hidden_layers,
        "test_data": eval_datasets[0],
        "action_size": 2*args["num_levels"] + 1,
        "state_size": 2 + (2*args["num_levels"] + 1),
        "sparse_weights_path": args["sparse_weights_path"],
        "log_wandb": args["log_wandb"],
        "lr": args["lr"],
    }
    if args["rl_run"]:
        rl_run(model, calib_data, eval_datasets, rl_args)
    

    

    