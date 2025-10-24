# Overview
This directory contains the main experiment scripts for creating level databases and running the EvoPress and RL algorithms on language models.


### Usage
To run experiments, use the `__main__.py` script with appropriate arguments, from the base directory. For example:

To create the level database for the `facebook/opt-125m` model, run:
```bash
python -m exps --model_name_or_path facebook/opt-125m --create_level_db --sparsity 0.5 --weights_diff 20000 --num_levels 8
```

To run EvoPress on the `facebook/opt-125m` model, run:
```bash
python -m exps --model_name_or_path facebook/opt-125m --run_evopress --sparse_weights_path path/to/sparse_weights_db
```

To run RL on the `facebook/opt-125m` model, run:
```bash
python -m exps --model_name_or_path facebook/opt-125m --rl_run --sparse_weights_path path/to/sparse_weights_db
```