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



class Environment:
    def __init__(self, model, test_data, num_blocks, model_blocks_attr, modules_in_block, num_levels, weights_path=""):
        self.num_blocks = num_blocks
        self.modules_in_block = modules_in_block
        self.num_modules_in_block = len(modules_in_block)
        self.num_levels = num_levels
        self.blocks = None

        temp = model
        for attr_name in model_blocks_attr.split("."):
            temp = getattr(temp, attr_name)
        self.blocks = temp

        self.weights_path = weights_path
        self.model = model
        self.test_data = test_data

        self.module_names = []
        for block_idx in range(self.num_blocks):
            for module_name in modules_in_block:
                full_module_name = f"{model_blocks_attr}.{block_idx}.{module_name}"
                self.module_names.append(full_module_name)
        assert len(self.module_names) == self.num_blocks * self.num_modules_in_block

    def init(self):
        self.cur_block = 0
        self.cur_module_idx = 0
        self.levels_sum = 0
        self.possible_levels = list(range(-self.num_levels, self.num_levels + 1))
        self.model.state = [0] * (self.num_blocks * self.num_modules_in_block)
        self.chosen_levels = [0] * (self.num_blocks * self.num_modules_in_block)

    def get_state(self):
        # ensure action would result in a level sum that can be cancelled out in the remaining modules
        num_modules = self.num_blocks * self.num_modules_in_block
        overall_module_idx = self.cur_block * self.num_modules_in_block + self.cur_module_idx
        num_used_modules = overall_module_idx + 1
        num_remaining_modules = num_modules - num_used_modules
        cur_level_sum = sum(self.chosen_levels[:overall_module_idx])
        masks = [1] * len(self.possible_levels)

        def sign(x):
            return 1 if x >= 0 else -1
        
        for i, possible_next_level in enumerate(self.possible_levels):
            projected_level_sum = cur_level_sum + possible_next_level
            # can the projected level sum be cancelled out in the remaining modules?
            max_negative_sums = -self.num_levels * num_remaining_modules
            max_positive_sums = self.num_levels * num_remaining_modules
            neg_sum = projected_level_sum + max_negative_sums
            pos_sum = projected_level_sum + max_positive_sums
            if (
                (neg_sum == 0 or pos_sum == 0) or 
                (sign(projected_level_sum) != sign(pos_sum)) or 
                (sign(projected_level_sum) != sign(neg_sum))
            ):
                masks[i] = 1
            elif (
                (projected_level_sum == 0 and num_remaining_modules < 2) or
                (sign(projected_level_sum) == sign(neg_sum)) or
                (sign(projected_level_sum) == sign(pos_sum))
            ):
                masks[i] = 0
            
        masks = torch.tensor(masks, dtype=torch.float32)
        return {
            "state": torch.tensor(
                [overall_module_idx/num_modules, cur_level_sum / (num_modules*self.num_levels)], dtype=torch.float32
            ),
            "action_mask": masks,
        }
 
    def reset(self):
        self.init()
        return self.get_state(), {}

    def step(self, action):
        level = self.possible_levels[action]
        self.chosen_levels[self.cur_block * self.num_modules_in_block + self.cur_module_idx] = level
        self.cur_module_idx += 1
        if self.cur_module_idx == self.num_modules_in_block:
            self.cur_block += 1
            self.cur_module_idx = 0
        overall_module_idx = self.cur_block * self.num_modules_in_block + self.cur_module_idx
        done = overall_module_idx >= self.num_blocks * self.num_modules_in_block
        reward = 0
        if done:
            load_modules(self.model, self.module_names, self.chosen_levels, self.weights_path)
            ppl = eval_ppl(self.model, self.test_data, self.test_data[0].shape[-1], device="cuda")
            reward = -ppl 
            if all(v == 0 for v in self.chosen_levels):
                reward *= 2  # discourage uniform model
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
        action_mask = state[:, -self.action_size:]

        x = self.base(state)
        logits = self.head(x)
        large_neg = torch.finfo(logits.dtype).min
        logits = torch.where(action_mask.to(self.device) == 1, logits, large_neg)
        
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        return dist

    def uniform_init(self):
        bias = self.head.bias.data.detach().clone()
        bias = torch.ones_like(bias)*(1/self.action_size)
        with torch.no_grad():
            uniform_idx = (self.action_size + 1)//2
            bias[uniform_idx] += 1
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
    

def load_modules(model: AutoModelForCausalLM, module_names: List[str], new_state: List[int], sparse_weights_path: str):
    assert hasattr(model, "state")
    for module_name, new_level, old_level in zip(module_names, new_state, model.state):
        if new_level != old_level:
            module = model.get_submodule(module_name)
            module.weight.data = torch.load(
                os.path.join(sparse_weights_path, module_name, f"{new_level}.pth"), map_location=module.weight.device
            ).to(module.weight.dtype)
    # Update model state
    model.state = new_state


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
    return states, actions, log_probs, values, returns, advantages

class RLLearner:
    def __init__(self, policy_n_value:PolicyValue, gamma: float = 0.99, lam: float = 0.95, lr=1e-4, device: str = "cuda"):
        self.policy_n_value = policy_n_value
        self.gamma = gamma
        self.lam = lam
        self.device = device
        self.policy_optimizer = torch.optim.Adam(self.policy_n_value.policy_model.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.policy_n_value.value_model.parameters(), lr=lr)
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

        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(torch.cuda.is_available() and "cuda" in str(self.device))
        )

        for epoch in range(epochs):
            stop_updates = False
            np.random.shuffle(inds)
            for start in range(0, len(states), bs):
                
                end = min(start+bs, len(states))
                b_inds = inds[start:end]

                x = states[b_inds].to(self.device)
                a = actions[b_inds].to(self.device)

                with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
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
                    loss = pg_loss + 0.1 * entropy_loss + v_loss

                self.policy_optimizer.zero_grad(); self.value_optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.policy_optimizer)
                self.scaler.unscale_(self.value_optimizer)
                nn.utils.clip_grad_norm_(self.policy_n_value.value_model.parameters(), max_grad_norm)
                nn.utils.clip_grad_norm_(self.policy_n_value.policy_model.parameters(), max_grad_norm)
                self.scaler.step(self.policy_optimizer)
                self.scaler.step(self.value_optimizer)
                self.scaler.update()

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

                if approx_kl > target_kl:
                    stop_updates = True
                    break
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
            state = torch.cat([state["state"], state["action_mask"]], dim=0).float().to(self.policy_n_value.device)
            if not deterministic:
                action, log_prob, _, value = self.policy_n_value.get_action_and_value(state.unsqueeze(0))
            else:
                action, log_prob = self.policy_model.act(state, deterministic=True)
                value = self.value_model(state.unsqueeze(0))
            next_state, reward, done, truncated, _ = self.env.step(action.item())
            done = done or truncated
            trajectory.append((state, action, reward, log_prob, value))
            state = next_state
        return trajectory


class Trainer:
    def __init__(self, env: Environment, policy_n_value: PolicyValue, gamma: float = 0.99, lam: float = 0.95, lr: float = 1e-4, log_wandb: bool = True):
        self.env = env
        self.policy_n_value = policy_n_value
        self.learner = RLLearner(policy_n_value, gamma=gamma, lam=lam, lr=lr, device=policy_n_value.device)
        self.rollout = PolicyValueRollout(env, policy_n_value)
        self.best_score = -1e20
        self.log_wandb = log_wandb

    def __call__(self, num_iters:int=100):
        for iter in range(num_iters):
            start_time = time.time()
            with torch.no_grad():
                trajectories = [self.rollout() for _ in range(10)]
            learner_results = self.learner(trajectories)
            with torch.no_grad():
                trj =  self.rollout(deterministic=True)
                rew = sum(tran[2] for tran in trj)
                if rew > self.best_score:
                    self.best_score = rew
                    torch.save(self.policy_n_value.policy_model.state_dict(), "best_policy.pt")
                    print(f"New best model saved with score {self.best_score}")
            end_time = time.time()
            loss = learner_results["learner/losses/policy_loss"]
            data = {
                **learner_results,
                "Score": rew,
                "Best Score": self.best_score,
                "Iteration Time": end_time - start_time,
                "Iteration": iter + 1,
                "ppl_eval/W2": self.env.ppl,
                "levels_sum": sum(self.env.chosen_levels),
                **{f"chosen_levels/{module_name}": level for module_name, level in zip(self.env.module_names, self.env.chosen_levels)},
            }
            if self.log_wandb: wandb.log(data)
            print(f"Iteration {iter+1}/{num_iters}, Loss: {loss:.4f}, Rew: {rew:.2f}, Global Step: {learner_results['global_step']}, Time: {end_time - start_time:.2f}s")
            del trajectories, trj
            torch.cuda.empty_cache()


def rl_run(model, calibration_data, eval_datasets, args):
    model_blocks_attr = args["model_blocks_attr"]
    modules_in_block = args["modules_in_block"]
    num_levels = args["num_levels"]
    num_blocks = args["num_blocks"]
    test_data = args["test_data"]
    action_size = args["action_size"]
    state_size = 2 + action_size
    device = "cuda"

    env = Environment(model, test_data, num_blocks, model_blocks_attr, modules_in_block, num_levels, weights_path=args["sparse_weights_path"])
    
    policy_model = Policy(state_size=state_size, action_size=action_size, device=device)
    value_model = Value(state_size=state_size, device=device)

    policy_n_value = PolicyValue(policy_model, value_model)
    policy_n_value.to(device)

    if args["log_wandb"]:
        wandb.init(config=args)
    trainer = Trainer(env, policy_n_value, gamma=0.99, lam=0.95, lr=args["lr"], log_wandb=args["log_wandb"])
    trainer(400)
