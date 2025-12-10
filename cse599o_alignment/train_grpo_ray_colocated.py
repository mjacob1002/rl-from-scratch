"""
GRPO Skeleton: Colocated Synchronous Training Loop (Simplified)
--------------------------------------------------------------
Students should complete the TODO parts to:
 - implement rollout generation with reward computation using TransformerLM
 - perform policy updates using GRPO algorithm
 - implement keyword inclusion reward function

This version combines Generator and Learner into a single actor for simplified
synchronous training without replay buffer, training directly on each trajectory.
"""

import argparse
import asyncio
import ray
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import tiktoken
import time
from typing import List, Dict, Any, Optional
import numpy as np

from cse599o_basics.transformer import MyTransformerLM
from cse599o_basics.optimizer import AdamW
from cse599o_alignment.grpo import (
    compute_group_normalized_reward,
    grpo_microbatch_train_step,
    gradient_clipping
)


# ===================== Basic setup =====================

G = 4  # group size (number of responses per prompt)
VOCAB_SIZE = tiktoken.get_encoding("gpt2").n_vocab
CONTEXT_LENGTH = 256
NUM_LAYERS = 4
D_MODEL = 512
NUM_HEADS = 16
D_FF = 1344
THETA = 10000
CHECKPOINT_PATH = ""
N_GRPO_STEPS: int = 100
LEARNING_RATE: float = 5e-4
GROUP_SIZE: int = 4
SAMPLING_TEMPERATURE: float = 0.8
SAMPLING_MAX_TOKENS: int = 60
ADVANTAGE_EPS: float = 1e-8
LOSS_TYPE: str = "grpo_clip"
USE_STD_NORMALIZATION: bool = True


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== Data container =====================

class Trajectory:
    """Stores a single rollout trajectory for text generation"""

    def __init__(
        self,
        prompt: str,  
        responses: torch.Tensor, # shape: (G, PROMT_TOK_LEN + SAMPLING_MAX_TOKENS)]; represent
        rewards: torch.Tensor,  # shape: [G]
        log_probs: torch.Tensor  # shape: [G, SAMPLING_MAX_TOKENS]
    ):
        self.prompt = prompt
        self.responses = responses
        self.rewards = rewards
        self.log_probs = log_probs
        self.response_masks = response_masks




# ===================== Base classes (no @ray.remote) =====================

class Generator:
    """Base class for text generation using TransformerLM"""

    def __init__(self):
        self.gen_device = get_device()
        self.gen_model = MyTransformerLM(vocab_size=VOCAB_SIZE, d_model=D_MODEL, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, d_ff=D_FF, theta=THETA, max_seq_len=CONTEXT_LENGTH, device=get_device())
        self.gen_model.load_checkpoint(CHECKPOINT_PATH)
        self.gen_tokenizer = tiktoken.get_encoding("gpt2")
        # TODO: Initialize the TransformerLM model
        # self.model = TransformerLM(...)
        # self.model.load_checkpoint(CHECKPOINT_PATH)
        # self.tokenizer = tiktoken.get_encoding("gpt2")

    def generate_trajectories(self, prompts: List[str]) -> List[Trajectory]:
        """
        Generate G responses for each prompt using TransformerLM.

        TODO: Implement this method
        - For each prompt, generate G responses using self.model
        - Calculate log probabilities for generated tokens
        - Return list of Trajectory objects with prompts, responses, log_probs
        """
        global G
        trajectories = []
        for prompt in prompts:
            tokenized_prompt = self.gen_tokenizer.encode(prompt) # list of token ids
            batched_tokenized_prompt = [tokenized_prompt] * G
            batched_tokenized_prompt_tensor = torch.tensor(batched_tokenized_prompt, device=self.gen_device)
            trajectory = Trajectory(prompt)
            list_of_log_probs_per_token = []
            for i in range(SAMPLING_MAX_TOKENS): # prompts aren't too long 
                prompt_logits = self.gen_model(batched_tokenized_prompt_tensor)
                distribution = Categorical(logits=prompt_logits[:,-1, :])
                # (bsize) tensor of the next tokens
                outputted_tensor = distribution.sample()
                batched_tokenized_prompt_tensor = torch.cat([batched_tokenizer_prompt_tensor, outputted_tensor.unsqueeze(-1)], dim=1)
                log_prob = distribution.log_prob(outputted_tensor)
                list_of_log_probs_per_token.append(log_prob)
            # now that I'm done with the generation of the rollouts for the prompt, I should
            log_probs_stacked = torch.cat(list_of_log_probs_per_token, dim=-1)
            trajectory = Trajectory(prompt=prompt, responses=batch_tokenized_prompt_tensor, rewards=None, log_probs=log_probs_stacked)
            trajectories.append(trajectory)
        # after computing all the trajectories, compute the reward on the trajectories
        for trajectory in trajectories:
            keyword = get_keyword(trajectory.prompt) # TODO: write this function
            reward_tensor = reward_for_trajectory(responses=trajectory.responses, keyword=keyword)
            trajectory.rewards = reward_tensor
        return trajectories
    

    def reward_for_trajectory(self, prompt: str, responses: torch.Tensor, keyword: str): # shape: (G, prompt_length + max_sampling_tokens)
        # first, trim out the prompt from the responses
        responses_list = responses.tolist()
        rewards = []
        for trajectory in responses_list:
            response_str = self.tokenizer.decode(trajectory)
            response_str = response_str[len(prompt):]
            if keyword in response_str:
                rewards.append(1)
            else:
                rewards.append(0)
        return torch.Tensor(rewards) # shape: (G)


class Learner:
    """Base learner class for policy gradient updates using TransformerLM."""
    def __init__(self):
        self.learner_device = get_device()
        self.learner_model = MyTransformerLM(vocab_size=VOCAB_SIZE, d_model=D_MODEL, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, d_ff=D_FF, theta=THETA, max_seq_len=CONTEXT_LENGTH, device=get_device())
        self.learner_model.load_checkpoint(CHECKPOINT_PATH)
        self.optimizer = torch.Optim.AdamW(self.model.parameters(, lr=1e-5))

        # TODO: Initialize the same TransformerLM model as Generator
        # self.model = TransformerLM(...)
        # self.model.load_checkpoint(CHECKPOINT_PATH)
        # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
    

    
    def compute_advantages(self, trajectories: List[Trajectory]) -> torch.Tensor:
        """Compute advantages for GRPO."""
        # TODO: Implement GRPO advantage computation
        group_rewards = torch.stack([traj.rewards for traj in trajectories])  # (N, G)
        group_means = group_rewards.mean(dim=1, keepdim=True) #(N,)
        shifted_rewards = group_rewards - group_means
        if USE_STD_NORMALIZATION:
            group_stds = group_rewards.std(dim=1, keepdim=True)  # (N,)
            advantages = shifted_rewards / (group_stds + ADVANTAGE_EPS)  # (N, G)
        else:
            advantages = shifted_rewards
        return advantages
        # This should implement the group-relative advantage computation
        # that's central to GRPO algorithm
        #return torch.zeros(len(trajectories), device=self.device)
    
    def update_policy(self, trajectories: List[Trajectory]) -> float:
        """Perform one policy update step."""
        # TODO: Implement GRPO/PPO policy update
        # 1. Compute advantages
        # 2. Compute policy gradient loss
        # 3. Perform optimizer step
        # 4. Return loss value
        advantages = self.compute_advantages(trajectories)
        # get the log-probs from the forward pass of each trajectory
        log_probs_list =[] 
        for trajectory in trajectories:
            logits = self.gen_model(trajectory.responses) # (G, seq_len, VOCAB_SIZE)
            non_prompt_tokens_logits = logits[:, -SAMPLING_MAX_TOKENS - 1:-1,:]
            vocab_log_probs = torch.log_softmax(non_prompt_tokens_logits, dim=-1)
            generated_tokens_tensor = trajectory.responses[:, -MAX_SAMPLING_TOKENS:].unsqueeze(-1)
            log_probs = torch.gather(vocab_log_probs, dim=2, index=generated_tokens_tensor)
            log_probs_list.append(log_probs)


        loss = grpo_microbatch_train_step()
        loss = torch.tensor(0.0, device=self.device)
        return float(loss.item())


# ===================== Combined Actor =====================

@ray.remote
class ColocatedWorker(Generator, Learner):
    """Combined Generator and Learner in a single Ray actor."""
    def __init__(self):
        Generator.__init__(self)
        Learner.__init__(self)
        self.step_count = 0
    
    def training_step(self, prompts: List[str]) -> Dict[str, Any]:
        """Perform one complete training step: generate rollout + update policy."""
        # Generate trajectories for the batch of prompts
        trajectories = self.generate_trajectories(prompts)
        
        # Update policy using GRPO
        loss = self.update_policy(trajectories)
        
        self.step_count += 1
        
        return {
            'step': self.step_count,
            'loss': loss,
            'num_trajectories': len(trajectories),
            'avg_reward': float(torch.cat([traj.rewards for traj in trajectories]).mean()) if trajectories else 0.0
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current training statistics."""
        return {
            'step_count': self.step_count,
            'model_parameters': sum(p.numel() for p in self.model.parameters()) if hasattr(self, 'model') else 0
        }


# ===================== Training loop =====================

def run_training(num_steps: int = 10, num_workers: int = 1):
    """Run colocated GRPO training with text generation."""
    
    # Create workers  
    
    # TODO: Define training prompts
    
    for step in range(num_steps):
        # TODO: 
        pass
        
    # Get final statistics
    pass


def run_once(num_steps: int = 10, num_workers: int = 1):
    """Entry point for training."""
    run_training(num_steps, num_workers)


# ===================== Entry point =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10, 
                       help="Number of training steps")
    parser.add_argument("--workers", type=int, default=1, 
                       help="Number of colocated workers")
    args = parser.parse_args()
    
    ray.init(ignore_reinit_error=True)
    
    try:
        run_once(num_steps=args.steps, num_workers=args.workers)
    finally:
        ray.shutdown()
