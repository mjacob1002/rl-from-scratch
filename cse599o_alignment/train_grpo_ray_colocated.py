"""
GRPO Skeleton: Colocated Synchronous Training Loop (Simplified)
--------------------------------------------------------------
Students should complete the TODO parts to:
 - implement rollout generation with reward computation using TransformerLM
 - perform policy updates using GRPO algorithm
 - implement keyword inclusion reward function
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import tiktoken
import time
from typing import List, Dict, Any, Optional
import numpy as np
import copy

from cse599o_basics.transformer import MyTransformerLM
from cse599o_basics.adamw import AdamW
from cse599o_alignment.grpo import (
    compute_group_normalized_reward,
    grpo_microbatch_train_step,
    masked_mean
)
from cse599o_alignment.gradient_clipping import gradient_clipping


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
CLIP_RANGE: float = 0.2  # PPO-style clipping parameter (epsilon)
GRADIENT_ACCUMULATION_STEPS: int = 1

def kl_divergence(
    policy_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    log_ratio = policy_log_probs - reference_log_probs
    kl = masked_mean(log_ratio, response_mask, dim=None)
    return kl

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"In the get device command: {device}")
    return device


def get_keyword(prompt: str) -> str:
    # Look for pattern: the word '<keyword>'
    import re
    match = re.search(r"the word '(\w+)'", prompt)
    if match:
        print(f"got keyword {match.group(1)}")
        return match.group(1)
    return ""



class Trajectory:
    """
    Stores a single rollout trajectory for text generation that are needed by GRPO
    - The original prompt
    - All G generated responses (full sequences including prompt tokens)
    - The rewards for each response
    - The log probabilities of the generated tokens (needed for policy gradient)
    - A mask indicating which tokens are part of the response (vs prompt/padding)
    """

    def __init__(
        self,
        prompt: str,  
        responses: torch.Tensor,      # (G, PROMPT_TOK_LEN + SAMPLING_MAX_TOKENS)
        rewards: torch.Tensor,        # (G,)
        log_probs: torch.Tensor,      # (G, SAMPLING_MAX_TOKENS)
        response_masks: torch.Tensor  # (G, SAMPLING_MAX_TOKENS)
    ):
        self.prompt = prompt
        self.responses = responses
        self.rewards = rewards
        self.log_probs = log_probs
        self.response_masks = response_masks




class Generator:
    """Base class for text generation using TransformerLM. """

    def __init__(self):
        self.gen_device = get_device()
        
        self.gen_model = MyTransformerLM(
            vocab_size=VOCAB_SIZE, 
            d_model=D_MODEL, 
            num_heads=NUM_HEADS, 
            num_layers=NUM_LAYERS, 
            d_ff=D_FF, 
            theta=THETA, 
            max_seq_len=CONTEXT_LENGTH, 
            device=self.gen_device
        )

        
        if CHECKPOINT_PATH:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.gen_device)
            self.gen_model.load_state_dict(checkpoint['model_state_dict'])
        
        self.gen_tokenizer = tiktoken.get_encoding("gpt2")

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
        self.gen_model.eval()

        with torch.no_grad():
            for prompt in prompts:
                tokenized_prompt = self.gen_tokenizer.encode(prompt)
                prompt_length = len(tokenized_prompt)
                
                batched_tokenized_prompt = [tokenized_prompt] * G
                batched_tokenized_prompt_tensor = torch.tensor(
                    batched_tokenized_prompt, 
                    device=self.gen_device
                )  # (G, prompt_length)
                
                list_of_log_probs_per_token = []
                
                # Autoregressive generation
                for i in range(SAMPLING_MAX_TOKENS):
                    prompt_logits = self.gen_model(batched_tokenized_prompt_tensor)  # (G, seq_len, vocab_size)
                    scaled_logits = prompt_logits[:, -1, :] / SAMPLING_TEMPERATURE
                    
                    # Create categorical distribution over vocabulary
                    distribution = Categorical(logits=scaled_logits)
                    
                    # Sample next token for each sequence in the batch
                    # Shape: (G,) - one token ID per sequence
                    outputted_tensor = distribution.sample()
                    
                    # Append the new token to each sequence
                    # Shape after cat: (G, current_seq_len + 1)
                    batched_tokenized_prompt_tensor = torch.cat(
                        [batched_tokenized_prompt_tensor, outputted_tensor.unsqueeze(-1)], 
                        dim=1
                    )  # (G, seq_len + 1)
                    log_prob = distribution.log_prob(outputted_tensor)  # (G,)
                    list_of_log_probs_per_token.append(log_prob)
                
                # Stack log probs: list of (G,) tensors -> (G, SAMPLING_MAX_TOKENS)
                log_probs_stacked = torch.stack(list_of_log_probs_per_token, dim=-1)
                
                # Create response mask: all generated tokens are valid responses
                # Shape: (G, SAMPLING_MAX_TOKENS) - must be boolean for masked_mean
                response_masks = torch.ones(G, SAMPLING_MAX_TOKENS, dtype=torch.bool, device=self.gen_device)
                
                keyword = get_keyword(prompt)
                reward_tensor = self.reward_for_trajectory(
                    prompt=prompt, 
                    responses=batched_tokenized_prompt_tensor, 
                    keyword=keyword
                )
                
                trajectory = Trajectory(
                    prompt=prompt, 
                    responses=batched_tokenized_prompt_tensor,  # (G, prompt_len + SAMPLING_MAX_TOKENS)
                    rewards=reward_tensor,                      # (G,)
                    log_probs=log_probs_stacked,                 # (G, SAMPLING_MAX_TOKENS)
                    response_masks=response_masks               # (G, SAMPLING_MAX_TOKENS)
                )
                trajectories.append(trajectory)
        
        return trajectories

    def reward_for_trajectory(self, prompt: str, responses: torch.Tensor, keyword: str) -> torch.Tensor:
        responses_list = responses.tolist()
        rewards = []
        
        for response_tokens in responses_list:
            response_str = self.gen_tokenizer.decode(response_tokens)
            print(response_str)
            # Filter out the prompt
            response_str = response_str[len(prompt):]
            
            if keyword in response_str:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        
        return torch.tensor(rewards, device=self.gen_device)  # (G,)


class Learner:
    """Base learner class for policy gradient updates using TransformerLM."""
    def __init__(self):
        self.learner_device = get_device()
        self.learner_model = MyTransformerLM(
            vocab_size=VOCAB_SIZE, 
            d_model=D_MODEL, 
            num_heads=NUM_HEADS, 
            num_layers=NUM_LAYERS, 
            d_ff=D_FF, 
            theta=THETA, 
            max_seq_len=CONTEXT_LENGTH, 
            device=self.learner_device
        )
        
        if CHECKPOINT_PATH:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.learner_device)
            self.learner_model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer = torch.optim.AdamW(self.learner_model.parameters(), lr=LEARNING_RATE)

    def compute_advantages(self, trajectories: List[Trajectory]) -> torch.Tensor:
        """Compute group-relative advantages for GRPO."""
        group_rewards = torch.stack([traj.rewards for traj in trajectories])  # (N, G)
        group_means = group_rewards.mean(dim=1, keepdim=True)  # (N, 1)
        shifted_rewards = group_rewards - group_means
        
        if USE_STD_NORMALIZATION:
            group_stds = group_rewards.std(dim=1, keepdim=True)  # (N, 1)
            advantages = shifted_rewards / (group_stds + ADVANTAGE_EPS)  # (N, G)
        else:
            advantages = shifted_rewards
            
        return advantages

    def update_policy(self, trajectories: List[Trajectory]) -> float:
        """
        Perform one GRPO policy update step.
        
        Steps:
        1. Compute group-relative advantages
        2. Forward pass to get current policy log probs 
        3. Compute GRPO clipped loss
        4. Backpropagate and update weights
        """
        self.learner_model.train()
        
        # Zero gradients before accumulation
        self.optimizer.zero_grad()
        
        # Step 1: Compute group-relative advantages
        # Shape: (N, G) where N = num prompts, G = group size
        advantages = self.compute_advantages(trajectories)
        
        total_loss = 0.0
        # For each trajectory, compute logits for the current model on the trajectory
        for traj_idx, trajectory in enumerate(trajectories):
            logits = self.learner_model(trajectory.responses)  # (G, seq_len, vocab_size)
            non_prompt_tokens_logits = logits[:, -SAMPLING_MAX_TOKENS - 1:-1, :]  # (G, SAMPLING_MAX_TOKENS, vocab_size)
            vocab_log_probs = torch.log_softmax(non_prompt_tokens_logits, dim=-1)  # (G, SAMPLING_MAX_TOKENS, vocab_size)
            generated_tokens_tensor = trajectory.responses[:, -SAMPLING_MAX_TOKENS:].unsqueeze(-1)  # (G, SAMPLING_MAX_TOKENS, 1)
            policy_log_probs = torch.gather(vocab_log_probs, dim=2, index=generated_tokens_tensor).squeeze(-1)  # (G, SAMPLING_MAX_TOKENS)
            traj_advantages = advantages[traj_idx].unsqueeze(-1)  # (G, 1)
            
            loss, _ = grpo_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=trajectory.response_masks,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS * len(trajectories),
                loss_type=LOSS_TYPE,
                advantages=traj_advantages,
                old_log_probs=trajectory.log_probs,
                cliprange=CLIP_RANGE
            )
            
            total_loss += loss.item()
        
        torch.nn.utils.clip_grad_norm_(self.learner_model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return total_loss


class ColocatedWorker(Learner, Generator):
    def __init__(self):
        Learner.__init__(self)
        Generator.__init__(self)  # Creates separate gen_model
        self._sync_weights()
        self.step_count = 0
        self.init_model = copy.deepcopy(self.gen_model)  # For KL-Divergence
        
        print(f"ColocatedWorker initialized on device: {self.learner_device}")
    
    def _sync_weights(self):
        self.gen_model.load_state_dict(self.learner_model.state_dict())
    
    def training_step(self, prompts: List[str], k = 1) -> Dict[str, Any]:
        rollout_start = time.time()
        trajectories_list = []
        # To support the off-policy, generate k batches at a time
        for i in range(k):
            trajectories = self.generate_trajectories(prompts)
            trajectories_list.append(trajectories)
        rollout_time = time.time() - rollout_start
        
        train_start = time.time()
        for i in range(k):
            trajectories = trajectories_list[i]
            loss = self.update_policy(trajectories)
        train_time = time.time() - train_start

        sync_start = time.time()
        self._sync_weights()
        weight_sync_time = time.time() - sync_start
        
        self.step_count += k
        
        all_rewards = torch.cat([traj.rewards for traj in trajectories])
        avg_reward = float(all_rewards.mean()) if trajectories else 0.0
        return {
            'step': self.step_count,
            'loss': loss,
            'num_trajectories': len(trajectories),
            'avg_reward': avg_reward,
            'rollout_time': rollout_time,
            'train_time': train_time,
            'weight_sync_time': weight_sync_time,
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'step_count': self.step_count,
            'model_parameters': sum(p.numel() for p in self.learner_model.parameters())
        }

TRAINING_PROMPTS = [
    "Write a sentence containing the word 'happy'.",
    "Write a sentence containing the word 'mountain'.",
    "Write a sentence containing the word 'ocean'.",
    "Write a sentence containing the word 'journey'.",
    "Write a sentence containing the word 'discovery'.",
    "Write a sentence containing the word 'harmony'.",
    "Write a sentence containing the word 'adventure'.",
    "Write a sentence containing the word 'wisdom'.",
]

def run_training(num_steps: int = 10, k: int = 1):
    print(f"Starting GRPO training for {num_steps} steps")
    print(f"Group size: {G}, Sampling temperature: {SAMPLING_TEMPERATURE}")
    print(f"Max tokens per response: {SAMPLING_MAX_TOKENS}")
    
    worker = ColocatedWorker()
    
    adjusted_num_steps = num_steps // k
    if num_steps % k != 0:
        adjusted_num_steps += 1
    
    for step in range(adjusted_num_steps):
        current_k = k
        if (step == adjusted_num_steps - 1) and (num_steps % k != 0):
            current_k = num_steps % k
        
        step_start_time = time.time()
        result = worker.training_step(TRAINING_PROMPTS, current_k)
        step_time = time.time() - step_start_time
        
        # Print progress; helpful for debugging
        print(f"Step {step + 1}/{adjusted_num_steps}: "
              f"loss={result['loss']:.4f}, "
              f"avg_reward={result['avg_reward']:.4f}, "
              f"trajectories={result['num_trajectories']}, "
              f"time={step_time:.2f}s")
        print(f"  Timing: rollout={result['rollout_time']:.2f}s, "
              f"train={result['train_time']:.2f}s, "
              f"weight_sync={result['weight_sync_time']:.2f}s")
    # Debug info 
    print("\n=== Training Complete ===")
    stats = worker.get_statistics()
    print(f"Final stats: {stats}")


def run_once(num_steps: int = 10):
    """Entry point for training."""
    start_time = time.time()
    run_training(num_steps, k=1)
    end_time = time.time()
    print(f"End to end training time: {end_time - start_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10, 
                       help="Number of training steps")
    args = parser.parse_args()
    
    run_once(num_steps=args.steps)
