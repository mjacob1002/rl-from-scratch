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
import copy

from cse599o_basics.transformer import MyTransformerLM
from cse599o_basics.adamw import AdamW
from cse599o_alignment.grpo import (
    compute_group_normalized_reward,
    grpo_microbatch_train_step,
    masked_mean
)
from cse599o_alignment.gradient_clipping import gradient_clipping


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
    """
    Extract the target keyword from a prompt for the keyword-inclusion reward.
    
    The prompts follow the format: "Write a sentence containing the word '<keyword>'."
    This function parses out the keyword between the single quotes.
    
    Args:
        prompt: The input prompt string
        
    Returns:
        The extracted keyword, or empty string if not found
    """
    # Look for pattern: the word '<keyword>'
    import re
    match = re.search(r"the word '(\w+)'", prompt)
    if match:
        print(f"got keyword {match.group(1)}")
        return match.group(1)
    return ""


# ===================== Data container =====================

class Trajectory:
    """
    Stores a single rollout trajectory for text generation.
    
    In GRPO, for each prompt we generate G responses. This class holds:
    - The original prompt
    - All G generated responses (full sequences including prompt tokens)
    - The rewards for each response
    - The log probabilities of the generated tokens (needed for policy gradient)
    - A mask indicating which tokens are part of the response (vs prompt/padding)
    """

    def __init__(
        self,
        prompt: str,  
        responses: torch.Tensor,      # shape: (G, PROMPT_TOK_LEN + SAMPLING_MAX_TOKENS) - full token sequences
        rewards: torch.Tensor,        # shape: (G,) - one reward per response in the group
        log_probs: torch.Tensor,      # shape: (G, SAMPLING_MAX_TOKENS) - log probs of generated tokens
        response_masks: torch.Tensor  # shape: (G, SAMPLING_MAX_TOKENS) - 1 for response tokens, 0 for padding
    ):
        self.prompt = prompt
        self.responses = responses
        self.rewards = rewards
        self.log_probs = log_probs
        self.response_masks = response_masks  # Mask to identify which tokens are actual responses vs padding




# ===================== Base classes (no @ray.remote) =====================

class Generator:
    """
    Base class for text generation using TransformerLM.
    
    The Generator is responsible for:
    1. Taking prompts and generating G responses per prompt (rollouts)
    2. Computing log probabilities for each generated token
    3. Computing rewards for each response
    
    In GRPO, we generate multiple responses per prompt to compute group-relative advantages.
    """

    def __init__(self):
        # Device setup - use GPU if available for faster generation
        self.gen_device = get_device()
        
        # Initialize the transformer language model with the specified architecture
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
        
        # Load pretrained weights if checkpoint path is provided
        if CHECKPOINT_PATH:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.gen_device)
            self.gen_model.load_state_dict(checkpoint['model_state_dict'])
        
        # GPT-2 tokenizer for encoding/decoding text
        self.gen_tokenizer = tiktoken.get_encoding("gpt2")

    def generate_trajectories(self, prompts: List[str]) -> List[Trajectory]:
        """
        Generate G responses for each prompt using TransformerLM.
        
        This is the core rollout generation function. For each prompt:
        1. Tokenize the prompt and replicate it G times (one for each response)
        2. Autoregressively generate SAMPLING_MAX_TOKENS new tokens
        3. Collect log probabilities for computing policy gradients later
        4. Compute rewards for each generated response
        
        Args:
            prompts: List of text prompts to generate responses for
            
        Returns:
            List of Trajectory objects, one per prompt
        """
        global G
        trajectories = []
        
        # Put model in eval mode for generation (disables dropout, etc.)
        self.gen_model.eval()
        
        with torch.no_grad():  # No gradients needed during generation
            for prompt in prompts:
                # Tokenize the prompt: convert text to list of token IDs
                tokenized_prompt = self.gen_tokenizer.encode(prompt)
                prompt_length = len(tokenized_prompt)
                
                # Replicate prompt G times to generate G responses in parallel
                # This is more efficient than generating one at a time
                batched_tokenized_prompt = [tokenized_prompt] * G
                batched_tokenized_prompt_tensor = torch.tensor(
                    batched_tokenized_prompt, 
                    device=self.gen_device
                )  # shape: (G, prompt_length)
                
                # Store log probs for each generated token
                list_of_log_probs_per_token = []
                
                # Autoregressive generation: generate one token at a time
                for i in range(SAMPLING_MAX_TOKENS):
                    # Forward pass through the model to get logits for next token
                    # Shape: (G, current_seq_len, vocab_size)
                    prompt_logits = self.gen_model(batched_tokenized_prompt_tensor)
                    
                    # Apply temperature scaling for controlled randomness
                    # Higher temperature = more random, lower = more deterministic
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
                    )
                    
                    # Store log probability of the sampled token (needed for policy gradient)
                    # Shape: (G,)
                    log_prob = distribution.log_prob(outputted_tensor)
                    list_of_log_probs_per_token.append(log_prob)
                
                # Stack log probs: list of (G,) tensors -> (G, SAMPLING_MAX_TOKENS)
                log_probs_stacked = torch.stack(list_of_log_probs_per_token, dim=-1)
                
                # Create response mask: all generated tokens are valid responses
                # Shape: (G, SAMPLING_MAX_TOKENS) - must be boolean for masked_mean
                response_masks = torch.ones(G, SAMPLING_MAX_TOKENS, dtype=torch.bool, device=self.gen_device)
                
                # Compute rewards for this prompt's responses
                keyword = get_keyword(prompt)
                reward_tensor = self.reward_for_trajectory(
                    prompt=prompt, 
                    responses=batched_tokenized_prompt_tensor, 
                    keyword=keyword
                )
                
                # Create Trajectory object with all the data
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
        """
        Compute keyword-inclusion reward for each response.
        
        This is a simple binary reward function:
        - Returns 1 if the response contains the target keyword
        - Returns 0 otherwise
        
        Args:
            prompt: The original prompt text
            responses: Token IDs tensor of shape (G, prompt_length + max_sampling_tokens)
            keyword: The target keyword that should appear in the response
            
        Returns:
            Tensor of shape (G,) with reward for each response
        """
        # Convert token IDs back to text
        responses_list = responses.tolist()
        rewards = []
        
        for response_tokens in responses_list:
            # Decode the full sequence (prompt + response) to text
            response_str = self.gen_tokenizer.decode(response_tokens)
            print(response_str)
            # Remove the prompt prefix to get just the generated response
            response_str = response_str[len(prompt):]
            
            # Binary reward: 1 if keyword is in response, 0 otherwise
            if keyword in response_str:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        
        return torch.tensor(rewards, device=self.gen_device)  # shape: (G,)


class Learner:
    """
    Base learner class for policy gradient updates using TransformerLM.
    
    The Learner is responsible for:
    1. Computing group-relative advantages from rewards
    2. Computing current policy log probabilities 
    3. Performing GRPO policy gradient updates
    
    In GRPO, we use the group mean as a baseline (instead of a learned value function),
    which simplifies training while still reducing variance.
    """
    
    def __init__(self):
        # Device setup
        self.learner_device = get_device()
        
        # Initialize the policy model (same architecture as Generator)
        # In ColocatedWorker, this will be the same model as the generator
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
        
        # Load pretrained weights if checkpoint path is provided
        if CHECKPOINT_PATH:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.learner_device)
            self.learner_model.load_state_dict(checkpoint['model_state_dict'])
        
        # AdamW optimizer - commonly used for transformer training
        self.optimizer = torch.optim.AdamW(self.learner_model.parameters(), lr=LEARNING_RATE)

    def compute_advantages(self, trajectories: List[Trajectory]) -> torch.Tensor:
        """
        Compute group-relative advantages for GRPO.
        
        This is the key insight of GRPO: instead of using a learned value function
        as baseline (like in PPO), we use the mean reward within each group.
        
        Advantage_i = (reward_i - mean(group_rewards)) / std(group_rewards)
        
        This removes the need for a critic network while still providing
        variance reduction through the group baseline.
        
        Args:
            trajectories: List of Trajectory objects, each containing G responses
            
        Returns:
            Tensor of shape (N, G) where N is number of prompts
        """
        # Stack rewards from all trajectories: each traj.rewards has shape (G,)
        # Result shape: (N, G) where N = number of prompts
        group_rewards = torch.stack([traj.rewards for traj in trajectories])
        
        # Compute mean reward for each group (baseline)
        # Shape: (N, 1) - one mean per prompt group
        group_means = group_rewards.mean(dim=1, keepdim=True)
        
        # Center rewards by subtracting group mean
        # This makes some responses "good" (positive) and others "bad" (negative)
        # relative to the group average
        shifted_rewards = group_rewards - group_means
        
        if USE_STD_NORMALIZATION:
            # Normalize by standard deviation for more stable training
            # This ensures advantages are on a consistent scale across groups
            group_stds = group_rewards.std(dim=1, keepdim=True)  # Shape: (N, 1)
            advantages = shifted_rewards / (group_stds + ADVANTAGE_EPS)  # Shape: (N, G)
        else:
            advantages = shifted_rewards
            
        return advantages

    def update_policy(self, trajectories: List[Trajectory]) -> float:
        """
        Perform one GRPO policy update step.
        
        Steps:
        1. Compute group-relative advantages
        2. Forward pass to get current policy log probabilities
        3. Compute GRPO clipped loss (like PPO but with group advantages)
        4. Backpropagate and update weights
        
        Args:
            trajectories: List of Trajectory objects from rollout generation
            
        Returns:
            The loss value as a float
        """
        # Put model in training mode
        self.learner_model.train()
        
        # Zero gradients before accumulation
        self.optimizer.zero_grad()
        
        # Step 1: Compute group-relative advantages
        # Shape: (N, G) where N = num prompts, G = group size
        advantages = self.compute_advantages(trajectories)
        
        total_loss = 0.0
        
        # Process each trajectory (one per prompt)
        for traj_idx, trajectory in enumerate(trajectories):
            # Step 2: Forward pass to get CURRENT policy log probabilities
            # We need fresh log probs because the policy has been updated since generation
            
            # Forward pass through the model
            # Input shape: (G, seq_len), Output shape: (G, seq_len, vocab_size)
            logits = self.learner_model(trajectory.responses)
            
            # Extract logits for the generated tokens (not the prompt)
            # We want logits that predict positions [prompt_len, prompt_len + SAMPLING_MAX_TOKENS)
            # The logits at position i predict token at position i+1
            # So logits[:, -SAMPLING_MAX_TOKENS-1:-1, :] predict the generated tokens
            non_prompt_tokens_logits = logits[:, -SAMPLING_MAX_TOKENS - 1:-1, :]  # (G, SAMPLING_MAX_TOKENS, vocab_size)
            
            # Convert logits to log probabilities
            vocab_log_probs = torch.log_softmax(non_prompt_tokens_logits, dim=-1)  # (G, SAMPLING_MAX_TOKENS, vocab_size)
            
            # Get the generated token IDs to index into log probs
            # Shape: (G, SAMPLING_MAX_TOKENS, 1) for gather operation
            generated_tokens_tensor = trajectory.responses[:, -SAMPLING_MAX_TOKENS:].unsqueeze(-1)
            
            # Gather log probs for the actual generated tokens
            # Shape: (G, SAMPLING_MAX_TOKENS, 1) -> squeeze to (G, SAMPLING_MAX_TOKENS)
            policy_log_probs = torch.gather(vocab_log_probs, dim=2, index=generated_tokens_tensor).squeeze(-1)
            
            # Get advantages for this trajectory's group
            # Shape: (G,) -> (G, 1) for broadcasting with per-token loss
            traj_advantages = advantages[traj_idx].unsqueeze(-1)  # (G, 1)
            
            # Step 3: Compute GRPO loss using the microbatch training step
            # This computes clipped policy gradient loss and calls backward()
            loss, _ = grpo_microbatch_train_step(
                policy_log_probs=policy_log_probs,           # (G, SAMPLING_MAX_TOKENS) - current policy
                response_mask=trajectory.response_masks,     # (G, SAMPLING_MAX_TOKENS) - which tokens to include
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS * len(trajectories),
                loss_type=LOSS_TYPE,
                advantages=traj_advantages,                  # (G, 1) - group-relative advantages
                old_log_probs=trajectory.log_probs,          # (G, SAMPLING_MAX_TOKENS) - from generation time
                cliprange=CLIP_RANGE
            )
            
            total_loss += loss.item()
        
        # Step 4: Apply gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(self.learner_model.parameters(), max_norm=1.0)
        
        # Step 5: Update model weights
        self.optimizer.step()
        
        return total_loss


# ===================== Combined Actor =====================

@ray.remote(num_gpus=1)
class ColocatedWorker(Learner, Generator):
    """
    Combined Generator and Learner in a single Ray actor.
    
    "Colocated" means the generation and learning happen in the same process/GPU.
    This is simpler than disaggregated training where they run separately.
    
    Key design: We use a SINGLE model for both generation and learning.
    This avoids:
    - Memory overhead of two models
    - Need to sync weights between generator and learner
    
    The training loop is synchronous:
    1. Generate rollouts with current policy
    2. Compute advantages and update policy
    3. Repeat (no weight sync needed since they share the same model)
    """
    
    def __init__(self):
        # Initialize Learner first (creates learner_model and optimizer)
        Learner.__init__(self)
        
        # Initialize Generator (creates gen_model and gen_tokenizer)
        # Note: This creates a SEPARATE model from learner_model
        Generator.__init__(self)
        
        # Sync generator weights to match learner weights at initialization
        self._sync_weights()
        
        # Training step counter
        self.step_count = 0
        # Initial model for KL-Divergence
        self.init_model = copy.deepcopy(self.gen_model)
        
        print(f"ColocatedWorker initialized on device: {self.learner_device}")
    
    def _sync_weights(self):
        """
        Synchronize weights from learner_model to gen_model.
        
        In the colocated setup, we maintain separate model instances but need to
        keep them in sync. After each training update, we copy the learner's
        weights to the generator so it uses the updated policy for rollouts.
        
        This manual sync approach lets us measure the weight synchronization overhead,
        which becomes important when comparing to disaggregated setups.
        """
        # Copy state dict from learner to generator
        self.gen_model.load_state_dict(self.learner_model.state_dict())
    
    def training_step(self, prompts: List[str]) -> Dict[str, Any]:
        """
        Perform one complete training step: generate rollouts + update policy.
        
        This is the main entry point called by the training loop.
        It combines generation and learning in a synchronous manner.
        
        Args:
            prompts: List of text prompts to train on
            
        Returns:
            Dictionary with training statistics including timing info
        """
        # Phase 1: Generate rollouts (G responses per prompt)
        # Uses Generator.generate_trajectories() with self.gen_model
        rollout_start = time.time()
        
        trajectories = self.generate_trajectories(prompts)
        rollout_time = time.time() - rollout_start
        
        # Phase 2: Update policy using GRPO
        # Uses Learner.update_policy() with self.learner_model
        train_start = time.time()
        loss = self.update_policy(trajectories)
        train_time = time.time() - train_start

        # before the weight sync, compute the KL divergence between the updated model weights and the pi_0, pi_i-1
        
        
        # Phase 3: Sync updated weights from learner_model to gen_model
        # This ensures the generator uses the updated policy for the next rollout
        sync_start = time.time()
        self._sync_weights()
        weight_sync_time = time.time() - sync_start
        
        self.step_count += 1
        
        # Compute average reward across all responses
        all_rewards = torch.cat([traj.rewards for traj in trajectories])
        avg_reward = float(all_rewards.mean()) if trajectories else 0.0
        
        return {
            'step': self.step_count,
            'loss': loss,
            'num_trajectories': len(trajectories),
            'avg_reward': avg_reward,
            # Timing statistics
            'rollout_time': rollout_time,
            'train_time': train_time,
            'weight_sync_time': weight_sync_time,
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current training statistics."""
        return {
            'step_count': self.step_count,
            'model_parameters': sum(p.numel() for p in self.learner_model.parameters())
        }


# ===================== Training loop =====================

# Training prompts for keyword-inclusion task
# The model learns to include specific keywords in its responses
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


def run_training(num_steps: int = 10, num_workers: int = 1):
    """
    Run colocated GRPO training with text generation.
    
    This is the main training loop that:
    1. Creates Ray worker actors
    2. Distributes prompts across workers
    3. Runs training steps and collects statistics
    
    Args:
        num_steps: Number of training iterations
        num_workers: Number of parallel worker actors
    """
    print(f"Starting GRPO training with {num_workers} workers for {num_steps} steps")
    print(f"Group size: {G}, Sampling temperature: {SAMPLING_TEMPERATURE}")
    print(f"Max tokens per response: {SAMPLING_MAX_TOKENS}")
    
    # Create Ray worker actors
    # Each worker has its own model and can train independently
    workers = [ColocatedWorker.remote() for _ in range(num_workers)]
    print(f"Created {len(workers)} ColocatedWorker actors")
    
    # Training loop
    for step in range(num_steps):
        step_start_time = time.time()
        
        # Distribute prompts across workers
        # Each worker gets a subset of prompts to train on
        prompts_per_worker = len(TRAINING_PROMPTS) // num_workers
        if prompts_per_worker == 0:
            prompts_per_worker = 1
        
        # Launch training steps on all workers in parallel
        # ray.get() blocks until all workers complete
        futures = []
        for i, worker in enumerate(workers):
            # Select prompts for this worker (round-robin style)
            start_idx = (i * prompts_per_worker) % len(TRAINING_PROMPTS)
            worker_prompts = TRAINING_PROMPTS[start_idx:start_idx + prompts_per_worker]
            if not worker_prompts:
                worker_prompts = [TRAINING_PROMPTS[i % len(TRAINING_PROMPTS)]]
            
            # Submit training step asynchronously
            futures.append(worker.training_step.remote(worker_prompts))
        
        # Wait for all workers to complete and collect results
        results = ray.get(futures)
        
        step_time = time.time() - step_start_time
        
        # Aggregate statistics across workers
        total_loss = sum(r['loss'] for r in results)
        avg_reward = np.mean([r['avg_reward'] for r in results])
        total_trajectories = sum(r['num_trajectories'] for r in results)
        
        # Aggregate timing statistics
        avg_rollout_time = np.mean([r['rollout_time'] for r in results])
        avg_train_time = np.mean([r['train_time'] for r in results])
        avg_weight_sync_time = np.mean([r['weight_sync_time'] for r in results])
        
        # Print progress
        print(f"Step {step + 1}/{num_steps}: "
              f"loss={total_loss:.4f}, "
              f"avg_reward={avg_reward:.4f}, "
              f"trajectories={total_trajectories}, "
              f"time={step_time:.2f}s")
        print(f"  Timing: rollout={avg_rollout_time:.2f}s, "
              f"train={avg_train_time:.2f}s, "
              f"weight_sync={avg_weight_sync_time:.2f}s")
    
    # Get final statistics from workers
    print("\n=== Training Complete ===")
    stats_futures = [worker.get_statistics.remote() for worker in workers]
    final_stats = ray.get(stats_futures)
    
    for i, stats in enumerate(final_stats):
        print(f"Worker {i}: {stats}")


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
    
    ray.init(
        runtime_env={
            "excludes": [
                ".git/**",  # git metadata and objects
                ".venv/**",  # virtual environment
                "tests/fixtures/**",  # test fixtures (large model files)
                "*.nsys-rep",  # profiling files
                "*.tar",
                "*.zip",
                "*.gz",  # archives
                "__pycache__/**",  # Python cache
                "*.egg-info/**",  # package info
            ]
        },
        _temp_dir="/homes/iws/mjacob2/raytmp",
        ignore_reinit_error=True,
    )
    
    try:
        run_once(num_steps=args.steps, num_workers=args.workers)
    finally:
        ray.shutdown()
