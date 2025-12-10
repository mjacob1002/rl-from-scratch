"""
GRPO Skeleton: Disaggregated Asynchronous Training Loop
--------------------------------------------------------
Disaggregated training separates Generator, Scorer, and Learner into different actors.
This allows for overlapping rollout generation with training for better throughput.
"""

import asyncio
import argparse
import ray
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import tiktoken
import time
from typing import List, Dict, Any, Optional
import numpy as np
import re

from cse599o_basics.transformer import MyTransformerLM
from cse599o_basics.adamw import AdamW
from cse599o_alignment.grpo import (
    compute_group_normalized_reward,
    grpo_microbatch_train_step,
    masked_mean
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
LEARNING_RATE: float = 5e-4
SAMPLING_TEMPERATURE: float = 0.8
SAMPLING_MAX_TOKENS: int = 60
ADVANTAGE_EPS: float = 1e-8
LOSS_TYPE: str = "grpo_clip"
USE_STD_NORMALIZATION: bool = True
CLIP_RANGE: float = 0.2
GRADIENT_ACCUMULATION_STEPS: int = 1


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_keyword(prompt: str) -> str:
    """Extract target keyword from prompt like "Write a sentence containing the word 'happy'." """
    match = re.search(r"the word '(\w+)'", prompt)
    return match.group(1) if match else ""


# ===================== Data container =====================

class Trajectory:
    """Stores rollout data for a single prompt with G responses."""
    def __init__(
        self,
        version: int,
        prompt: str,
        responses: torch.Tensor,      # (G, seq_len)
        rewards: torch.Tensor,        # (G,)
        log_probs: torch.Tensor,      # (G, SAMPLING_MAX_TOKENS)
        response_masks: torch.Tensor, # (G, SAMPLING_MAX_TOKENS)
    ):
        self.version = version
        self.prompt = prompt
        self.responses = responses
        self.rewards = rewards
        self.log_probs = log_probs
        self.response_masks = response_masks


# ===================== Actors =====================

@ray.remote
class TrajectoryQueue:
    """Async queue for passing unscored trajectories from Generator to Scorer."""
    def __init__(self):
        self.queue = asyncio.Queue()

    async def put(self, traj: Trajectory):
        await self.queue.put(traj)

    async def get(self, timeout: float = 1.0) -> Optional[Trajectory]:
        try:
            return await asyncio.wait_for(self.queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def size(self) -> int:
        return self.queue.qsize()


@ray.remote
class ReplayBuffer:
    """Stores scored trajectories for sampling by the Learner."""
    def __init__(self, max_size: int = 1000):
        self.data = []
        self.max_size = max_size

    def put(self, traj: Trajectory):
        self.data.append(traj)
        if len(self.data) > self.max_size:
            self.data.pop(0)  # Remove oldest

    def sample(self, k: int) -> List[Trajectory]:
        k = min(k, len(self.data))
        if k == 0:
            return []
        indices = np.random.choice(len(self.data), k, replace=False)
        return [self.data[i] for i in indices]

    def size(self) -> int:
        return len(self.data)


@ray.remote(max_concurrency=2)
class Scorer:
    """Computes rewards for generated responses and stores in replay buffer."""
    def __init__(self, traj_q, replay_buf):
        self.traj_q = traj_q
        self.replay_buf = replay_buf
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.running = False

    def score_trajectory(self, traj: Trajectory) -> Trajectory:
        """Compute keyword-inclusion reward for each response."""
        keyword = get_keyword(traj.prompt)
        responses_list = traj.responses.tolist()
        rewards = []
        
        for response_tokens in responses_list:
            response_str = self.tokenizer.decode(response_tokens)
            response_str = response_str[len(traj.prompt):]
            rewards.append(1.0 if keyword in response_str else 0.0)
        
        traj.rewards = torch.tensor(rewards)
        return traj

    async def run(self):
        """Continuously fetch trajectories, score them, and store in replay buffer."""
        self.running = True
        while self.running:
            traj = await self.traj_q.get.remote()
            if traj is not None:
                scored_traj = self.score_trajectory(traj)
                await self.replay_buf.put.remote(scored_traj)

    def stop(self):
        """Stop the scoring loop."""
        self.running = False


@ray.remote(num_gpus=1)
class Learner:
    """Performs policy updates using GRPO from replay buffer."""
    def __init__(self, replay_buf):
        self.device = get_device()
        self.model = MyTransformerLM(
            vocab_size=VOCAB_SIZE, d_model=D_MODEL, num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS, d_ff=D_FF, theta=THETA,
            max_seq_len=CONTEXT_LENGTH, device=self.device
        )
        if CHECKPOINT_PATH:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)
        self.replay_buf = replay_buf
        self.version = 0

    def compute_advantages(self, trajectories: List[Trajectory]) -> torch.Tensor:
        """Compute group-relative advantages."""
        group_rewards = torch.stack([traj.rewards for traj in trajectories])
        group_means = group_rewards.mean(dim=1, keepdim=True)
        shifted = group_rewards - group_means
        if USE_STD_NORMALIZATION:
            stds = group_rewards.std(dim=1, keepdim=True)
            return shifted / (stds + ADVANTAGE_EPS)
        return shifted

    def step(self, batch_size: int = 4) -> Dict[str, Any]:
        """One GRPO update step from replay buffer."""
        trajectories = ray.get(self.replay_buf.sample.remote(batch_size))
        if not trajectories:
            return {'loss': 0.0, 'version': self.version}
        
        self.model.train()
        self.optimizer.zero_grad()
        advantages = self.compute_advantages(trajectories)
        total_loss = 0.0

        for traj_idx, traj in enumerate(trajectories):
            # Move tensors to device
            responses = traj.responses.to(self.device)
            old_log_probs = traj.log_probs.to(self.device)
            masks = traj.response_masks.to(self.device)
            
            # Forward pass for current policy log probs
            logits = self.model(responses)
            non_prompt_logits = logits[:, -SAMPLING_MAX_TOKENS - 1:-1, :]
            vocab_log_probs = torch.log_softmax(non_prompt_logits, dim=-1)
            gen_tokens = responses[:, -SAMPLING_MAX_TOKENS:].unsqueeze(-1)
            policy_log_probs = torch.gather(vocab_log_probs, dim=2, index=gen_tokens).squeeze(-1)
            
            traj_advantages = advantages[traj_idx].unsqueeze(-1).to(self.device)
            
            loss, _ = grpo_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=masks,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS * len(trajectories),
                loss_type=LOSS_TYPE,
                advantages=traj_advantages,
                old_log_probs=old_log_probs,
                cliprange=CLIP_RANGE
            )
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.version += 1
        
        return {'loss': total_loss, 'version': self.version}

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Return model weights for syncing to Generator."""
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def get_version(self) -> int:
        return self.version


@ray.remote(num_gpus=1)
class Generator:
    """Generates text rollouts using the policy model."""
    def __init__(self, traj_q):
        self.device = get_device()
        self.model = MyTransformerLM(
            vocab_size=VOCAB_SIZE, d_model=D_MODEL, num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS, d_ff=D_FF, theta=THETA,
            max_seq_len=CONTEXT_LENGTH, device=self.device
        )
        if CHECKPOINT_PATH:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.traj_q = traj_q
        self.version = 0

    def generate(self, prompts: List[str]) -> List[Trajectory]:
        """Generate G responses per prompt and send to queue."""
        self.model.eval()
        trajectories = []
        
        with torch.no_grad():
            for prompt in prompts:
                tokens = self.tokenizer.encode(prompt)
                batch = torch.tensor([tokens] * G, device=self.device)
                log_probs_list = []
                
                for _ in range(SAMPLING_MAX_TOKENS):
                    logits = self.model(batch)
                    scaled = logits[:, -1, :] / SAMPLING_TEMPERATURE
                    dist = Categorical(logits=scaled)
                    next_tokens = dist.sample()
                    batch = torch.cat([batch, next_tokens.unsqueeze(-1)], dim=1)
                    log_probs_list.append(dist.log_prob(next_tokens))
                
                log_probs = torch.stack(log_probs_list, dim=-1)
                masks = torch.ones(G, SAMPLING_MAX_TOKENS, dtype=torch.bool, device=self.device)
                
                traj = Trajectory(
                    version=self.version,
                    prompt=prompt,
                    responses=batch.cpu(),
                    rewards=torch.zeros(G),  # Will be filled by Scorer
                    log_probs=log_probs.cpu(),
                    response_masks=masks.cpu()
                )
                trajectories.append(traj)
                ray.get(self.traj_q.put.remote(traj))
        
        return trajectories

    def update_weights(self, weights: Dict[str, torch.Tensor], version: int):
        """Load updated weights from Learner."""
        state_dict = self.model.state_dict()
        for name, param in weights.items():
            state_dict[name] = param.to(self.device)
        self.model.load_state_dict(state_dict)
        self.version = version


# ===================== Training loop =====================

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


def run_training(num_steps: int = 10):
    """
    Run disaggregated GRPO training with staleness=1.
    
    The key insight: overlap rollout N+1 with training on batch N.
    This hides rollout latency behind training compute.
    
    Timeline:
        Step 0: [Rollout 0 (blocking)] -> [Scorer auto-scores] -> [Train] -> [Sync]
        Step 1: [Rollout 1 (async)] | [Train] [Sync] | [Wait for Rollout 1]
        ...
    
    The Scorer runs continuously in the background, automatically processing
    trajectories as they arrive in the queue.
    """
    print(f"Starting disaggregated GRPO training for {num_steps} steps")
    print(f"Group size: {G}, Max tokens: {SAMPLING_MAX_TOKENS}")
    
    # Create actors
    traj_q = TrajectoryQueue.remote()
    replay_buf = ReplayBuffer.remote()
    learner = Learner.remote(replay_buf)
    scorer = Scorer.remote(traj_q, replay_buf)
    generator = Generator.remote(traj_q)
    
    print("Created disaggregated actors")
    
    # Start the Scorer in the background - it will continuously process trajectories
    scorer_task = scorer.run.remote()
    print("Started Scorer background task")
    
    # Step 0: Initial rollout (blocking) - need data to bootstrap training
    print("Generating initial rollout...")
    ray.get(generator.generate.remote(TRAINING_PROMPTS))
    
    # Wait for scorer to process initial trajectories
    time.sleep(0.5)  # Give scorer time to process
    
    for step in range(num_steps):
        step_start = time.time()
        
        # Phase 1: Launch next rollout ASYNC (staleness=1)
        # Generator pushes to traj_q, Scorer auto-processes in background
        rollout_start = time.time()
        next_rollout_future = generator.generate.remote(TRAINING_PROMPTS)
        
        # Phase 2: Train on current replay buffer
        train_start = time.time()
        train_result = ray.get(learner.step.remote(batch_size=len(TRAINING_PROMPTS)))
        train_time = time.time() - train_start
        
        # Phase 3: Sync weights from Learner to Generator
        sync_start = time.time()
        weights = ray.get(learner.get_weights.remote())
        version = ray.get(learner.get_version.remote())
        ray.get(generator.update_weights.remote(weights, version))
        sync_time = time.time() - sync_start
        
        # Phase 4: Wait for next rollout to complete
        ray.get(next_rollout_future)
        rollout_time = time.time() - rollout_start
        
        # Scorer automatically processes trajectories in background
        # Small sleep to ensure scoring completes before next train step
        time.sleep(0.1)
        
        step_time = time.time() - step_start
        
        # Compute stats
        buf_size = ray.get(replay_buf.size.remote())
        
        print(f"Step {step + 1}/{num_steps}: "
              f"loss={train_result['loss']:.4f}, "
              f"version={train_result['version']}, "
              f"buf_size={buf_size}, "
              f"time={step_time:.2f}s")
        print(f"  Timing: rollout={rollout_time:.2f}s, "
              f"train={train_time:.2f}s, "
              f"weight_sync={sync_time:.2f}s")
    
    # Stop the scorer
    ray.get(scorer.stop.remote())
    print("\n=== Training Complete ===")


def run_once(num_steps: int = 10):
    """Entry point for training."""
    run_training(num_steps)


# ===================== Entry point =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    
    #ray.init(ignore_reinit_error=True)
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
        run_once(num_steps=args.steps)
    finally:
        ray.shutdown()
