"""
GRPO Skeleton: Disaggregated Asynchronous Training Loop
--------------------------------------------------------
Disaggregated training separates Generator and Learner into different actors.
This allows for overlapping rollout generation with training for better throughput.

This version reuses Generator and Learner base classes from the colocated implementation.
"""

import argparse
import ray
import torch
import time
from typing import List, Dict, Any, Optional
import numpy as np

# Reuse base classes and utilities from colocated implementation
from cse599o_alignment.train_grpo_ray_colocated import (
    Generator,
    Learner,
    Trajectory,
    # Constants
    G, VOCAB_SIZE, CONTEXT_LENGTH, NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, THETA,
    CHECKPOINT_PATH, LEARNING_RATE, SAMPLING_TEMPERATURE, SAMPLING_MAX_TOKENS,
    ADVANTAGE_EPS, LOSS_TYPE, USE_STD_NORMALIZATION, CLIP_RANGE, GRADIENT_ACCUMULATION_STEPS,
    # Helpers
    get_device, get_keyword,
)

from ray.experimental.collective import create_collective_group


RDT = True 

# ===================== Replay Buffer =====================

@ray.remote
class ReplayBuffer:
    """Stores scored trajectories for sampling by the Learner."""
    def __init__(self, max_size: int = 1000):
        self.data = []
        self.max_size = max_size

    def put(self, traj: Trajectory):
        self.data.append(traj)
        if len(self.data) > self.max_size:
            self.data.pop(0)

    def sample(self, k: int) -> List[Trajectory]:
        k = min(k, len(self.data))
        if k == 0:
            return []
        indices = np.random.choice(len(self.data), k, replace=False)
        return [self.data[i] for i in indices]

    def size(self) -> int:
        return len(self.data)


# ===================== Disaggregated Actors =====================

@ray.remote(num_gpus=1)
class DisaggregatedGenerator(Generator):
    """
    Ray actor wrapper around Generator base class.
    
    Adds:
    - Direct push to ReplayBuffer (Generator already computes rewards)
    - Weight sync from Learner via update_weights()
    - Version tracking for staleness
    """
    def __init__(self, replay_buf):
        super().__init__()  # Initializes gen_model, gen_tokenizer from Generator
        self.replay_buf = replay_buf
        self.version = 0

    def generate(self, prompts: List[str]) -> int:
        """
        Generate trajectories and push directly to replay buffer.
        
        Uses parent's generate_trajectories() which already computes rewards.
        """
        # Parent method handles generation + reward computation
        trajectories = self.generate_trajectories(prompts)
        
        # Push to replay buffer (move tensors to CPU for cross-actor transfer)
        for traj in trajectories:
            traj.responses = traj.responses.cpu()
            traj.log_probs = traj.log_probs.cpu()
            traj.rewards = traj.rewards.cpu()
            traj.response_masks = traj.response_masks.cpu()
            self.replay_buf.put.remote(traj)
        
        return len(trajectories)

    def update_weights(self, weights: Dict[str, torch.Tensor], version: int):
        """Load updated weights from Learner."""
        self.gen_model.load_state_dict(weights)
        self.version = version

    def get_version(self) -> int:
        return self.version


@ray.remote(num_gpus=1)
class DisaggregatedLearner(Learner):
    """
    Ray actor wrapper around Learner base class.
    
    Adds:
    - Sampling from ReplayBuffer
    - Weight export via get_weights()
    - Version tracking
    """
    def __init__(self, replay_buf):
        super().__init__()  # Initializes learner_model, optimizer from Learner
        self.replay_buf = replay_buf
        self.version = 0

    def step(self, batch_size: int = 8) -> Dict[str, Any]:
        """
        One training step: sample from buffer and update policy.
        """
        trajectories = ray.get(self.replay_buf.sample.remote(batch_size))
        if not trajectories:
            return {'loss': 0.0, 'version': self.version}
        
        # Move tensors to GPU for training
        for traj in trajectories:
            traj.responses = traj.responses.to(self.learner_device)
            traj.log_probs = traj.log_probs.to(self.learner_device)
            traj.rewards = traj.rewards.to(self.learner_device)
            traj.response_masks = traj.response_masks.to(self.learner_device)
        
        # Use parent's update_policy()
        loss = self.update_policy(trajectories)
        self.version += 1
        
        return {'loss': loss, 'version': self.version}

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Export model weights to CPU for transfer to Generator."""
        return {k: v.cpu() for k, v in self.learner_model.state_dict().items()}
    
    @ray.method(tensor_transport="nccl")
    def get_weights_nccl(self) -> Dict[str, torch.Tensor]:
        return self.learner_model.state_dict()

    def get_version(self) -> int:
        return self.version


# ===================== Training Loop =====================

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
    
    Overlaps rollout N+1 with training on batch N for better throughput.
    """
    print(f"Starting disaggregated GRPO training for {num_steps} steps")
    print(f"Group size: {G}, Max tokens: {SAMPLING_MAX_TOKENS}")
    
    # Create actors
    replay_buf = ReplayBuffer.remote()
    generator = DisaggregatedGenerator.remote(replay_buf)
    learner = DisaggregatedLearner.remote(replay_buf)

    if RDT:
        print(f"Using Ray Direct Transfer on nccl")
        group = create_collective_group([learner, generator], backend="nccl")
    
    print("Created disaggregated actors (Generator, Learner, ReplayBuffer)")
    
    # Initial rollout (blocking) - need data to bootstrap
    print("Generating initial rollout...")
    ray.get(generator.generate.remote(TRAINING_PROMPTS))
    
    for step in range(num_steps):
        step_start = time.time()
        
        # Phase 1: Launch next rollout ASYNC (staleness=1)
        rollout_start = time.time()
        next_rollout_future = generator.generate.remote(TRAINING_PROMPTS)
        
        # Phase 2: Train on current replay buffer
        train_start = time.time()
        train_result = ray.get(learner.step.remote(batch_size=len(TRAINING_PROMPTS)))
        train_time = time.time() - train_start
        
        # Phase 3: Sync weights from Learner to Generator
        sync_start = time.time()
        if RDT:
            weights = learner.get_weights_nccl.remote()
        else:
            weights = learner.get_weights.remote()
        version = learner.get_version.remote()
        generator.update_weights.remote(weights, version)
        sync_time = time.time() - sync_start
        
        # Phase 4: Wait for next rollout to complete
        #ray.get(next_rollout_future)
        ray.get(next_rollout_future)
        rollout_time = time.time() - rollout_start
        
        step_time = time.time() - step_start
        #buf_size = ray.get(replay_buf.size.remote())
        
        print(f"Step {step + 1}/{num_steps}: "
              f"loss={train_result['loss']:.4f}, "
              f"version={train_result['version']}, "
              f"time={step_time:.2f}s")
        print(f"  Timing: rollout={rollout_time:.2f}s, "
              f"train={train_time:.2f}s, "
              f"weight_sync={sync_time:.2f}s")
    
    print("\n=== Training Complete ===")
    ray.timeline("ray_dump.json")


def run_once(num_steps: int = 10):
    """Entry point for training."""
    start_time = time.time()
    run_training(num_steps)
    end_time = time.time()
    print(f"Disaggregated total time: {end_time - start_time}")


# ===================== Entry point =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    ray.init(
        runtime_env={
            "excludes": [
                ".git/**",
                ".venv/**",
                "tests/fixtures/**",
                "*.nsys-rep",
                "*.tar",
                "*.zip",
                "*.gz",
                "__pycache__/**",
                "*.egg-info/**",
            ]
        },
        _temp_dir="/homes/iws/mjacob2/raytmp",
        ignore_reinit_error=True,
    )
    
    try:
        run_once(num_steps=args.steps)
    finally:
        ray.shutdown()

