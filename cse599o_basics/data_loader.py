import torch
from collections.abc import Iterable
import numpy as np
from typing import Optional

class DataLoader:
    
    def __init__(self, x: np.ndarray, batch_size: int, context_length: int, device: Optional[torch.device] = None):
        self.x = x
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device

    def get_batch(self):
        input_batch, next_token_batch = torch.LongTensor(self.batch_size, self.context_length), torch.LongTensor(self.batch_size, self.context_length)
        for i in range(self.batch_size):
            start_idx = np.random.randint(0, len(self.x) - self.context_length)
            x_batch = self.x[start_idx:start_idx+self.context_length]
            y_batch = self.x[start_idx+1:start_idx+self.context_length+1]
            input_batch[i] = torch.LongTensor(x_batch)
            next_token_batch[i] = torch.LongTensor(y_batch)
        input_batch = input_batch.to(self.device)
        next_token_batch = next_token_batch.to(self.device)
        return input_batch, next_token_batch