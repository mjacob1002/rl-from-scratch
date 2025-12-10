import torch
import os

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out):
    model_state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    checkpoint = {
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'iteration': iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']

class MyCheckpoint:

    def __init__(self, save_interval: int, save_dir: str):
        self.save_interval = save_interval
        self.save_dir = save_dir

    def save(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, fname: str):
        save_checkpoint(model, optimizer, iteration, os.path.join(self.save_dir, fname))
