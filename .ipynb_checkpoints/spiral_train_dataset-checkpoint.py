"""
Construct spiral MRI training data.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class spi_train_dataset(Dataset):
    def __init__(self, input_array, target_array):
        # Convert NumPy arrays to PyTorch Tensors immediately
        self.inputs = torch.from_numpy(input_array).float()
        self.targets = torch.from_numpy(target_array).float()

    def __len__(self):
        # Tells the DataLoader how many images are in the set
        return len(self.inputs)

    def __getitem__(self, idx):
        # Grabs one 'tray' of data for the DataLoader to batch
        return self.inputs[idx], self.targets[idx]