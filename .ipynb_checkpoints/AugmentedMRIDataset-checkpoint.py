import torch
from torch.utils.data import Dataset
import random

class AugmentedMRIDataset(Dataset):
    def __init__(self, inputs, targets):
        """
        inputs: Tensor of shape (N, 2, H, W)
        targets: Tensor of shape (N, 2, H, W)
        """
        self.inputs = torch.from_numpy(inputs).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        img_in = self.inputs[idx]
        img_tgt = self.targets[idx]

        # --- AUGMENTATION LOGIC ---
        
        # 1. Random 90-degree rotations (Fixes the "Transposed Test" failure)
        # k is the number of times to rotate 90 degrees (0, 1, 2, or 3)
        if random.random() > 0.5:
            k = random.randint(1, 3)
            img_in = torch.rot90(img_in, k, dims=[1, 2]) # [1, 2] are H and W
            img_tgt = torch.rot90(img_tgt, k, dims=[1, 2])

        # 2. Random Horizontal Flip
        if random.random() > 0.5:
            img_in = torch.flip(img_in, dims=[2]) # W dimension
            img_tgt = torch.flip(img_tgt, dims=[2])

        # 3. Random Vertical Flip
        if random.random() > 0.5:
            img_in = torch.flip(img_in, dims=[1]) # H dimension
            img_tgt = torch.flip(img_tgt, dims=[1])

        return img_in, img_tgt