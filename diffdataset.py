import copy
import random
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, target_emb, guidance_emb):
        if isinstance(target_emb, torch.Tensor) and isinstance(guidance_emb, torch.Tensor):
            self.target = target_emb
            self.guidance = guidance_emb
        else:
            self.target = torch.tensor(target_emb, dtype=torch.float32)
            self.guidance = torch.tensor(guidance_emb, dtype=torch.float32)

        assert self.target.shape == self.guidance.shape, 

    def __len__(self):
        return self.target.shape[0]  

    def __getitem__(self, idx):
        return {
            'target': self.target[idx],
            'guidance': self.guidance[idx]
        }
