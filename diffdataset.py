import copy
import random
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, target_emb, guidance_emb):
        """
        Args:
            target_emb (np.array or torch.Tensor): 目标嵌入矩阵，形状[1000, 64]
            guidance_emb (np.array or torch.Tensor): 引导嵌入矩阵，形状[1000, 64]
        """
        if isinstance(target_emb, torch.Tensor) and isinstance(guidance_emb, torch.Tensor):
            self.target = target_emb
            self.guidance = guidance_emb
        else:
            # 将输入转换为PyTorch Tensor
            self.target = torch.tensor(target_emb, dtype=torch.float32)
            self.guidance = torch.tensor(guidance_emb, dtype=torch.float32)

        # 验证形状是否匹配
        assert self.target.shape == self.guidance.shape, "两个嵌入矩阵形状必须相同"

    def __len__(self):
        return self.target.shape[0]  # 返回样本总数1000

    def __getitem__(self, idx):
        return {
            'target': self.target[idx],
            'guidance': self.guidance[idx]
        }