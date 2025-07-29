import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
from torch.nn.init import xavier_normal_, constant_, xavier_uniform_


class LearnableScalar(nn.Module):
    def __init__(self, init_val=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(init_val))

    def forward(self, x):
        return self.weight * x

class DNN(nn.Module):
    """
    A deep neural network for the reverse process of latent diffusion.
    """
    def __init__(self, in_dims, out_dims, time_emb_size, guidance_emb_size, time_type="cat", norm=False, act_func='tanh', dropout=0.1):
        super(DNN, self).__init__()
        self.num_heads = 4

        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_emb_dim = time_emb_size
        self.guidance_emb_dim = guidance_emb_size
        self.time_type = time_type
        self.norm = norm
        
        self.time_emb_layer = nn.Linear(self.time_emb_dim, self.guidance_emb_dim)
        self.guidance_emb_layer = nn.Linear(self.guidance_emb_dim, self.guidance_emb_dim)
        self.condition_fuse_layer = nn.Linear(self.guidance_emb_dim, self.guidance_emb_dim)
        self.guidance_proj = nn.Linear(self.guidance_emb_dim*2, self.guidance_emb_dim)
        if act_func == 'tanh':
            self.nonlinear = nn.Tanh()
        elif act_func == 'relu':
            self.nonlinear = nn.ReLU()
        elif act_func == 'sigmoid':
            self.nonlinear = nn.Sigmoid()
        elif act_func == 'leaky_relu':
            self.nonlinear = nn.LeakyReLU()
        else:
            self.nonlinear = nn.ReLU()        # default ReLU
            print("ReLU.")

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0]] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims

        self.in_modules = []
        for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:]):
            self.in_modules.append(nn.Linear(d_in, d_out))
            if act_func == 'tanh':
                self.in_modules.append(nn.Tanh())
            elif act_func == 'relu':
                self.in_modules.append(nn.ReLU())
            elif act_func == 'sigmoid':
                self.in_modules.append(nn.Sigmoid())
            elif act_func == 'leaky_relu':
                self.in_modules.append(nn.LeakyReLU())
            else:
                raise ValueError
        self.in_layers = nn.Sequential(*self.in_modules)

        self.out_modules = []
        for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:]):
            self.out_modules.append(nn.Linear(d_in, d_out))
            if act_func == 'tanh':
                self.out_modules.append(nn.Tanh())
            elif act_func == 'relu':
                self.out_modules.append(nn.ReLU())
            elif act_func == 'sigmoid':
                self.out_modules.append(nn.Sigmoid())
            elif act_func == 'leaky_relu':
                self.out_modules.append(nn.LeakyReLU())
            else:
                raise ValueError
        self.out_modules.pop()
        self.out_layers = nn.Sequential(*self.out_modules)

        self.dropout = nn.Dropout(dropout)

        # attention
        self.condition_fusion = nn.MultiheadAttention(
            embed_dim=self.guidance_emb_dim,
            num_heads=self.num_heads,
            batch_first=True
        )
        # 门控参数
        self.gate = nn.Parameter(torch.tensor(0.1))
        self.att_gate = nn.Parameter(torch.tensor(0.1))

        self.pre_norm = nn.LayerNorm(self.guidance_emb_dim)
        self.post_norm = nn.LayerNorm(self.guidance_emb_dim)

        self.apply(xavier_normal_initialization)


    def forward(self, x, timesteps, guidance_emb):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        time_emb = self.time_emb_layer(time_emb)
        time_emb = self.nonlinear(time_emb)

        guidance_emb = self.guidance_emb_layer(guidance_emb)
        guidance_emb = self.nonlinear(guidance_emb)
        guidance_emb = guidance_emb

        x = self.pre_norm(x)
        x = self.dropout(x)

        query = x.unsqueeze(1)
        cond_emb = torch.stack([time_emb, guidance_emb], dim=1)
        cond_emb = self.condition_fuse_layer(cond_emb)
        cond_emb = self.nonlinear(cond_emb)
        attn_output, _ = self.condition_fusion(query, cond_emb, cond_emb)
        h_fuse = attn_output.squeeze(1)

        h = h_fuse + self.att_gate * h_fuse

        h = self.post_norm(h)

        h = self.in_layers(h)
        h = self.out_layers(h)

        return h

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def xavier_normal_initialization(module):
    r""" using `xavier_normal_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.
    .. _`xavier_normal_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_normal_#torch.nn.init.xavier_normal_
    Examples:
        >>> self.apply(xavier_normal_initialization)
    """
    if isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)         



