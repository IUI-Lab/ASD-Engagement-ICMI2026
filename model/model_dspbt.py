
import torch
import torch.nn as nn
from itertools import combinations

from .tf_blocks import CustomEncoderLayer


class AttentionPooling(nn.Module):
    """Additive attention pooling over the sequence dimension.

    score = w^T tanh(W x), softmax over the sequence axis, weighted sum -> (B, D)
    """

    def __init__(self, d_model: int, d_attn: int = None):
        super().__init__()
        d_attn = d_attn or d_model
        self.proj = nn.Linear(d_model, d_attn)
        self.score = nn.Linear(d_attn, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        attn_logits = self.score(torch.tanh(self.proj(x)))   
        attn_weights = torch.softmax(attn_logits, dim=1)     
        return (attn_weights * x).sum(dim=1)              
