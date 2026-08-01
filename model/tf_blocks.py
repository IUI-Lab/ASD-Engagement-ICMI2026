import torch
import torch.nn as nn
import copy


class CustomEncoderLayer(nn.Module):

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, activation: nn.Module = None):
        super().__init__()

        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)


        if activation is None:
            activation = nn.GELU()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            activation,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        

        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                attn_mask: torch.Tensor = None, 
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:

        attn_output, _ = self.mha(q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        

        q = self.norm1(q + self.dropout(attn_output))
        

        ffn_output = self.ffn(q)
        
        q = self.norm2(q + self.dropout(ffn_output))
        
        return q


class CustomQKVTransformerEncoder(nn.Module):

    def __init__(self, encoder_layer: nn.Module, num_layers: int):
        super().__init__()

        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                attn_mask: torch.Tensor = None, 
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:

        output = q
        for layer in self.layers:
            output = layer(output, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return output
