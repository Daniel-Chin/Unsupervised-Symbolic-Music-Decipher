from typing import *
from functools import lru_cache
from itertools import count

import torch
from torch import Tensor
from torch.nn.modules.transformer import Transformer

class KeyEventEncoder(torch.nn.Module):
    def __init__(self, d_model: int, d_hidden: int, n_layers: int):
        super().__init__()
        self.layers = []
        current_dim = 1 + 1 + 88
        for _ in range(n_layers - 1):
            self.layers.append(torch.nn.Linear(current_dim, d_hidden))
            current_dim = d_hidden
            self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.Linear(current_dim, d_model))
        self.sequential = torch.nn.Sequential(*self.layers)
    
    def forward(self, x: Tensor, /) -> Tensor:
        return self.sequential(x)

class TransformerPianoModel(Transformer):
    def __init__(
        self, d_model: int, nhead: int, 
        num_encoder_layers: int, num_decoder_layers: int, 
        dim_feedforward: int, 
    ):
        super().__init__(
            d_model, nhead, num_encoder_layers, num_decoder_layers, 
            dim_feedforward, 
        )

    def forward(self, src: Tensor, tgt: Tensor, src_lens: List[int]) -> Tensor:
        batch_size, max_n_notes, _ = src.shape
        device = src.device
        ladder = torch.arange(max_n_notes, device=device).expand(batch_size, -1)
        key_padding_mask = ladder >= torch.tensor(
            src_lens, device=device, 
        ).unsqueeze(1)
        return super().forward(
            src, tgt, 
            src_key_padding_mask=key_padding_mask, 
            memory_key_padding_mask=key_padding_mask,
        )

@lru_cache()
def positionalEncoding(max_len: int, d_model: int, device: torch.device) -> Tensor:
    pe = torch.zeros(max_len, d_model, device=device)
    ladder = torch.arange(max_len, dtype=pe.dtype, device=device)
    for i in count():
        try:
            pe[:, 2 * i    ] = torch.sin(ladder / (max_len ** (2 * i / d_model)))
            pe[:, 2 * i + 1] = torch.cos(ladder / (max_len ** (2 * i / d_model)))
        except IndexError:
            break
    return pe
