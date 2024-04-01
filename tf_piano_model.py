from functools import lru_cache
from itertools import count

import torch
from torch import Tensor
from torch.nn.modules.transformer import Transformer
from matplotlib import pyplot as plt

from shared import *

class KeyEventEncoder(torch.nn.Module):
    def __init__(self, d_model: int, n_layers: int, d_hidden: Optional[int]):
        super().__init__()
        self.layers = []
        current_dim = 1 + 1 + 88
        for _ in range(n_layers - 1):
            assert d_hidden is not None
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
            dim_feedforward, batch_first=True, 
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
def positionalEncoding(length: int, d_model: int, device: torch.device) -> Tensor:
    MAX_LEN = 2000
    assert length < MAX_LEN

    pe = torch.zeros(length, d_model, device=device)
    ladder = torch.arange(length, dtype=pe.dtype, device=device)
    for i in count():
        try:
            pe[:, 2 * i    ] = torch.sin(ladder / (MAX_LEN ** (2 * i / d_model)))
            pe[:, 2 * i + 1] = torch.cos(ladder / (MAX_LEN ** (2 * i / d_model)))
        except IndexError:
            break
    return pe

class TFPiano(torch.nn.Module):
    def __init__(
        self, keyEventEncoder: KeyEventEncoder, 
        transformerPianoModel: TransformerPianoModel,
    ) -> None:
        super().__init__()
        self.keyEventEncoder = keyEventEncoder
        self.transformerPianoModel = transformerPianoModel
        self.outputProjector = torch.nn.Linear(
            transformerPianoModel.d_model, 
            ENCODEC_N_BOOKS * ENCODEC_N_WORDS_PER_BOOK, 
        )
    
    def forward(
        self, x: Tensor, x_lens: List[int], 
    ):
        device = x.device
        batch_size, _, _ = x.shape
        # print('x', x.shape)
        key_event_embeddings = self.keyEventEncoder.forward(x)
        # print('key_event_embeddings', key_event_embeddings.shape)
        transformer_out = self.transformerPianoModel.forward(
            key_event_embeddings, 
            positionalEncoding(
                N_TOKENS_PER_DATAPOINT, self.transformerPianoModel.d_model, device, 
            ).expand(batch_size, -1, -1), 
            x_lens,
        )
        return self.outputProjector.forward(transformer_out).view(
            batch_size, ENCODEC_N_BOOKS, 
            N_TOKENS_PER_DATAPOINT, ENCODEC_N_WORDS_PER_BOOK,
        )

def inspectPositionalEncoding():
    pe = positionalEncoding(100, 32, CPU).numpy()
    plt.imshow(pe, aspect='auto', interpolation='nearest')
    plt.colorbar()
    plt.xlabel('embedding dim')
    plt.ylabel('time step')
    plt.show()
    for i in (0, 1, 16, 17, 30, 31):
        plt.plot(pe[:, i])
        plt.title(f'embedding dim {i}')
        plt.show()

if __name__ == '__main__':
    inspectPositionalEncoding()
