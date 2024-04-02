import torch
from torch import Tensor
from torch.nn.modules.transformer import Transformer

from shared import *

class KeyEventEncoder(torch.nn.Module):
    def __init__(
        self, d_key_event: int, d_model: int, 
        n_layers: int, d_hidden: Optional[int], 
    ):
        super().__init__()
        self.layers = []
        current_dim = d_key_event
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

    def forward(self, src: Tensor, tgt: Tensor, key_padding_mask: Tensor) -> Tensor:
        return super().forward(
            src, tgt, 
            src_key_padding_mask=key_padding_mask, 
            memory_key_padding_mask=key_padding_mask,
        )

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
        self, x_and_mask: Tensor, 
    ):
        device = x_and_mask.device
        batch_size, _, _ = x_and_mask.shape
        x = x_and_mask[:, :, :-1]
        mask = x_and_mask[:, :, -1]
        # print('x', x.shape)
        key_event_embeddings = self.keyEventEncoder.forward(x)
        # print('key_event_embeddings', key_event_embeddings.shape)
        transformer_out = self.transformerPianoModel.forward(
            key_event_embeddings, 
            positionalEncoding(
                N_TOKENS_PER_DATAPOINT, self.transformerPianoModel.d_model, device, 
            ).expand(batch_size, -1, -1), 
            mask, 
        )
        return self.outputProjector.forward(transformer_out).view(
            batch_size, ENCODEC_N_BOOKS, 
            N_TOKENS_PER_DATAPOINT, ENCODEC_N_WORDS_PER_BOOK,
        )
