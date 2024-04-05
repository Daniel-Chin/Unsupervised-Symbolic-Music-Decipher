from functools import lru_cache

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
    
    @lru_cache(maxsize=1)
    def causalMask(self):
        return torch.ones((
            N_TOKENS_PER_DATAPOINT, N_TOKENS_PER_DATAPOINT, 
        )).tril().log()

    def forward(
        self, src: Tensor, tgt: Tensor, key_padding_mask: Tensor, 
        is_causal: bool, 
    ) -> Tensor:
        return super().forward(
            src, tgt, 
            src_key_padding_mask=key_padding_mask, 
            memory_key_padding_mask=key_padding_mask,
            tgt_is_causal=is_causal, 
            tgt_mask=self.causalMask() if is_causal else None, 
        )

class TFPiano(torch.nn.Module):
    def __init__(
        self, keyEventEncoder: KeyEventEncoder, 
        transformerPianoModel: TransformerPianoModel,
        is_decoder_auto_regressive: bool, 
    ) -> None:
        super().__init__()
        self.keyEventEncoder = keyEventEncoder
        self.transformerPianoModel = transformerPianoModel
        self.is_decoder_auto_regressive = is_decoder_auto_regressive
        self.outputProjector = torch.nn.Linear(
            transformerPianoModel.d_model, 
            ENCODEC_N_BOOKS * ENCODEC_N_WORDS_PER_BOOK, 
        )
        if is_decoder_auto_regressive:
            self.sos_emb = torch.nn.Parameter(torch.randn((
                1, 1, transformerPianoModel.d_model, 
            )))
            self.embedding = torch.nn.ModuleList([torch.nn.Embedding(
                ENCODEC_N_WORDS_PER_BOOK, transformerPianoModel.d_model, 
            ) for _ in range(ENCODEC_N_BOOKS)])
        else:
            self.sos_emb = None
            self.embedding = None
    
    def forward(
        self, x: Tensor, mask: Tensor, y_hat_unshifted: Optional[Tensor],
    ):
        # y_hat_unshifted is what's already predicted when trained autoregressively

        device = x.device
        batch_size, _, _ = x.shape
        # print('x', x.shape)
        # print('mask', mask.shape)
        key_event_embeddings = self.keyEventEncoder.forward(x)
        # print('key_event_embeddings', key_event_embeddings.shape)
        positional_encoding = positionalEncoding(
            N_TOKENS_PER_DATAPOINT, self.transformerPianoModel.d_model, device, 
        ).expand(batch_size, -1, -1)
        if self.is_decoder_auto_regressive:
            assert y_hat_unshifted is not None
            assert self.sos_emb is not None
            assert self.embedding is not None

            y_hat_no_tail = y_hat_unshifted[:, :, :-1]
            embs = [
                self.embedding[i].forward(y_hat_no_tail[:, i, :]) 
                for i in range(ENCODEC_N_BOOKS)
            ]
            
            tgt = torch.cat((
                self.sos_emb.expand(batch_size, 1, -1), 
                torch.stack(embs, dim=3).sum(dim=3) / ENCODEC_N_BOOKS ** 0.5, 
            ), dim=1) + positional_encoding
        else:
            tgt = positional_encoding
        transformer_out = self.transformerPianoModel.forward(
            key_event_embeddings, 
            tgt, 
            mask, 
            is_causal=self.is_decoder_auto_regressive,
        )
        return self.outputProjector.forward(transformer_out).view(
            batch_size, ENCODEC_N_BOOKS, 
            N_TOKENS_PER_DATAPOINT, ENCODEC_N_WORDS_PER_BOOK,
        )
    
    def autoRegress(self, x: Tensor, mask: Tensor):
        batch_size = x.shape[0]
        device = x.device
        y_hat = torch.zeros((
            batch_size, ENCODEC_N_BOOKS, N_TOKENS_PER_DATAPOINT, 
        ), device=device)
        for t in range(N_TOKENS_PER_DATAPOINT):
            y_hat[:, :, t] = self.forward(
                x, mask, y_hat, 
            )[:, :, t, :].argmax(dim=-1)
        return y_hat
