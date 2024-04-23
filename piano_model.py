from functools import lru_cache

import torch
from torch import Tensor

from shared import *
from hparams import (
    HParams, PianoArchType, CNNHParam, TransformerHParam, 
    CNNResidualBlockHParam, 
)
from music import PIANO_RANGE

class PermuteLayer(torch.nn.Module):
    def forward(
        self, x: Tensor, 
    ) -> Tensor:
        return x.permute(0, 2, 1)

class ConvBlock(torch.nn.Sequential):
    def __init__(
        self, in_n_channel: int, out_n_channel: int, radius: int, 
        dropout: float, 
    ) -> None:
        super().__init__()

        self.append(torch.nn.Conv1d(
            in_n_channel, out_n_channel, 
            kernel_size=radius * 2 + 1, padding=radius,
        ))
        self.append(PermuteLayer())
        self.append(torch.nn.LayerNorm([out_n_channel]))
        self.append(PermuteLayer())
        if dropout != 0.0:
            self.append(torch.nn.Dropout(dropout))
        self.append(torch.nn.ReLU())

class CNNResidualBlock(torch.nn.Module):
    def __init__(
        self, hParam: CNNResidualBlockHParam, in_n_channel: int, 
        dropout: float, 
        name: Optional[str] = None, 
    ):
        super().__init__()

        self.name = name
        self.sequential = torch.nn.Sequential()
        current_n_channel = in_n_channel
        for radius, n_channel in hParam:
            self.sequential.append(ConvBlock(
                current_n_channel, n_channel, radius, dropout, 
            ))
            current_n_channel = n_channel
        self.out_n_channel = current_n_channel
        assert self.out_n_channel == in_n_channel   # otherwise residual connection is not possible
    
    def forward(
        self, x: Tensor, 
    ) -> Tensor:
        return x + self.sequential(x)

class CNNPianoModel(torch.nn.Module):
    def __init__(self, hParams: HParams, cnn_hp: CNNHParam):
        super().__init__()

        self.entrance = ConvBlock(
            2 * (PIANO_RANGE[1] - PIANO_RANGE[0]), cnn_hp.entrance_n_channel, 
            0, hParams.piano_dropout, 
        )
        current_n_channel = cnn_hp.entrance_n_channel
        self.resBlocks = torch.nn.Sequential()
        for i, block_hp in enumerate(cnn_hp.blocks):
            resBlock = CNNResidualBlock(
                block_hp, current_n_channel, hParams.piano_dropout, 
                name=f'res_{i}', 
            )
            self.resBlocks.append(resBlock)
            current_n_channel = resBlock.out_n_channel
        self.outProjector = torch.nn.Linear(
            current_n_channel, ENCODEC_N_BOOKS * ENCODEC_N_WORDS_PER_BOOK, 
        )
        receptive_field = (sum(
            sum(
                radius for radius, _ in x
            ) for x in cnn_hp.blocks
        ) * 2 + 1) / ENCODEC_FPS
        print(f'{receptive_field = : .2f} sec')
    
    def forward(
        self, x: Tensor, 
    ) -> Tensor:
        batch_size, n_pianoroll_channels, n_pitches, n_frames = x.shape
        assert n_pianoroll_channels == 2
        assert n_pitches == PIANO_RANGE[1] - PIANO_RANGE[0]
        assert n_frames == N_TOKENS_PER_DATAPOINT
        x = x.view(batch_size, n_pianoroll_channels * n_pitches, n_frames)
        x = self.entrance.forward(x)
        x = self.resBlocks.forward(x)
        x = x.permute(0, 2, 1)
        x = self.outProjector.forward(x)
        x = x.view(batch_size, n_frames, ENCODEC_N_BOOKS, ENCODEC_N_WORDS_PER_BOOK)
        x = x.permute(0, 2, 1, 3)
        return x

class TransformerPianoModel(torch.nn.Module):
    def __init__(self, hParams: HParams, tf_hp: TransformerHParam):
        super().__init__()

        self.tf_hp = tf_hp
        
        self.inProjector = torch.nn.Linear(
            2 * (PIANO_RANGE[1] - PIANO_RANGE[0]), tf_hp.d_model, 
        )
        encoder_layer = torch.nn.TransformerEncoderLayer(
            tf_hp.d_model, tf_hp.n_heads, tf_hp.d_feedforward, 
            hParams.piano_dropout, batch_first=True, 
        )
        self.tf = torch.nn.TransformerEncoder(
            encoder_layer, tf_hp.n_layers, 
        )
        self.outProjector = torch.nn.Linear(
            tf_hp.d_model, ENCODEC_N_BOOKS * ENCODEC_N_WORDS_PER_BOOK, 
        )

        if tf_hp.attn_radius is None:
            self.attn_mask = None
        else:
            self.attn_mask = self.attnMask(
                N_TOKENS_PER_DATAPOINT, tf_hp.attn_radius, 
            )
            receptive_field = (tf_hp.attn_radius * tf_hp.n_layers * 2 + 1) / ENCODEC_FPS
            print(f'{receptive_field = : .2f} sec')
    
    def forward(
        self, x: Tensor, 
    ) -> Tensor:
        batch_size, n_pianoroll_channels, n_pitches, n_frames = x.shape
        assert n_pianoroll_channels == 2
        assert n_pitches == PIANO_RANGE[1] - PIANO_RANGE[0]
        assert n_frames == N_TOKENS_PER_DATAPOINT
        device = x.device
        x = x.view(batch_size, n_pianoroll_channels * n_pitches, n_frames)
        x = x.permute(0, 2, 1)
        x = self.inProjector.forward(x)
        x = x + positionalEncoding(n_frames, self.tf_hp.d_model, device=device)
        x = self.tf.forward(x, mask=self.attn_mask)
        x = self.outProjector.forward(x)
        x = x.view(batch_size, n_frames, ENCODEC_N_BOOKS, ENCODEC_N_WORDS_PER_BOOK)
        x = x.permute(0, 2, 1, 3)
        return x

    @lru_cache()
    @staticmethod
    def attnMask(n_tokens: int, radius: int):
        x = torch.ones((n_tokens, n_tokens))
        torch.triu(x, diagonal=-radius, out=x)
        torch.tril(x, diagonal=+radius, out=x)
        return x.log()

def PianoModel(hParams: HParams):
    if hParams.piano_arch_type == PianoArchType.CNN:
        cnn_hp = hParams.piano_arch_hparam
        assert isinstance(cnn_hp, CNNHParam)
        return CNNPianoModel(hParams, cnn_hp)
    elif hParams.piano_arch_type == PianoArchType.Transformer:
        tf_hp = hParams.piano_arch_hparam
        assert isinstance(tf_hp, TransformerHParam)
        return TransformerPianoModel(hParams, tf_hp)
    else:
        raise ValueError(f'unknown arch type: {hParams.piano_arch_type}')
