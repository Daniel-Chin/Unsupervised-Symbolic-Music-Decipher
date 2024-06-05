from functools import lru_cache
import math

import torch
from torch import Tensor

from performance_net import PerformanceNet

from shared import *
from hparams import (
    PianoArchHParam, PianoOutType, 
    CNNHParam, CNNResidualBlockHParam, TransformerHParam, 
    GRUHParam, PerformanceNetHParam, CNN_LSTM_HParam,
)
from music import PIANO_RANGE

class PianoInnerModel(torch.nn.Module):
    def dimOutput(self) -> int:
        raise NotImplementedError()

class PianoModel(torch.nn.Module):
    def __init__(self, arch_hP: PianoArchHParam, out_type: PianoOutType, dropout: float):
        super().__init__()

        self.out_type = out_type
        if isinstance(arch_hP, CNNHParam):
            self.mainModel = CNNInnerModel(arch_hP, dropout)
            self.need_out_projector = True
        elif isinstance(arch_hP, TransformerHParam):
            self.mainModel = TransformerInnerModel(arch_hP, dropout)
            self.need_out_projector = True
        elif isinstance(arch_hP, GRUHParam):
            self.mainModel = GRUInnerModel(arch_hP, dropout)
            self.need_out_projector = True
        elif isinstance(arch_hP, PerformanceNetHParam):
            self.mainModel = PerformanceNetModel(arch_hP, dropout)
            self.need_out_projector = False
        elif isinstance(arch_hP, CNN_LSTM_HParam):
            self.mainModel = CNN_LSTM_InnerModel(arch_hP, dropout)
            self.need_out_projector = True
        else:
            raise ValueError(f'unknown arch type: {type(arch_hP)}')
        if self.need_out_projector:
            self.outProjector = torch.nn.Linear(
                self.mainModel.dimOutput(), math.prod(out_type.shape),
            )
    
    def forward(self, x: Tensor) -> Tensor:
        batch_size, n_pianoroll_channels, n_pitches, n_frames = x.shape
        assert n_pianoroll_channels == 2
        assert n_pitches == PIANO_RANGE[1] - PIANO_RANGE[0]
        assert n_frames == N_FRAMES_PER_DATAPOINT

        x = self.mainModel.forward(x)
        if self.need_out_projector:
            x = self.outProjector.forward(x)
        if self.hP.out_type == PianoOutType.EncodecTokens:
            x = x.view(batch_size, n_frames, ENCODEC_N_BOOKS, ENCODEC_N_WORDS_PER_BOOK)
            x = x.permute(0, 2, 1, 3)
        if self.hP.out_type == PianoOutType.LogSpectrogram:
            _, _, n_bins = fftTools()
            x = x.view(batch_size, n_frames, n_bins)
            x = x.permute(0, 2, 1)
        if self.hP.out_type == PianoOutType.Score:
            x = x.view(batch_size, n_frames, 2, PIANO_RANGE[1] - PIANO_RANGE[0])
            x = x.permute(0, 2, 3, 1)
        return x

class PermuteLayer(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
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
    
    def forward(self, x: Tensor) -> Tensor:
        return x + self.sequential(x)

class CNNInnerModel(PianoInnerModel):
    def __init__(self, cnn_hp: CNNHParam, dropout: float):
        super().__init__()

        
        self.entrance = ConvBlock(
            2 * (PIANO_RANGE[1] - PIANO_RANGE[0]), cnn_hp.entrance_n_channel, 
            0, dropout, 
        )
        current_n_channel = cnn_hp.entrance_n_channel
        self.resBlocks = torch.nn.Sequential()
        for i, block_hp in enumerate(cnn_hp.blocks):
            resBlock = CNNResidualBlock(
                block_hp, current_n_channel, dropout, 
                name=f'res_{i}', 
            )
            self.resBlocks.append(resBlock)
            current_n_channel = resBlock.out_n_channel
        self.dim_output = current_n_channel
        receptive_field = (sum(
            sum(
                radius for radius, _ in x
            ) for x in cnn_hp.blocks
        ) * 2 + 1) / ENCODEC_FPS
        print(f'{receptive_field = : .2f} sec')
    
    def dimOutput(self) -> int:
        return self.dim_output
    
    def forward(self, x: Tensor) -> Tensor:
        batch_size, n_pianoroll_channels, n_pitches, n_frames = x.shape
        assert n_pianoroll_channels == 2
        assert n_pitches == PIANO_RANGE[1] - PIANO_RANGE[0]
        assert n_frames == N_FRAMES_PER_DATAPOINT
        x = x.view(batch_size, n_pianoroll_channels * n_pitches, n_frames)
        x = self.entrance.forward(x)
        x = self.resBlocks.forward(x)
        x = x.permute(0, 2, 1)
        return x

class TransformerInnerModel(PianoInnerModel):
    def __init__(self, tf_hp: TransformerHParam, dropout: float):
        super().__init__()

        self.tf_hp = tf_hp
        
        self.inProjector = torch.nn.Linear(
            2 * (PIANO_RANGE[1] - PIANO_RANGE[0]), tf_hp.d_model, 
        )
        encoder_layer = torch.nn.TransformerEncoderLayer(
            tf_hp.d_model, tf_hp.n_heads, tf_hp.d_feedforward, 
            dropout, batch_first=True, 
        )
        self.tf = torch.nn.TransformerEncoder(
            encoder_layer, tf_hp.n_layers, 
        )

        if tf_hp.attn_radius is None:
            self.attn_mask = None
        else:
            self.attn_mask = __class__.attnMask(
                N_FRAMES_PER_DATAPOINT, tf_hp.attn_radius, 
            )
            receptive_field = (tf_hp.attn_radius * tf_hp.n_layers * 2 + 1) / ENCODEC_FPS
            print(f'{receptive_field = : .2f} sec')
    
    def dimOutput(self) -> int:
        return self.tf_hp.d_model
    
    def forward(self, x: Tensor) -> Tensor:
        batch_size, n_pianoroll_channels, n_pitches, n_frames = x.shape
        assert n_pianoroll_channels == 2
        assert n_pitches == PIANO_RANGE[1] - PIANO_RANGE[0]
        assert n_frames == N_FRAMES_PER_DATAPOINT
        device = x.device
        if self.attn_mask is not None and self.attn_mask.device != device:
            self.attn_mask = self.attn_mask.to(device)
        x = x.view(batch_size, n_pianoroll_channels * n_pitches, n_frames)
        x = x.permute(0, 2, 1)
        x = self.inProjector.forward(x)
        x = x + positionalEncoding(n_frames, self.tf_hp.d_model, device=device)
        x = self.tf.forward(x, mask=self.attn_mask)
        return x

    @staticmethod
    @lru_cache()
    def attnMask(n_tokens: int, radius: int):
        x = torch.ones((n_tokens, n_tokens))
        torch.triu(x, diagonal=-radius, out=x)
        torch.tril(x, diagonal=+radius, out=x)
        return x.log()

class GRUInnerModel(PianoInnerModel):
    def __init__(self, gru_hp: GRUHParam, dropout: float):
        super().__init__()

        self.gru_hp = gru_hp
        
        self.inProjector = torch.nn.Linear(
            2 * (PIANO_RANGE[1] - PIANO_RANGE[0]), gru_hp.n_hidden, 
        )
        self.gru = torch.nn.GRU(
            gru_hp.n_hidden, gru_hp.n_hidden, gru_hp.n_layers, 
            batch_first=True, dropout=dropout, bidirectional=True, 
        )

    def dimOutput(self) -> int:
        return self.gru_hp.n_hidden * 2 # bidirectional
    
    def forward(self, x: Tensor) -> Tensor:
        batch_size, n_pianoroll_channels, n_pitches, n_frames = x.shape
        assert n_pianoroll_channels == 2
        assert n_pitches == PIANO_RANGE[1] - PIANO_RANGE[0]
        assert n_frames == N_FRAMES_PER_DATAPOINT
        x = x.view(batch_size, n_pianoroll_channels * n_pitches, n_frames)
        x = x.permute(0, 2, 1)
        x = self.inProjector.forward(x)
        x, _ = self.gru.forward(x)
        return x

class PerformanceNetModel(torch.nn.Module):
    def __init__(self, pn_hp: PerformanceNetHParam, dropout: float):
        super().__init__()

        self.pn_hp = pn_hp
        
        self.inProjector = torch.nn.Linear(
            2 * (PIANO_RANGE[1] - PIANO_RANGE[0]), pn_hp.start_channels, 
        )
        self.performanceNet = PerformanceNet(
            pn_hp.depth, pn_hp.start_channels, pn_hp.end_channels, 
        )

    def forward(self, x: Tensor) -> Tensor:
        batch_size, n_pianoroll_channels, n_pitches, n_frames = x.shape
        assert n_pianoroll_channels == 2
        assert n_pitches == PIANO_RANGE[1] - PIANO_RANGE[0]
        assert n_frames == N_FRAMES_PER_DATAPOINT
        x = x.view(batch_size, n_pianoroll_channels * n_pitches, n_frames)
        x = x.permute(0, 2, 1)
        x = self.inProjector.forward(x)
        x = x.permute(0, 2, 1)
        x = self.performanceNet.forward(
            x, 
            torch.zeros_like(x), # onset/offset condition
        )
        x = x.log()
        return x

class CNN_LSTM_InnerModel(PianoInnerModel):
    def __init__(self, cnn_lstm_hp: CNN_LSTM_HParam, dropout: float):
        super().__init__()
        
        self.entrance = ConvBlock(
            2 * (PIANO_RANGE[1] - PIANO_RANGE[0]), cnn_lstm_hp.entrance_n_channel, 
            0, dropout, 
        )
        current_n_channel = cnn_lstm_hp.entrance_n_channel
        self.resBlocks = torch.nn.Sequential()
        for i, block_hp in enumerate(cnn_lstm_hp.blocks):
            resBlock = CNNResidualBlock(
                block_hp, current_n_channel, dropout, 
                name=f'res_{i}', 
            )
            self.resBlocks.append(resBlock)
            current_n_channel = resBlock.out_n_channel
        self.lstm = torch.nn.LSTM(
            current_n_channel, 
            cnn_lstm_hp.lstm_hidden_size, 
            cnn_lstm_hp.lstm_n_layers,
            batch_first=True, dropout=dropout,
            bidirectional=False,
        )
        current_n_channel = cnn_lstm_hp.lstm_hidden_size
        self.last_conv = ConvBlock(
            current_n_channel, 
            cnn_lstm_hp.last_conv_n_channel, 
            cnn_lstm_hp.last_conv_kernel_radius, 
            dropout, 
        )
        current_n_channel = cnn_lstm_hp.last_conv_n_channel
        self.dim_output = current_n_channel
    
    def dimOutput(self) -> int:
        return self.dim_output
    
    def forward(self, x: Tensor) -> Tensor:
        batch_size, n_pianoroll_channels, n_pitches, n_frames = x.shape
        assert n_pianoroll_channels == 2
        assert n_pitches == PIANO_RANGE[1] - PIANO_RANGE[0]
        assert n_frames == N_FRAMES_PER_DATAPOINT
        x = x.view(batch_size, n_pianoroll_channels * n_pitches, n_frames)
        x = self.entrance.forward(x)
        x = self.resBlocks.forward(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm.forward(x)
        x = x.permute(0, 2, 1)
        x = self.last_conv.forward(x)
        x = x.permute(0, 2, 1)
        return x
