import torch
from torch import Tensor

from shared import *
from hparams import HParams, CNNResidualBlockHParam
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
    def __init__(self, hParams: HParams):
        super().__init__()

        entrance_n_channel, blocks_hp = hParams.cnn_piano_architecture
        self.entrance = ConvBlock(
            2 * (PIANO_RANGE[1] - PIANO_RANGE[0]), entrance_n_channel, 0,
            hParams.cnn_piano_dropout, 
        )
        current_n_channel = entrance_n_channel
        self.resBlocks = torch.nn.Sequential()
        for i, block_hp in enumerate(blocks_hp):
            resBlock = CNNResidualBlock(
                block_hp, current_n_channel, hParams.cnn_piano_dropout, 
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
            ) for x in hParams.cnn_piano_architecture[1]
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
