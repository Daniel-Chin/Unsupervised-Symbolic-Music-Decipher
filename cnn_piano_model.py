import torch
from torch import Tensor

from shared import *
from hparams import HParams
from music import PIANO_RANGE

class PermuteLayer(torch.nn.Module):
    def forward(
        self, x: Tensor, 
    ) -> Tensor:
        return x.permute(0, 2, 1)

class CNNPianoModel(torch.nn.Module):
    def __init__(self, hParams: HParams):
        super().__init__()
        self.convs = torch.nn.Sequential()
        current_n_channel = 2 * (PIANO_RANGE[1] - PIANO_RANGE[0])
        for radius, n_channel in hParams.cnn_piano_cnn:
            self.convs.append(torch.nn.Conv1d(
                current_n_channel, n_channel, 
                kernel_size=radius * 2 + 1, padding=radius,
            ))
            current_n_channel = n_channel
            self.convs.append(PermuteLayer())
            self.convs.append(torch.nn.LayerNorm([n_channel]))
            self.convs.append(PermuteLayer())
            self.convs.append(torch.nn.ReLU())
        self.fcs = torch.nn.ModuleList()
        self.outProjectors = torch.nn.ModuleList()
        for widths in hParams.cnn_piano_fc:
            fc = torch.nn.Sequential()
            self.fcs.append(fc)
            for width in widths:
                fc.append(torch.nn.Linear(current_n_channel, width))
                current_n_channel = width
                fc.append(torch.nn.ReLU())
            outProjector = torch.nn.Linear(
                current_n_channel, ENCODEC_N_BOOKS * ENCODEC_N_WORDS_PER_BOOK, 
            )
            self.outProjectors.append(outProjector)
        receptive_field = (sum(
            radius for radius, _ in hParams.cnn_piano_architecture
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
        x = self.convs(x)
        x = x.permute(0, 2, 1)
        x = self.outProjector(x)
        x = x.view(batch_size, n_frames, ENCODEC_N_BOOKS, ENCODEC_N_WORDS_PER_BOOK)
        x = x.permute(0, 2, 1, 3)
        return x
