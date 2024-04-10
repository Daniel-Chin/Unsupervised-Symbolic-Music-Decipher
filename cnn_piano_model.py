import torch
from torch import Tensor

from shared import *
from hparams import HParams
from music import PIANO_RANGE

class CNNPianoModel(torch.nn.Module):
    def __init__(self, hParams: HParams):
        super().__init__()
        self.convs = torch.nn.Sequential()
        current_n_channel = 2 * (PIANO_RANGE[1] - PIANO_RANGE[0])
        for radius, n_channel in hParams.cnn_piano_architecture:
            self.convs.append(torch.nn.Conv1d(
                current_n_channel, n_channel, 
                kernel_size=radius * 2 + 1, padding=radius,
            ))
            current_n_channel = n_channel
            self.convs.append(torch.nn.ReLU())
        self.outProjector = torch.nn.Linear(
            current_n_channel, ENCODEC_N_BOOKS * ENCODEC_N_WORDS_PER_BOOK, 
        )
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