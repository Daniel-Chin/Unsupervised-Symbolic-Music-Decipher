import torch
from torch import Tensor
import torch.nn.functional as F

from shared import *
from hparams import HParamsDecipher
from music import PIANO_RANGE
from sample_with_ste_backward import sampleWithSTEBackward

class Interpreter(torch.nn.Module):
    def __init__(self, hParams: HParamsDecipher) -> None:
        super().__init__()

        self.hP = hParams
        PIANO_N_KEYS = PIANO_RANGE[1] - PIANO_RANGE[0]
        self.w = torch.nn.Parameter(torch.randn((
            PIANO_N_KEYS, # n of piano keys
            PIANO_N_KEYS, # n of midi pitches
        ), requires_grad=True))
    
    def forward(self, x: torch.Tensor):
        batch_size, n_pianoroll_channels, n_pitches, n_frames = x.shape
        assert n_pianoroll_channels == 2
        assert n_pitches == PIANO_RANGE[1] - PIANO_RANGE[0]
        assert n_frames == N_FRAMES_PER_DATAPOINT
        x = x.permute(0, 1, 3, 2)
        # (batch_size, n_pianoroll_channels, n_frames, n_pitches)
        simplex = self.w.softmax(dim=0)
        if self.hP.interpreter_sample_not_polyphonic:
            w = sampleWithSTEBackward(simplex.T, n=batch_size)
            # (n_pitches, batch_size, n_keys)
            w = w.unsqueeze(3).unsqueeze(4).permute(1, 3, 4, 2, 0)
            # (batch_size, 1, 1, n_keys, n_pitches)
            x = x.unsqueeze(4)
            # (batch_size, n_pianoroll_channels, n_frames, n_pitches, 1)
            x = w @ x
            # (batch_size, n_pianoroll_channels, n_frames, n_keys, 1)
            x = x.squeeze(4)
        else:
            x = F.linear(x, simplex)
        # (batch_size, n_pianoroll_channels, n_frames, n_keys)
        x = x.permute(0, 1, 3, 2)
        return x
