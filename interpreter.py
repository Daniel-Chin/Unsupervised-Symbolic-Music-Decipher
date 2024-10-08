import torch
from torch import Tensor
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from shared import *
from hparams import HParamsDecipher, NoteIsPianoKeyHParam, InterpreterPolicy
from music import PIANO_RANGE
from sample_permutation import samplePermutation
from doubly_stochastic import sinkhornKnopp

class Interpreter(torch.nn.Module):
    def __init__(self, hParams: HParamsDecipher) -> None:
        super().__init__()

        self.hP = hParams
        strategy_hP = hParams.strategy_hparam
        assert isinstance(strategy_hP, NoteIsPianoKeyHParam)
        self.strategy_hP: NoteIsPianoKeyHParam = strategy_hP

        PIANO_N_KEYS = PIANO_RANGE[1] - PIANO_RANGE[0]
        if strategy_hP.init_oracle_w_offset is None:
            w = torch.randn((
                PIANO_N_KEYS, # n of piano keys
                PIANO_N_KEYS, # n of midi pitches
            ))
            if self.hP.project_w_to_doubly_stochastic:
                w = w.softmax(dim=0)
        else:
            o = strategy_hP.init_oracle_w_offset
            w = torch.diag_embed(
                torch.ones((PIANO_N_KEYS, )), offset=o, 
            )[:PIANO_N_KEYS, :PIANO_N_KEYS] * 6.7   # logits yielding prob=90%
        self.w = torch.nn.Parameter(w, requires_grad=True)
        if self.hP.project_w_to_doubly_stochastic:
            self.sinkhornKnopp()
    
    def sinkhornKnopp(self):
        sinkhornKnopp(self.w)
    
    def simplex(self):
        if self.hP.project_w_to_doubly_stochastic:
            return self.w
        else:
            return self.w.softmax(dim=0)
    
    def forward(self, x: torch.Tensor):
        batch_size, n_pianoroll_channels, n_pitches, n_frames = x.shape
        assert n_pianoroll_channels == 2
        assert n_pitches == PIANO_RANGE[1] - PIANO_RANGE[0]
        assert n_frames == N_FRAMES_PER_DATAPOINT
        x = x.permute(0, 1, 3, 2)
        # (batch_size, n_pianoroll_channels, n_frames, n_pitches)
        if self.strategy_hP.interpreter_policy == InterpreterPolicy.Polyphonic:
            x = F.linear(x, self.simplex())
        else:
            if   self.strategy_hP.interpreter_policy == InterpreterPolicy.SamplePermutation:
                w = samplePermutation(self.simplex(), n=batch_size)
            elif self.strategy_hP.interpreter_policy == InterpreterPolicy.SampleSelection:
                w = Categorical(probs=self.simplex()).sample_n(batch_size)
            else:
                assert False, self.strategy_hP.interpreter_policy
            w = w.permute(2, 1, 0)
            # (n_pitches, batch_size, n_keys)
            w = w.unsqueeze(3).unsqueeze(4).permute(1, 3, 4, 2, 0)
            # (batch_size, 1, 1, n_keys, n_pitches)
            x = x.unsqueeze(4)
            # (batch_size, n_pianoroll_channels, n_frames, n_pitches, 1)
            x = w @ x
            # (batch_size, n_pianoroll_channels, n_frames, n_keys, 1)
            x = x.squeeze(4)
        # (batch_size, n_pianoroll_channels, n_frames, n_keys)
        x = x.permute(0, 1, 3, 2)
        return x
