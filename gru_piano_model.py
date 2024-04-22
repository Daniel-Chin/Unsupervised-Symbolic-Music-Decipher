import torch
from torch import Tensor

from shared import *
from hparams import HParams
from music import PIANO_RANGE

class BiDiGRUPianoModel(torch.nn.Module):
    def __init__(self, hParams: HParams):
        super().__init__()

        self.gru = torch.nn.GRU(
            input_size = 2 * (PIANO_RANGE[1] - PIANO_RANGE[0]), 
            hidden_size = hParams.gru_piano_hidden_size,
            num_layers = hParams.gru_piano_n_layers,
            batch_first=True, 
            bidirectional=True, 
            dropout=hParams.gru_drop_out, 
        )
        self.outProjector = torch.nn.Linear(
            2 * hParams.gru_piano_hidden_size, 
            ENCODEC_N_BOOKS * ENCODEC_N_WORDS_PER_BOOK, 
        )
    
    def forward(self, x: Tensor):
        batch_size, two, piano_range, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(
            batch_size, N_TOKENS_PER_DATAPOINT, 2 * (PIANO_RANGE[1] - PIANO_RANGE[0]), 
        )
        x, _ = self.gru.forward(x)
        x = self.outProjector.forward(x)
        x = x.view(
            batch_size, N_TOKENS_PER_DATAPOINT, ENCODEC_N_BOOKS, ENCODEC_N_WORDS_PER_BOOK, 
        ).permute(0, 2, 1, 3)
        return x
