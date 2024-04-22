import torch

from shared import *
from hparams import HParams
from music import PIANO_RANGE

class BiDiGRUPianoModel(torch.nn.GRU):
    def __init__(self, hParams: HParams):
        super().__init__(
            input_size = 2 * (PIANO_RANGE[1] - PIANO_RANGE[0]), 
            hidden_size = hParams.gru_piano_hidden_size,
            num_layers = hParams.gru_piano_n_layers,
            batch_first=True, 
            bidirectional=True, 
            dropout=hParams.gru_drop_out, 
        )
