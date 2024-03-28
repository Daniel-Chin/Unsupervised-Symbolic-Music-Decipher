from typing import *
import json
from os import path

import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from music import PIANO_RANGE
from shared import *

class TransformerPianoDataset(Dataset):
    def __init__(
        self, 
        limit_n_notes: int, # in case positional encoding repeats
        dir_path: str = TRANSFORMER_PIANO_DATASET_DIR, 
        offset: int = 0, 
        truncate_to_size: Optional[int] = None, 
        device: torch.device = CPU, 
    ):
        def getStems():
            with open(path.join(dir_path, 'index.json'), encoding='utf-8') as f:
                stems = json.load(f)
            stems = stems[offset:]
            if truncate_to_size is not None:
                assert truncate_to_size <= len(stems)
                stems = stems[:truncate_to_size]
            return stems
        self.stems = getStems()
        self.X = []
        self.Y = torch.zeros((
            len(self.stems), ENCODEC_N_BOOKS, N_TOKENS_PER_DATAPOINT, 
        ), dtype=torch.int16, device=device)
        n_notes_array = torch.zeros(len(self.stems))
        for i, datapoint_id in enumerate(tqdm(self.stems, 'Load dataset')):
            x: Tensor = torch.load(path.join(dir_path, f'{datapoint_id}_x.pt'))
            y: Tensor = torch.load(path.join(dir_path, f'{datapoint_id}_y.pt'))
            n_notes, _ = x.shape
            n_notes_array[i] = n_notes
            real_x = torch.zeros((n_notes, 1 + 1 + 88))
            real_x[:, :2] = x[:, :2]
            ladder = torch.arange(n_notes)
            real_x[ladder, 2 + x[ladder, 2].to(torch.int) - PIANO_RANGE[0]] = 1.0
            self.X.append(real_x.to(device))
            self.Y[i, :, :] = y
        max_notes = round(n_notes_array.max().item())
        print('The densest piece has', max_notes, 'notes.')
        assert max_notes <= limit_n_notes
        print('mean + 2std:', n_notes_array.mean().item() + 2 * n_notes_array.std().item())

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, index: int):
        return self.X[index], self.Y[index, :, :]

if __name__ == '__main__':
    dataset = TransformerPianoDataset(limit_n_notes=10000)
    import IPython; IPython.embed()
