import json
from os import path

import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from shared import *
from music import PIANO_RANGE

class PianoDataset(Dataset):
    def __init__(
        self, name: str, dir_path: str, 
        size: int, offset: int = 0, 
        score_only: bool = False, 
        device: torch.device = CPU, 
    ):
        self.name = name
        self.score_only = score_only
        def getDataIds():
            s: List[str] = []
            for dir_ in LA_DATASET_DIRS:
                with open(path.join(dir_path, dir_, 'index.json'), encoding='utf-8') as f:
                    for x in json.load(f):
                        s.append(path.join(dir_, x))
                        # `path.join` behaves differently across platforms, resulting in different data_ids. That's ok, for now.
            s = s[offset:]
            assert size <= len(s)
            s = s[:size]
            return s
        self.data_ids = getDataIds()
        self.X = torch.zeros((
            len(self.data_ids), 
            2, 
            PIANO_RANGE[1] - PIANO_RANGE[0], 
            N_FRAMES_PER_DATAPOINT, 
        ), device=device)
        if not score_only:
            self.Y = torch.zeros((
                len(self.data_ids), ENCODEC_N_BOOKS, N_FRAMES_PER_DATAPOINT, 
            ), dtype=torch.int16, device=device)
        for i, datapoint_id in enumerate(tqdm(self.data_ids, f'Load dataset "{self.name}"')):
            x: Tensor = torch.load(path.join(dir_path, f'{datapoint_id}_x.pt'))
            self.X[i, :, :, :] = x
            if not score_only:
                y: Tensor = torch.load(path.join(dir_path, f'{datapoint_id}_y.pt'))
                self.Y[i, :, :] = y
        ram = 0
        def ramOf(t: Tensor):
            return t.nelement() * t.element_size()
        ram += len(self.X) * ramOf(self.X[0])
        if not score_only:
            ram += ramOf(self.Y)
        print(f'dataset RAM: {ram / 2**30 : .2f} GB', flush=True)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, index: int):
        '''
        `x`: (2, PIANO_RANGE[1] - PIANO_RANGE[0], N_TOKENS_PER_DATAPOINT)
        `y`: (ENCODEC_N_BOOKS, N_TOKENS_PER_DATAPOINT)
        `data_id`: str
        '''
        if self.score_only:
            return self.X[index, :, :, :], self.data_ids[index]
        else:
            return self.X[index, :, :, :], self.Y[index, :, :], self.data_ids[index]

BatchType = Tuple[Tensor, Tensor, Tuple[str]]
BatchTypeScoreOnly = Tuple[Tensor, Tuple[str]]

if __name__ == '__main__':
    dataset = PianoDataset(
        '0', PIANO_MONKEY_DATASET_DIR, 32, 
    )
    import IPython; IPython.embed()

    # CollateCandidates.profile()
