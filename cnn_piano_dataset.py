import json
from os import path

import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from shared import *
from music import PIANO_RANGE

class TransformerPianoDataset(Dataset):
    def __init__(
        self, name: str, dir_path: str, 
        size: int, offset: int = 0, 
        device: torch.device = CPU, 
    ):
        self.name = name
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
            SEC_PER_DATAPOINT * ENCODEC_FPS, 
        ), device=device)
        self.Y = torch.zeros((
            len(self.data_ids), ENCODEC_N_BOOKS, N_TOKENS_PER_DATAPOINT, 
        ), dtype=torch.int16, device=device)
        for i, datapoint_id in enumerate(tqdm(self.data_ids, f'Load dataset "{self.name}"')):
            x: Tensor = torch.load(path.join(dir_path, f'{datapoint_id}_x.pt'))
            y: Tensor = torch.load(path.join(dir_path, f'{datapoint_id}_y.pt'))
            self.X[i, :, :, :] = x
            self.Y[i, :, :] = y
        ram = 0
        def ramOf(t: Tensor):
            return t.nelement() * t.element_size()
        ram += len(self.X) * ramOf(self.X[0])
        ram += ramOf(self.Y)
        print(f'dataset RAM: {ram / 2**30 : .2f} GB', flush=True)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, index: int):
        '''
        `x`: (2, PIANO_RANGE[1] - PIANO_RANGE[0], SEC_PER_DATAPOINT * ENCODEC_FPS)
        `y`: (ENCODEC_N_BOOKS, N_TOKENS_PER_DATAPOINT)
        `data_id`: str
        '''
        return self.X[index, :, :, :], self.Y[index, :, :], self.data_ids[index]

CollateFnIn = List[Tuple[Tensor, Tensor, str]]
CollateFnOut = Tuple[Tensor, Tensor, Tensor, List[str]]

if __name__ == '__main__':
    dataset = TransformerPianoDataset(
        '0', TRANSFORMER_PIANO_MONKEY_DATASET_DIR, 32, 
    )
    import IPython; IPython.embed()

    # CollateCandidates.profile()
