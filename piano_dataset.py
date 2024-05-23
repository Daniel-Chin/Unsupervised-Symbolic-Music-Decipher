import json
from os import path
from threading import Lock

import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from shared import *
from music import PIANO_RANGE

class PianoDataset(Dataset):
    def __init__(
        self, name: str, dir_path: str, 
        need_encodec_tokens: bool, need_log_spectrogram: bool, 
        size: int, offset: int = 0, 
        device: torch.device = CPU, 
    ):
        self.name = name
        self.dir_path = dir_path
        self.size = size
        self.has_encodec_tokens = need_encodec_tokens
        self.has_log_spectrogram = need_log_spectrogram
        self.lock = Lock()
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
        self.score = torch.zeros((
            len(self.data_ids), 
            2, 
            PIANO_RANGE[1] - PIANO_RANGE[0], 
            N_FRAMES_PER_DATAPOINT, 
        ), device=device)
        if need_encodec_tokens:
            self.encodec_tokens = torch.zeros((
                len(self.data_ids), ENCODEC_N_BOOKS, N_FRAMES_PER_DATAPOINT, 
            ), dtype=torch.int16, device=device)
        if need_log_spectrogram:
            _, _, n_bins = fftTools()
            self.log_spectrigram = torch.zeros((
                len(self.data_ids), n_bins, N_FRAMES_PER_DATAPOINT, 
            ), dtype=torch.float16, device=device)

        ram = 0
        def ramOf(t: Tensor):
            return t.nelement() * t.element_size()
        ram += ramOf(self.score)
        if self.has_encodec_tokens:
            ram += ramOf(self.encodec_tokens)
        if self.has_log_spectrogram:
            ram += ramOf(self.log_spectrigram)
        print(f'dataset RAM: {ram / 2**30 : .2f} GB', flush=True)

        self._has_cached = set()
        
    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        '''
        `score`: (2, PIANO_RANGE[1] - PIANO_RANGE[0], N_FRAMES_PER_DATAPOINT)
        `encodec_tokens`: (ENCODEC_N_BOOKS, N_FRAMES_PER_DATAPOINT)
        `log_spectrogram`: (n_bins, N_FRAMES_PER_DATAPOINT)
        `data_id`: str
        '''
        with self.lock:
            lets_fetch = index not in self._has_cached
        if lets_fetch:
            datapoint_id = self.data_ids[index]
            prefix = path.join(self.dir_path, datapoint_id)
            score: Tensor = torch.load(prefix + '_score.pt')
            if self.has_encodec_tokens:
                encodec_tokens: Tensor = torch.load(prefix + '_encodec_tokens.pt')
            if self.has_log_spectrogram:
                log_spectrogram: Tensor = torch.load(prefix + '_log_spectrogram.pt')
            with self.lock:
                if index not in self._has_cached:
                    self.score[index, :, :, :] = score
                    if self.has_encodec_tokens:
                        self.encodec_tokens[index, :, :] = encodec_tokens
                    if self.has_log_spectrogram:
                        self.log_spectrigram[index, :, :] = log_spectrogram
                    self._has_cached.add(index)
        return (
            self.score[index, :, :, :], 
            self.encodec_tokens[index, :, :] if self.has_encodec_tokens else None, 
            self.log_spectrigram[index, :, :].to(torch.float32) if self.has_log_spectrogram else None, 
            self.data_ids[index],
        )

BatchType = Tuple[Tensor, Optional[Tensor], Optional[Tensor], Tuple[str]]

if __name__ == '__main__':
    dataset = PianoDataset(
        '0', PIANO_MONKEY_DATASET_DIR, True, True, 32, 
    )
    import IPython; IPython.embed()

    # CollateCandidates.profile()
