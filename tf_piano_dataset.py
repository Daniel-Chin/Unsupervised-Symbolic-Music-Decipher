import json
from os import path
from time import perf_counter, sleep

import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from shared import *
from music import PIANO_RANGE

class TransformerPianoDataset(Dataset):
    def __init__(
        self, dir_path: str, 
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
        n_notes_array = torch.zeros((len(self.stems), ))
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
        print('mean + 2std:', n_notes_array.mean().item() + 2 * n_notes_array.std().item())

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, index: int):
        # x: (n_notes, 1 + 1 + 88)
        # y: (ENCODEC_N_BOOKS, N_TOKENS_PER_DATAPOINT)
        return self.X[index], self.Y[index, :, :]

class CollateCandidates:
    @staticmethod
    def usingZip(data: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor, List[int]]:
        X, Y = zip(*data)
        X: Tuple[Tensor]
        Y: Tuple[Tensor]
        X_lens = [x.shape[0] for x in X]
        return (
            torch.nn.utils.rnn.pad_sequence([*X], batch_first=True), 
            torch.stack(Y, dim=0),
            X_lens,
        )

    @staticmethod
    def usingInplace(data: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor, List[int]]:
        device = data[0][0].device
        batch_size = len(data)
        X_lens = [x.shape[0] for x, _ in data]
        X_width = max(X_lens)
        X = torch.zeros((batch_size, X_width, data[0][0].shape[1]), device=device)
        Y = torch.zeros((
            batch_size, ENCODEC_N_BOOKS, N_TOKENS_PER_DATAPOINT, 
        ), dtype=torch.int16, device=device)
        for i, (x, y) in enumerate(data):
            X[i, :x.shape[0], :] = x
            Y[i, :, :] = y
        return X, Y, X_lens

    @staticmethod
    def usingStack(data: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor, List[int]]:
        X: List[Tensor] = []
        Y: List[Tensor] = []
        for x, y in data:
            X.append(x)
            Y.append(y)
        X_lens = [x.shape[0] for x in X]
        return (
            torch.nn.utils.rnn.pad_sequence(X, batch_first=True), 
            torch.stack(Y, dim=0), 
            X_lens,
        )

    @staticmethod
    def profile():
        candidates: List[Tuple[str, Callable[
            [List[Tuple[Tensor, Tensor]]], Tuple[Tensor, Tensor, List[int]]
        ]]] = [
            ('zip',     CollateCandidates.usingZip), 
            ('inplace', CollateCandidates.usingInplace), 
            ('stack',   CollateCandidates.usingStack), 
        ]
        dataset = TransformerPianoDataset(TRANSFORMER_PIANO_MONKEY_DATASET_DIR)
        data = [dataset[i] for i in range(64)]
        while True:
            for name, f in candidates:
                print(name)
                dts = []
                for _ in range(1000):
                    start = perf_counter()
                    f(data)
                    dt = perf_counter() - start
                    dts.append(dt)
                print(np.mean(dts) * 1e3, 'ms')
            sleep(0.1)

collate = CollateCandidates.usingZip

if __name__ == '__main__':
    dataset = TransformerPianoDataset(TRANSFORMER_PIANO_MONKEY_DATASET_DIR)
    import IPython; IPython.embed()

    # CollateCandidates.profile()
