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
        self, name: str, dir_path: str, 
        size: int, offset: int = 0, 
        device: torch.device = CPU, 
    ):
        self.name = name
        def getStems():
            with open(path.join(dir_path, 'index.json'), encoding='utf-8') as f:
                stems: List[str] = json.load(f)
            stems = stems[offset:]
            assert size <= len(stems)
            stems = stems[:size]
            return stems
        self.stems = getStems()
        self.X: List[Tensor] = []
        self.Y = torch.zeros((
            len(self.stems), ENCODEC_N_BOOKS, N_TOKENS_PER_DATAPOINT, 
        ), dtype=torch.int16, device=device)
        n_notes_array = torch.zeros((len(self.stems), ))
        for i, datapoint_id in enumerate(tqdm(self.stems, f'Load dataset "{self.name}"')):
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
        ram = 0
        def ramOf(t: Tensor):
            return t.nelement() * t.element_size()
        ram += len(self.X) * ramOf(self.X[0])
        ram += ramOf(self.Y)
        print('dataset RAM:', ram / 2**30, 'GB', flush=True)

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, index: int):
        # x: (n_notes, 1 + 1 + 88)
        # y: (ENCODEC_N_BOOKS, N_TOKENS_PER_DATAPOINT)
        # stem: str
        return self.X[index], self.Y[index, :, :], self.stems[index]

class CollateCandidates:
    @staticmethod
    def usingZip(data: List[Tuple[Tensor, Tensor, str]]) -> Tuple[Tensor, Tensor, List[int], List[str]]:
        X, Y, stems = zip(*data)
        X: Tuple[Tensor]
        Y: Tuple[Tensor]
        X_lens = [x.shape[0] for x in X]
        return (
            torch.nn.utils.rnn.pad_sequence([*X], batch_first=True), 
            torch.stack(Y, dim=0),
            X_lens,
            stems, 
        )

    @staticmethod
    def usingInplace(data: List[Tuple[Tensor, Tensor, str]]) -> Tuple[Tensor, Tensor, List[int], List[str]]:
        device = data[0][0].device
        batch_size = len(data)
        X_lens = [x.shape[0] for x, _, _ in data]
        X_width = max(X_lens)
        X = torch.zeros((batch_size, X_width, data[0][0].shape[1]), device=device)
        Y = torch.zeros((
            batch_size, ENCODEC_N_BOOKS, N_TOKENS_PER_DATAPOINT, 
        ), dtype=torch.int16, device=device)
        for i, (x, y, stem) in enumerate(data):
            X[i, :x.shape[0], :] = x
            Y[i, :, :] = y
        return X, Y, X_lens, [stem for _, _, stem in data]

    @staticmethod
    def usingStack(data: List[Tuple[Tensor, Tensor, str]]) -> Tuple[Tensor, Tensor, List[int], List[str]]:
        X: List[Tensor] = []
        Y: List[Tensor] = []
        stems: List[str] = []
        for x, y, stem in data:
            X.append(x)
            Y.append(y)
            stems.append(stem)
        X_lens = [x.shape[0] for x in X]
        return (
            torch.nn.utils.rnn.pad_sequence(X, batch_first=True), 
            torch.stack(Y, dim=0), 
            X_lens,
            stems,
        )

    @staticmethod
    def profile(n=64):
        candidates: List[Tuple[str, Callable[
            [List[Tuple[Tensor, Tensor, str]]], Tuple[Tensor, Tensor, List[int], List[str]]
        ]]] = [
            ('zip',     CollateCandidates.usingZip), 
            ('inplace', CollateCandidates.usingInplace), 
            ('stack',   CollateCandidates.usingStack), 
        ]
        dataset = TransformerPianoDataset('0', TRANSFORMER_PIANO_MONKEY_DATASET_DIR, n)
        data = [dataset[i] for i in range(n)]
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

def collate(data: List[Tuple[Tensor, Tensor, str]]) -> Tuple[Tensor, Tensor, List[int], List[str]]:
    x, y, x_lens, stems = CollateCandidates.usingZip(data)
    return x, y.to(torch.int64), x_lens, stems

if __name__ == '__main__':
    dataset = TransformerPianoDataset('0', TRANSFORMER_PIANO_MONKEY_DATASET_DIR, 64)
    import IPython; IPython.embed()

    # CollateCandidates.profile()
