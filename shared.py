import socket
from functools import lru_cache
import pdb

from torch import Tensor
import torchaudio.transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from matplotlib.axes import Axes

import init as _
from paths import *
from torchwork_mini import *
from domestic_typing import *
from music import PIANO_RANGE

DO_CHECK_NAN = not bool(os.environ.get('SLURM_JOB_ID'))

SEC_PER_DATAPOINT = 30
ENCODEC_SR = 32000
ENCODEC_N_BOOKS = 4
ENCODEC_N_WORDS_PER_BOOK = 2048
ENCODEC_FPS = 50
N_FRAMES_PER_DATAPOINT = SEC_PER_DATAPOINT * ENCODEC_FPS
ENCODEC_RECEPTIVE_RADIUS = 0.1   # sec

LA_DATASET_DIRS = [*'0123456789abcdef']

TWO_PI = np.pi * 2

def initMainProcess():
    print('hostname:', socket.gethostname())
    print(f'{GPU_NAME = }')
    print(f'{DEVICE.index = }')

@lru_cache(maxsize=1)
def fftTools():
    n_fft = round(ENCODEC_RECEPTIVE_RADIUS * 2 * ENCODEC_SR)
    hop_length = ENCODEC_SR // ENCODEC_FPS
    power = 2
    
    stft = torchaudio.transforms.Spectrogram(
        n_fft, 
        hop_length = hop_length, 
        power = power, 
    ).to(DEVICE)

    griffinLim = torchaudio.transforms.GriffinLim(
        n_fft, 
        n_iter = 32, 
        hop_length = hop_length, 
        power = power, 
    ).to(DEVICE)

    n_bins = n_fft // 2 + 1
    print('# of freq bins for STFT:', n_bins, flush=True)

    return stft, griffinLim, n_bins

def plotScore(score: Tensor, ax: Axes):
    return ax.imshow(
        score.permute(1, 0, 2).reshape(
            2 * (PIANO_RANGE[1] - PIANO_RANGE[0]), 
            N_FRAMES_PER_DATAPOINT, 
        ), aspect='auto', interpolation='nearest', 
        origin='lower', 
    )

def howMuchNan(x: Tensor, /):
    return x.isnan().sum().item() / x.numel()

def myChosenDataLoader(dataset: Dataset, batch_size: int, shuffle: bool):
    kw = dict(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collateWithNone, 
    )
    # return SingleProcessNewThreadPreFetchDataLoader(
    #     **kw, # type: ignore
    #     prefetch_factor=3, 
    # )
    return DataLoader(
        **kw, # type: ignore
        # num_workers=2, 
        # prefetch_factor=2, 
        num_workers=0, 
        # persistent_workers=True,
    )
