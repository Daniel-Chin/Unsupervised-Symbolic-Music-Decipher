import socket
from functools import lru_cache

import torchaudio.transforms
import numpy as np

import init as _
from paths import *
from torchwork_mini import *
from domestic_typing import *

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

    return stft, griffinLim, n_bins
