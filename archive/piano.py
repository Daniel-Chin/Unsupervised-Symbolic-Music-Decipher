from functools import lru_cache

import torch
import torchaudio.transforms as T
import numpy as np

from shared import *
from music import pitch2freq

SR = 16000
N_SAMPLES_PER_SEG = 2048
N_SAMPLES_PER_HOP = 1024
N_FREQ_BINS = N_SAMPLES_PER_SEG // 2 + 1
FREQ_BINS = np.linspace(0, SR / 2, N_FREQ_BINS)
# WINDOW_NAME = 'hann'
# WINDOW_TENSOR = torch.Tensor(scipy.signal.get_window(
#     WINDOW_NAME, N_SAMPLES_PER_SEG, True, 
# ))

MAX_SEC_PER_NOTE = 4.0
N_SAMPLES_PER_NOTE = round(SR * MAX_SEC_PER_NOTE)
DECAY_MASK = np.exp(np.linspace(
    0.0, -1.5 * MAX_SEC_PER_NOTE, N_SAMPLES_PER_NOTE, 
))

STFT = T.Spectrogram(
    n_fft     =N_SAMPLES_PER_SEG, 
    win_length=N_SAMPLES_PER_SEG, 
    hop_length=N_SAMPLES_PER_HOP, 
)

def sinusoidal(freq: float, length: int, phase: float):
    return np.sin((np.arange(length) * freq / SR + phase) * TWO_PI)

def spectrogramToAudio(spectrogram: torch.Tensor):
    return griffinlim(
        spectrogram, WINDOW_TENSOR, 
        n_fft     =N_SAMPLES_PER_SEG, 
        hop_length=N_SAMPLES_PER_HOP, 
        win_length=N_SAMPLES_PER_SEG,
        power=1.0, n_iter=32, momentum=0.99, 
        length=None, rand_init=False, 
    )

def timbre(freq: float):
    if freq < 300:
        return 1.0 - (300 - freq) * 0.1
    if freq < 500:
        return 1.0 - (freq - 300) / 200 * 0.6
    if freq < 700:
        return 0.4 - (freq - 500) / 200 * 0.3
    if freq < 2000:
        return 0.1
    return 0.0

@lru_cache(maxsize=128)
def keySpectrogram(pitch: int):
    assert pitch in range(128)
    buf = []
    f0 = pitch2freq(pitch)
    freq = f0
    while freq < SR / 2:
        phase = torch.rand(1).item() * 0
        buf.append(sinusoidal(freq, N_SAMPLES_PER_NOTE, phase) * timbre(freq))
        freq += f0
    signal = np.sum(buf, axis=0) * DECAY_MASK
    if pitch not in range(21, 109):
        # not in 88-key piano
        signal *= 0.0
    f, t, Zxx = stft(
        signal, fs=SR, nperseg=N_SAMPLES_PER_SEG, 
        noverlap=N_SAMPLES_PER_SEG - N_SAMPLES_PER_HOP, 
        window=WINDOW_NAME, 
    )
    assert len(t) == N_SAMPLES_PER_NOTE // N_SAMPLES_PER_HOP + 1, (len(t) , N_SAMPLES_PER_NOTE // N_SAMPLES_PER_HOP + 1)
    f: np.ndarray
    assert np.sum((f - FREQ_BINS) ** 2) < 1e-6
    return torch.Tensor(np.abs(Zxx))
