import socket
from functools import lru_cache
from dataclasses import dataclass
import pdb

from torch import Tensor
import torchaudio.transforms
from torch.utils.data import Dataset, DataLoader
import lightning
import numpy as np
from matplotlib.axes import Axes
import pretty_midi

import init as _
from paths import *
from torchwork_mini import *
from domestic_typing import *
from music import *

DO_CHECK_NAN = not bool(os.environ.get('SLURM_JOB_ID'))

SEC_PER_DATAPOINT = 30
ENCODEC_SR = 32000
N_SAMPLES_PER_DATAPOINT = SEC_PER_DATAPOINT * ENCODEC_SR
ENCODEC_N_BOOKS = 4
ENCODEC_N_WORDS_PER_BOOK = 2048
ENCODEC_FPS = 50
N_FRAMES_PER_DATAPOINT = SEC_PER_DATAPOINT * ENCODEC_FPS
ENCODEC_RECEPTIVE_RADIUS = 0.1   # sec

LA_DATASET_DIRS = [*'0123456789abcdef']

TWO_PI = np.pi * 2

@dataclass(frozen=True)
class PcmFormat:
    name: str
    np_dtype: np.dtype

DEFAULT_PCM_FORMAT = PcmFormat(
    's16le',
    np.dtype(np.int16).newbyteorder('<'), 
)   # these two have to agree

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

def sortByNoteOn(piano: pretty_midi.Instrument):
    # Note pretty_midi write + load doesn't preserve note order.  
    piano.notes.sort(key=lambda x: x.start)

def bytesToAudioWave(
    b: bytes, /, in_n_channels: int = 1, 
    in_format: PcmFormat = DEFAULT_PCM_FORMAT, 
    out_type: np.dtype = np.dtype(np.float32), 
):
    '''
    Always return mono.  
    '''
    dtype = in_format.np_dtype
    wave_int = np.frombuffer(b, dtype=dtype)
    format_factor: int = 2 ** (dtype.itemsize * 8 - 1)  # needs type hint because type checker doesn't know dtype.itemsize > 0
    wave_float = wave_int.astype(out_type) / format_factor
    if in_n_channels == 1:
        wave_mono = wave_float
    else:
        wave_mono = wave_float.reshape(-1, in_n_channels).mean(axis=1)
    return wave_mono

def printMidi(piano: pretty_midi.Instrument):
    WIDTH = 116
    INTERVAL = 0.05  # sec
    def velocityAsChar(x: int, /):
        s = str(int(x / 128 * 10))
        assert len(s) == 1
        return s

    sortByNoteOn(piano)
    print('=====================')
    rows = [[' '] * WIDTH for _ in range(*PIANO_RANGE)]
    for note in piano.notes:
        note: pretty_midi.Note
        start = int(note.start / INTERVAL)
        end = int(note.end / INTERVAL)
        if start >= WIDTH:
            break
        row = rows[note.pitch - PIANO_RANGE[0]]
        if row[start] in (' ', '-'):
            row[start] = velocityAsChar(note.velocity)
        else:
            row[start] = '?'
        for i in range(start + 1, end):
            try:
                row[i] = '-'
            except IndexError:
                break
    for p, row in reversed([*enumerate(rows)]):
        try:
            heading = pitch2name(p + PIANO_RANGE[0])
        except NonDiatone:
            heading = '  '
        print(heading, ':', *row, sep='')
