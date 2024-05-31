from __future__ import annotations

import sys
import os
from os import path
from subprocess import Popen, DEVNULL, PIPE
from io import BytesIO
import audioread
from audioread.rawread import RawAudioFile
from contextlib import contextmanager

import numpy as np
from scipy.signal import resample
from matplotlib import pyplot as plt

from shared import *

NYUSH_FLUIDSYNTH_FORMAT = DEFAULT_PCM_FORMAT

def midiSynthWave(
    midi_path: str, temp_synth_path: str, 
    synthAnomalyChecker: SynthAnomalyChecker, 
    verbose: bool = False, 
    is_fluidsynth_nyush: bool = False,
):
    if verbose:
        os.system('which fluidsynth')
        stdout = sys.stdout
        stderr = sys.stderr
    else:
        stdout = DEVNULL
        stderr = DEVNULL
    synth_out_path = temp_synth_path + (
        '.pcm' if is_fluidsynth_nyush else '.wav'
    )
    cmd = [
        'fluidsynth', '-ni', SOUNDFONT_FILE, midi_path,
        '-F', synth_out_path, 
        '--sample-rate', str(ENCODEC_SR),
    ]
    for _ in range(4):
        if verbose:
            print(cmd, flush=True)
        with Popen(cmd, stdout=stdout, stderr=stderr) as p:
            p.wait()
        if path.isfile(synth_out_path):
            break
        print('fluidsynth did not write file, retry...')
        print('>which fluidsynth', flush=True)
        os.system('which fluidsynth')
    else:
        raise Exception('max retries exceeded, fluidsynth did not write file')
    
    if is_fluidsynth_nyush:
        # wave = customResample(synth_out_path)
        with open(synth_out_path, 'rb') as f:
            data = f.read()
        wave = bytesToAudioWave(data, in_n_channels=2, in_format=NYUSH_FLUIDSYNTH_FORMAT)
    else:
        buf = BytesIO()
        with audioread.audio_open(synth_out_path) as aF:
            aF: RawAudioFile
            assert aF.samplerate == ENCODEC_SR
            n_channels = aF.channels
            for chunk in aF.read_data():
                buf.write(chunk)
        buf.seek(0)
        wave = bytesToAudioWave(buf.read(), n_channels)
    
    synthAnomalyChecker.look(wave, midi_path)
    wave_trimmed = wave[:N_SAMPLES_PER_DATAPOINT]
    wave_padded = np.pad(wave_trimmed, (0, N_SAMPLES_PER_DATAPOINT - len(wave_trimmed)))
    return wave_padded

def customResample(pcm_path: str) -> np.ndarray:
    # It was such a hassle to configure ffmpeg... Let's code from scratch.  
    with open(pcm_path, 'rb') as f:
        data = f.read()
    wave = bytesToAudioWave(data, in_n_channels=2, in_format=NYUSH_FLUIDSYNTH_FORMAT)
    resampled = resample(wave, round(SEC_PER_DATAPOINT * ENCODEC_SR))
    return resampled    # type: ignore

# legacy
# def assertSR(ffprob_output_str: str):
#     _, duration_line = ffprob_output_str.split(
#         '\n  Duration: ', 
#     )
#     h, m, s = duration_line.split(',', 1)[0].split(':')
#     duration = (int(h) * 60 + int(m)) * 60 + float(s)
#     _, sr_line = ffprob_output_str.split(
#         '\n  Stream #0:0: Audio', 
#     )
#     sr_l, _ = sr_line.split(' Hz,')
#     declared_sr = int(sr_l.split(' ')[-1])
#     n_samples = declared_sr * duration
#     inferred_sr = n_samples / SEC_PER_DATAPOINT
#     assert abs(math.log(NYUSH_FLUIDSYNTH_SR / inferred_sr)) < math.log(1.1), inferred_sr

ALLOWED_TAIL = 0.9  # sec
ALLOWED_TAIL_N_SAMPLES = round(ALLOWED_TAIL * ENCODEC_SR)
class SynthAnomalyChecker:
    def __init__(self) -> None:
        self.n_good = 0
        self.n_kinda_bad = 0
        self.n_bad = 0
        self.ready = False

    def look(
        self, wave: np.ndarray, 
        midi_path: str, 
    ) -> None:
        assert self.ready, 'Use me as a python context'
        extra_time = (len(wave) - N_SAMPLES_PER_DATAPOINT) / ENCODEC_SR
        if extra_time < -1.0:
            self.n_kinda_bad += 1
            return
        forbidden_tail = wave[N_SAMPLES_PER_DATAPOINT + ALLOWED_TAIL_N_SAMPLES:]
        if len(forbidden_tail) == 0:
            self.n_good += 1
            return
        if np.sum(np.abs(forbidden_tail) > 1e5) < 0.1 * ENCODEC_SR:
            self.n_good += 1
        else:
            self.n_bad += 1
            plt.plot(np.abs(forbidden_tail) > 1e5)
            plt.show()
        if self.total() >= 30:
            self.checkRatio()
    
    def total(self):
        return self.n_good + self.n_kinda_bad + self.n_bad
    
    def checkRatio(self) -> None:
        t = (self.n_good, self.n_kinda_bad, self.n_bad)
        assert self.n_bad / self.total() <= 0.01, t
        assert self.n_kinda_bad / self.total() <= 0.1, t
    
    @contextmanager
    def context(self):
        self.ready = True
        try:
            yield self
        finally:
            self.close()
    
    def close(self):
        # print(*self.data, sep='\n')
        # plt.hist(self.data, bins=50)
        # plt.show()
        self.checkRatio()
