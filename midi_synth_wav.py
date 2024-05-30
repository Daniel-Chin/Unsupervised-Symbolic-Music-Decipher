import sys
import os
from os import path
from subprocess import Popen, DEVNULL, PIPE
from io import BytesIO
import audioread
from audioread.rawread import RawAudioFile
from scipy.signal import resample

import numpy as np

from shared import *

NYUSH_FLUIDSYNTH_FORMAT = DEFAULT_PCM_FORMAT

def midiSynthWave(
    midi_path: str, temp_synth_path: str, 
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
        wave = customResample(synth_out_path)
    else:
        buf = BytesIO()
        with audioread.audio_open(synth_out_path) as f:
            f: RawAudioFile
            assert f.samplerate == ENCODEC_SR
            n_channels = f.channels
            for chunk in f.read_data():
                buf.write(chunk)
        buf.seek(0)
        wave = bytesToAudioWave(buf.read(), n_channels)
    
    theoretical_len = int(np.ceil(SEC_PER_DATAPOINT * ENCODEC_SR))
    leeway = len(wave) - theoretical_len
    # <10ms of fluidsynth lengthening effect, usually 6ms
    assert abs(leeway / ENCODEC_SR) < 0.010, leeway / ENCODEC_SR
    wave_trimmed = wave[leeway // 2 :][: theoretical_len]
    return wave_trimmed

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
