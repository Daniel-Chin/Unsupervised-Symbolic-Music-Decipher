import sys
import os
from os import path
from subprocess import Popen, DEVNULL

from shared import *

def midiSynthWav(
    midi_path: str, wav_path: str, 
    verbose: bool = False, 
    do_fluidsynth_write_pcm: bool = False,
):
    if verbose:
        os.system('which fluidsynth')
        stdout = sys.stdout
        stderr = sys.stderr
    else:
        stdout = DEVNULL
        stderr = DEVNULL
    pcm_path = wav_path + '.pcm'
    synth_out = pcm_path if do_fluidsynth_write_pcm else wav_path
    cmd = [
        'fluidsynth', '-ni', SOUNDFONT_FILE, midi_path,
        '-F', synth_out, 
        '-r', str(ENCODEC_SR), 
    ]
    for _ in range(4):
        if verbose:
            print(cmd, flush=True)
        with Popen(cmd, stdout=stdout, stderr=stderr) as p:
            p.wait()
        if path.isfile(synth_out):
            break
        print('fluidsynth did not write file, retry...')
        print('>which fluidsynth', flush=True)
        os.system('which fluidsynth')
    else:
        raise Exception('max retries exceeded, fluidsynth did not write file')
    
    if do_fluidsynth_write_pcm:
        cmd = [
            'ffmpeg', '-f', 's16le', '-ar', str(ENCODEC_SR), 
            '-ac', '1', '-i', pcm_path, '-y', wav_path, 
        ]
        if verbose:
            print(cmd, flush=True)
        with Popen(cmd, stdout=stdout, stderr=stderr) as p:
            p.wait()
