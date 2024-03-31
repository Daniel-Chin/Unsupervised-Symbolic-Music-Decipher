import os
from os import path
from subprocess import Popen

from shared import *

def midiSynthWav(
    midi_path: str, wav_path: str, 
    do_fluidsynth_write_pcm: bool = False,
    debug: bool = False, 
):
    if debug:
        os.system('which fluidsynth')
    pcm_path = wav_path + '.pcm'
    synth_out = pcm_path if do_fluidsynth_write_pcm else wav_path
    cmd = [
        'fluidsynth', '-ni', SOUNDFONT_FILE, midi_path,
        '-F', synth_out, 
        '-r', str(ENCODEC_SR), 
    ]
    for _ in range(4):
        print(cmd, flush=debug)
        with Popen(cmd) as p:
            p.wait()
        if path.isfile(synth_out):
            break
        print('fluidsynth did not write file, retry...')
        print('>which fluidsynth', flush=True)
        os.system('which fluidsynth')
    else:
        raise Exception('max retries exceeded, fluidsynth did not write file')
    
    if do_fluidsynth_write_pcm:
        print(flush=debug)
        cmd = [
            'ffmpeg', '-f', 's16le', '-ar', str(ENCODEC_SR), 
            '-ac', '1', '-i', pcm_path, '-y', wav_path, 
        ]
        print(cmd, flush=debug)
        with Popen(cmd) as p:
            p.wait()
