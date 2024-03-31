import os
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
    cmd = [
        'fluidsynth', '-ni', SOUNDFONT_FILE, midi_path,
        '-F', pcm_path if do_fluidsynth_write_pcm else wav_path, 
        '-r', str(ENCODEC_SR), 
    ]
    print(cmd, flush=debug)
    with Popen(cmd) as p:
        p.wait()
    
    if do_fluidsynth_write_pcm:
        print(flush=debug)
        cmd = [
            'ffmpeg', '-f', 's16le', '-ar', str(ENCODEC_SR), 
            '-ac', '1', '-i', pcm_path, '-y', wav_path, 
        ]
        print(cmd, flush=debug)
        with Popen(cmd) as p:
            p.wait()
