from subprocess import Popen
from io import TextIOWrapper

from shared import *

def midiSynthWav(
    midi_path: str, wav_path: str, 
    logStream: TextIOWrapper, do_fluidsynth_write_pcm: bool = False,
):
    pcm_path = wav_path + '.pcm'
    cmd = [
        'fluidsynth', '-ni', SOUNDFONT_FILE, midi_path,
        '-F', pcm_path if do_fluidsynth_write_pcm else wav_path, 
        '-r', str(ENCODEC_SR), 
    ]
    print(cmd, file=logStream)
    with Popen(cmd, stdout=logStream) as p:
        p.wait()
    
    if do_fluidsynth_write_pcm:
        print(file=logStream)
        cmd = [
            'ffmpeg', '-f', 's16le', '-ar', str(ENCODEC_SR), 
            '-ac', '1', '-i', pcm_path, wav_path, 
        ]
        print(cmd, file=logStream)
        with Popen(cmd, stdout=logStream, stderr=logStream) as p:
            p.wait()
