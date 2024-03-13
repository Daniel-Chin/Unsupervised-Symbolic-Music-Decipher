from __future__ import annotations

from typing import *
import copy
import os
from os import path
import random

import pretty_midi
import tqdm

from shared import *

PIANOABLE_INSTRUMENTS = [
    *range(0, 8),     # piano
    *range(8, 16),    # chromatic percussion
    *range(16, 24),   # organ
    *range(24, 32),   # guitar
    *range(32, 40),   # bass
    *range(40, 48),   # strings
    *range(48, 56),   # ensemble
    *range(56, 64),   # brass
    *range(64, 72),   # reed
    *range(72, 80),   # pipe
    *range(80, 88),   # synth lead
    # *range(88, 96),   # synth pad
    # *range(96, 104),  # synth effects
    # *range(104, 112), # ethnic
    # *range(112, 120), # percussive
    # *range(120, 128), # sound effects
]

def isPianoable(instrument: pretty_midi.Instrument):
    return (
        instrument.program in PIANOABLE_INSTRUMENTS 
    and 
        not instrument.is_drum
    )

def everythingPiano(midi: pretty_midi.PrettyMIDI):
    # turn every instrument into piano. 
    new_midi = copy.deepcopy(midi)
    for instrument in new_midi.instruments:
        instrument: pretty_midi.Instrument
        instrument.program = 0
        instrument.name = 'Piano'
        instrument.is_drum = False
        instrument.pitch_bends = []
        instrument.control_changes = []
    return new_midi

def filterInstruments(midi: pretty_midi.PrettyMIDI):
    new_midi = copy.deepcopy(midi)
    instruments: List[pretty_midi.Instrument] = new_midi.instruments
    new_midi.instruments = [
        instrument for instrument in instruments 
        if isPianoable(instrument)
    ]
    return new_midi

def isolateFilteredInstruments(midi: pretty_midi.PrettyMIDI):
    new_midi = copy.deepcopy(midi)
    instruments: List[pretty_midi.Instrument] = new_midi.instruments
    new_midi.instruments = [
        instrument for instrument in instruments 
        if not isPianoable(instrument)
    ]
    return new_midi

def inspect(midi_path: str):
    basename = path.basename(midi_path)
    midis: Dict[str, pretty_midi.PrettyMIDI] = {}
    original = pretty_midi.PrettyMIDI(midi_path)
    midis['0_original'] = original
    midis['1_force_piano'] = everythingPiano(original)
    midis['2_smart_piano'] = everythingPiano(filterInstruments(original))
    midis['3_removed'] = isolateFilteredInstruments(original)
    for name, midi in midis.items():
        midi.write(path.join(INSPECT_DIR, basename + f'.{name}.mid'))

def test():
    inspect(path.join(
        LA_MIDI_DIR, '2', '2a0d0faf73ae43fdd6da1e9fd0556231.mid',
    ))
    for _ in tqdm.tqdm(range(16)):
        dir_ = random.choice(os.listdir(LA_MIDI_DIR))
        filename = random.choice(os.listdir(path.join(LA_MIDI_DIR, dir_)))
        inspect(path.join(LA_MIDI_DIR, dir_, filename))

if __name__ == '__main__':
    test()
