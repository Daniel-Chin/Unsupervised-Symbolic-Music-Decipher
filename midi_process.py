from __future__ import annotations

import copy
import os
from os import path
import random
from pprint import pprint
import json
import argparse

import pretty_midi
import tqdm
from matplotlib import pyplot as plt
from scipy.stats import norm

from shared import *
from music import PIANO_RANGE

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
    if instrument.program not in PIANOABLE_INSTRUMENTS:
        return False
    if instrument.is_drum:
        return False
    for note in instrument.notes:
        note: pretty_midi.Note
        if note.pitch not in range(*PIANO_RANGE):
            return False
    return True

def everythingPiano(midi: pretty_midi.PrettyMIDI):
    # turn every instrument into piano. 
    new_midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(0, is_drum=False, name='Piano')
    piano.pitch_bends = []
    piano.control_changes = []
    new_midi.instruments.append(piano)
    for instrument in midi.instruments:
        instrument: pretty_midi.Instrument
        for note in instrument.notes:
            note: pretty_midi.Note
            piano.notes.append(pretty_midi.Note(
                note.velocity, note.pitch, note.start, note.end,
            ))
    piano.notes.sort(key=lambda note: note.start)
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

def main(select_dirs: Optional[List[str]] = None, limit: Optional[int] = None):
    dirs = select_dirs or LA_DATASET_DIRS
    for dir_i, dir_ in enumerate(dirs):
        OK = 'OK'
        midi_exceptions = { OK: 0 }
        def accException(e: Exception):
            midi_exceptions[repr(e)] = midi_exceptions.get(repr(e), 0) + 1
        os.makedirs(path.join(
            PIANO_LA_DATASET_DIR, dir_, 
        ), exist_ok=True)
        srcs = os.listdir(path.join(LA_MIDI_DIR, dir_))
        if limit is not None:
            srcs = random.choices(srcs, k=limit)
        dests = []
        for src_basename in tqdm.tqdm(
            srcs, desc=f'midi {dir_i}/{len(dirs)}',
        ):
            src_abs_name = path.join(LA_MIDI_DIR, dir_, src_basename)
            try:
                original = pretty_midi.PrettyMIDI(src_abs_name)
            except Exception as e:
                accException(e)
                continue
            smart_piano = everythingPiano(filterInstruments(original))
            if not smart_piano.instruments[0].notes:
                accException(ValueError('No notes in piano track'))
                continue
            trimStart(smart_piano)
            dest_basename = src_basename
            smart_piano.write(path.join(
                PIANO_LA_DATASET_DIR, dir_, dest_basename, 
            ))
            midi_exceptions[OK] += 1
            dests.append(dest_basename)
        with open(path.join(
            PIANO_LA_DATASET_DIR, dir_, 'index.json', 
        ), 'w', encoding='utf-8') as f:
            json.dump(dests, f)
        print()
        pprint(midi_exceptions)
    print('OK')

def noteStats(limit: Optional[int] = None):
    with open(path.join(
        PIANO_LA_DATASET_DIR, 'index.json', 
    ), 'r', encoding='utf-8') as f:
        all_dir_ = json.load(f)
    densities = []
    durations = []
    velocities = []
    pitches = []
    for dir_ in tqdm.tqdm(all_dir_):
        with open(path.join(
            PIANO_LA_DATASET_DIR, dir_, 'index.json', 
        ), 'r', encoding='utf-8') as f:
            filenames = json.load(f)
        if limit is not None:
            filenames = random.choices(filenames, k=limit)
        for filename in filenames:
            midi = pretty_midi.PrettyMIDI(path.join(
                PIANO_LA_DATASET_DIR, dir_, filename,
            ))
            ( piano, ) = midi.instruments
            piano: pretty_midi.Instrument
            for note in piano.notes:
                note: pretty_midi.Note
                durations.append(note.end - note.start)
                velocities.append(note.velocity)
                pitches.append(note.pitch)
            density = len(piano.notes) / piano.get_end_time()
            densities.append(density)
    log_densities = [np.log(x) for x in densities]
    log_durations = [np.log(x) for x in durations]
    density_mu, density_sigma = norm.fit(log_densities)
    duration_mu, duration_sigma = norm.fit(log_durations)
    velocity_mu, velocity_sigma = norm.fit(velocities)
    pitch_mu, pitch_sigma = norm.fit(pitches)
    print(f'{(density_mu, density_sigma) = }')
    print(f'{(duration_mu, duration_sigma) = }')
    print(f'{(velocity_mu, velocity_sigma) = }')
    print(f'{(pitch_mu, pitch_sigma) = }')
    
    def show(
        densities_bins=30, durations_bins=80, 
        velocities_bins=127, pitches_bins=88, 
    ):
        plt.hist(velocities, density=True, bins=velocities_bins)
        X = np.linspace(min(velocities), max(velocities), 100)
        Y = norm.pdf(X, velocity_mu, velocity_sigma)
        plt.plot(X, Y, 'r--', linewidth=2)
        plt.title('Velocity')
        plt.show()

        plt.hist(pitches, density=True, bins=pitches_bins)
        X = np.linspace(min(pitches), max(pitches), 100)
        Y = norm.pdf(X, pitch_mu, pitch_sigma)
        plt.plot(X, Y, 'r--', linewidth=2)
        plt.title('Pitch')
        plt.show()

        plt.hist(log_densities, density=True, bins=densities_bins)
        X = np.linspace(min(log_densities), max(log_densities), 100)
        Y = norm.pdf(X, density_mu, density_sigma)
        plt.plot(X, Y, 'r--', linewidth=2)
        plt.title('Log Note Density, ln(notes / sec)')
        plt.show()

        plt.hist(log_durations, density=True, bins=durations_bins)
        X = np.linspace(min(log_durations), max(log_durations), 100)
        Y = norm.pdf(X, duration_mu, duration_sigma)
        plt.plot(X, Y, 'r--', linewidth=2)
        plt.title('Log Note Duration, ln(sec)')
        plt.show()
    show()
    import IPython; IPython.embed()

def trimStart(midi: pretty_midi.PrettyMIDI):
    PADDING = 0.5
    piano, = midi.instruments
    piano: pretty_midi.Instrument
    notes = piano.notes
    notes: List[pretty_midi.Note]
    song_start = min(note.start for note in notes) - PADDING
    for note in notes:
        note.start -= song_start
        note.end -= song_start

if __name__ == '__main__':
    # test()
    # main([*LA_DATASET_DIRS], 4)
    # noteStats(30)

    parser = argparse.ArgumentParser()
    parser.add_argument('--select_dirs', nargs='+')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    main(args.select_dirs, args.limit)
