import typing as tp
import math

import pretty_midi

from shared import *

DECAY = 1.0

class MidiReasonablizer:
    def __init__(self, piano: tp.Optional[pretty_midi.Instrument]) -> None:
        self.piano = piano or pretty_midi.Instrument(0, is_drum=False, name='Piano')

        self.last_note: tp.List[tp.Optional[pretty_midi.Note]] = [None] * 128
    
    def add(self, note: pretty_midi.Note):
        if self.helper(note):
            self.piano.notes.append(note)
            self.last_note[note.pitch] = note
    
    def helper(self, note: pretty_midi.Note):
        p = note.pitch
        last = self.last_note[p]
        if last is None:
            return True
        assert last.start <= note.start
        if last.end < note.start:
            return True
        if abs(last.start - note.start) < 1e-6:
            last.velocity = max(last.velocity, note.velocity)
            last.end = max(last.end, note.end)
            return False
        dt = note.start - last.start
        last_energy = last.velocity ** 2 * math.exp(-dt * DECAY)
        this_energy = note.velocity ** 2
        if this_energy < last_energy:
            return False
        last.end = note.start
        return True
    
    def get(self):
        return self.piano
