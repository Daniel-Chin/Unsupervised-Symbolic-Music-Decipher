from dataclasses import dataclass

from modular_encoding import MODULAR_ENCODING_LEN

@dataclass(frozen=True)
class Chunk:
    start: int
    length: int
    @property
    def end(self): 
        return self.start + self.length

class KeyEventFormat:
    def __init__(
        self, onset_as_positional_encoding: bool, d_model: int, 
        key_as_modular_encoding: bool, 
        velocity_as_modular_encoding: bool, 
        is_modular_encoding_soft: bool, 
    ):
        self.onset_as_positional_encoding = onset_as_positional_encoding
        self.key_as_modular_encoding = key_as_modular_encoding
        self.velocity_as_modular_encoding = velocity_as_modular_encoding
        self.is_modular_encoding_soft = is_modular_encoding_soft
        cursor = 0
        if onset_as_positional_encoding:
            self.onset = Chunk(0, d_model)
        else:
            self.onset = Chunk(0, 1)
        cursor += self.onset.length
        if velocity_as_modular_encoding:
            self.velocity = Chunk(cursor, MODULAR_ENCODING_LEN)
        else:
            self.velocity = Chunk(cursor, 1)
        cursor += self.velocity.length
        if key_as_modular_encoding:
            self.key = Chunk(cursor, MODULAR_ENCODING_LEN)
        else:
            self.key = Chunk(cursor, 88)
        cursor += self.key.length
        self.length = cursor
