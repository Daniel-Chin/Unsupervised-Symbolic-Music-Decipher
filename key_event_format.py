from dataclasses import dataclass

@dataclass(frozen=True)
class Chunk:
    start: int
    length: int
    @property
    def end(self): 
        return self.start + self.length

class KeyEventFormat:
    def __init__(self, onset_as_positional_encoding: bool, d_model: int):
        self.onset_as_positional_encoding = onset_as_positional_encoding
        cursor = 0
        if onset_as_positional_encoding:
            self.onset = Chunk(0, d_model)
        else:
            self.onset = Chunk(0, 1)
        cursor += self.onset.length
        self.velocity = Chunk(cursor, 1)
        cursor += self.velocity.length
        self.key = Chunk(cursor, 88)
        self.length = self.key.end
