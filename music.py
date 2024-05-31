import numpy as np

PIANO_RANGE = (21, 109) # excluding 109

def pitch2freq(pitch: float) -> float:
    return np.exp((pitch + 36.37631656229591) * 0.0577622650466621)

def freq2pitch(f: float) -> float:
    return np.log(f) * 17.312340490667562 - 36.37631656229591

class NonDiatone(NotImplementedError): pass

def pitch2name(pitch: int):
    def f(i):
        if i % 2 != 0:
            raise NonDiatone()
        return i // 2
    chroma = pitch % 12
    octave = pitch // 12 - 1
    if chroma < 5:
        c = 'CDE'[f(chroma)]
    else:
        c = 'FGAB'[f(chroma - 5)]
    return f'{c}{octave}'
