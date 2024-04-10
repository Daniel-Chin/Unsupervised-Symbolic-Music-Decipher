import numpy as np

PIANO_RANGE = (21, 109) # excluding 109

def pitch2freq(pitch: float) -> float:
    return np.exp((pitch + 36.37631656229591) * 0.0577622650466621)

def freq2pitch(f: float) -> float:
    return np.log(f) * 17.312340490667562 - 36.37631656229591
