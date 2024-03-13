import os
from os import path

import init as _

PROJ_DIR = path.dirname(path.abspath(__file__))

DATA_DIR    = path.join(PROJ_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
INSPECT_DIR = path.join(DATA_DIR, 'inspect')
os.makedirs(INSPECT_DIR, exist_ok=True)

def getEnv(name: str):
    v = os.getenv(name)
    assert v is not None, f'Environment variable {name} is not set.'
    return path.abspath(v)
LA_MIDI_DIR = getEnv('LA_MIDI_PATH')

if __name__ == '__main__':
    for k, v in [*globals().items()]:
        if k.endswith('_DIR'):
            print(f'{k} = {v}')
