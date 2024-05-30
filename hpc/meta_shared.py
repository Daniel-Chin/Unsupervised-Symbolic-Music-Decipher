'''
Doesn't require the main env.  
'''

from enum import Enum
import socket

class AVH_Stage(Enum):
    piano = 'piano'
    decipher = 'decipher'

compute_node = socket.gethostname().split('.', 1)[0]

if compute_node.startswith('hpclogin'):
    on_low_not_high = False
elif compute_node.startswith('login'):
    on_low_not_high = True
else:
    raise ValueError(f'{compute_node = }')
