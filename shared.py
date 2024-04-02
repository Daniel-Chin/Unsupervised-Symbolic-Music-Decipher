import numpy as np
import socket

import init as _
from paths import *
from torchwork_mini import *
from domestic_typing import *

SEC_PER_DATAPOINT = 20
ENCODEC_SR = 32000
ENCODEC_N_BOOKS = 4
ENCODEC_N_WORDS_PER_BOOK = 2048
ENCODEC_FPS = 50
N_TOKENS_PER_DATAPOINT = SEC_PER_DATAPOINT * ENCODEC_FPS

LA_DATASET_DIRS = [*'0123456789abcdef']

TWO_PI = np.pi * 2

def initMainProcess():
    print('hostname:', socket.gethostname())
    print(f'{GPU_NAME = }')
    print(f'{DEVICE.index = }')
