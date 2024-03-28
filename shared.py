import torch
import numpy as np

import init as _
from paths import *

TWO_PI = np.pi * 2

SEC_PER_DATAPOINT = 30
ENCODEC_SR = 32000
ENCODEC_N_BOOKS = 4
ENCODEC_FPS = 50
N_TOKENS_PER_DATAPOINT = SEC_PER_DATAPOINT * ENCODEC_FPS

HAS_CUDA = torch.cuda.is_available()
CUDA = torch.device("cuda:0")
CPU  = torch.device("cpu")
# print(f'{__name__ = }')
if HAS_CUDA:
    DEVICE = CUDA
    print('We have CUDA.', flush=True)
    gpu_name = torch.cuda.get_device_name(DEVICE)
    print(f'{gpu_name = }', flush=True)
else:
    DEVICE = CPU
    print("We DON'T have CUDA.", flush=True)
