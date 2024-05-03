from os import path

from dotenv import load_dotenv
import torch

from torchwork_mini import *

PROJ_DIR = path.dirname(path.abspath(__file__))

load_dotenv(path.join(PROJ_DIR, 'active.env'))

if GPU_NAME in (
    'NVIDIA GeForce RTX 3050 Ti Laptop GPU', 
    'NVIDIA GeForce RTX 3090', 
):
    torch.set_float32_matmul_precision('high')
