import dataclasses
from functools import lru_cache
from datetime import datetime
from itertools import count

import torch
from torch import Tensor
import git
import lightning as L
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from domestic_typing import *

__all__ = [
    'HAS_CUDA', 'CUDA', 'CPU', 'DEVICE', 'GPU_NAME', 
    'getParams', 'getGradNorm', 'getCommitHash', 
    'writeLightningHparams', 'currentTimeDirName', 
    'positionalEncoding', 'positionalEncodingAt',
]

HAS_CUDA = torch.cuda.is_available()
CUDA = torch.device("cuda:0")
CPU  = torch.device("cpu")
if HAS_CUDA:
    DEVICE = CUDA
    # print('We have CUDA.', flush=True)
    GPU_NAME = torch.cuda.get_device_name(DEVICE)
    # print(f'{GPU_NAME = }', flush=True)
else:
    DEVICE = CPU
    # print("We DON'T have CUDA.", flush=True)
    GPU_NAME = None

def getParams(optim: torch.optim.Optimizer):
    s: List[torch.Tensor] = []
    for param_group in optim.param_groups:
        for param in param_group['params']:
            param: torch.Tensor
            if param.grad is not None:
                s.append(param)
    return s

def getGradNorm(params: List[torch.Tensor]):
    buffer = torch.zeros((len(params), ), device=DEVICE)
    for i, param in enumerate(params):
        grad = param.grad
        assert grad is not None
        buffer[i] = grad.norm(2)
    return buffer.norm(2).cpu()

@lru_cache(maxsize=1)
def getCommitHash(do_assert_working_tree_clean: bool = False):
    repo = git.Repo('.', search_parent_directories=True)
    if do_assert_working_tree_clean:
        assert not repo.is_dirty()
    return next(repo.iter_commits()).hexsha

def writeLightningHparams(
    dataclassObject, litModule: L.LightningModule, 
    do_assert_working_tree_clean: bool = False, 
):
    litModule.save_hyperparameters(dataclasses.asdict(dataclassObject))
    litModule.save_hyperparameters(dict(
        commit_hash = getCommitHash(do_assert_working_tree_clean), 
    ))

def currentTimeDirName():
    # file system friendly
    return datetime.now().strftime('%Y_m%m_d%d@%H_%M_%S')

@lru_cache()
def positionalEncoding(length: int, d_model: int, device: torch.device) -> Tensor:
    MAX_LEN = 10000
    assert length < MAX_LEN

    pe = torch.zeros(length, d_model, device=device)
    ladder = torch.arange(length, dtype=pe.dtype, device=device)
    for i in count():
        try:
            pe[:, 2 * i    ] = torch.sin(ladder / (MAX_LEN ** (2 * i / d_model)))
            pe[:, 2 * i + 1] = torch.cos(ladder / (MAX_LEN ** (2 * i / d_model)))
        except IndexError:
            break
    return pe

def positionalEncodingAt(
    pos: Tensor, length: int, d_model: int, device: torch.device, 
):
    # element in pos: [0.0, 1.0]
    pe = positionalEncoding(length, d_model, device)
    idx = pos.clamp_max(1.0 - 1e-6) * (length - 1)
    left = torch.floor(idx)
    right = left + 1.0
    left_k = right - idx
    right_k = idx - left
    left_pe  = pe[left .to(torch.int), :]
    right_pe = pe[right.to(torch.int), :]
    return left_k.unsqueeze(1) * left_pe + right_k.unsqueeze(1) * right_pe

def __inspectPositionalEncoding():
    pe = positionalEncoding(100, 32, CPU).numpy()
    plt.imshow(pe, aspect='auto', interpolation='nearest')
    plt.colorbar()
    plt.xlabel('embedding dim')
    plt.ylabel('time step')
    plt.show()
    for i in (0, 1, 16, 17, 30, 31):
        plt.plot(pe[:, i])
        plt.title(f'embedding dim {i}')
        plt.show()

def __inspectPositionalEncodingAt():
    pe = positionalEncoding(20, 32, CPU).numpy()
    t = torch.linspace(0.0, 1.0, 100)
    pea = positionalEncodingAt(t, 20, 32, CPU).numpy()
    fig, axes = plt.subplots(2, 1)
    for x, ax, title in zip((pe, pea), axes, ('pe', 'pe at')):
        ax: Axes
        ax.imshow(x, aspect='auto', interpolation='nearest')
        ax.set_xlabel('embedding dim')
        ax.set_ylabel('time step')
        ax.set_title(title)
    plt.show()

if __name__ == '__main__':
    __inspectPositionalEncoding()
    __inspectPositionalEncodingAt()
