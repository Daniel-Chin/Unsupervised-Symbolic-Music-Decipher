import dataclasses
from functools import lru_cache
from datetime import datetime

import torch
import git
import lightning as L

from domestic_typing import *

__all__ = [
    'HAS_CUDA', 'CUDA', 'CPU', 'DEVICE', 'GPU_NAME', 
    'getParams', 'getGradNorm', 'getCommitHash', 
    'writeLightningHparams', 'currentTimeDirName', 
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
