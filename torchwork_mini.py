import dataclasses
from functools import lru_cache
from datetime import datetime

import torch
import git
import lightning as L

from domestic_typing import *

__all__ = [
    'HAS_CUDA', 'CUDA', 'CPU', 'DEVICE', 'getParams', 'getGradNorm', 
    'getCommitHashAndAssertWorkingTreeClean', 'writeLightningHparams', 
    'currentTimeDirName', 
]

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
def getCommitHashAndAssertWorkingTreeClean():
    repo = git.Repo('.', search_parent_directories=True)
    assert not repo.is_dirty()
    return next(repo.iter_commits()).hexsha

def writeLightningHparams(dataclassObject, litModule: L.LightningModule):
    for k, v in dataclasses.asdict(dataclassObject).items():
        litModule.save_hyperparameters(k, v)
    litModule.save_hyperparameters('commit_hash', getCommitHashAndAssertWorkingTreeClean())

def currentTimeDirName():
    # file system friendly
    return datetime.now().strftime('%Y_m%m_d%d@%H_%M_%S')
