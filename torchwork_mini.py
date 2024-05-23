import os
from os import path
from functools import lru_cache
from datetime import datetime
from itertools import count
import json
from queue import Queue, Empty
from threading import Thread
import math
import typing
import time

import torch
from torch import Tensor
from torch.utils.data import default_collate, Dataset
import git
from lightning.pytorch.loggers import Logger
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
import uuid

from domestic_typing import *

__all__ = [
    'HAS_CUDA', 'CUDA', 'CPU', 'DEVICE', 'GPU_NAME', 
    'getParams', 'getGradNorm', 'getCommitHash', 
    'logJobMeta', 'currentTimeDirName', 
    'positionalEncoding', 'positionalEncodingAt',
    'tensorCacheAndClone', 'freeze', 'collateWithNone', 
    'colorBar', 'SingleProcessNewThreadPreFetchDataLoader', 
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

def logJobMeta(
    logger: Logger, 
    do_assert_working_tree_clean: bool = False, 
):
    d = dict(
        commit_hash = getCommitHash(do_assert_working_tree_clean), 
        slurm_job_id = os.environ.get('SLURM_JOB_ID'),
    )
    assert logger.log_dir is not None
    os.makedirs(logger.log_dir, exist_ok=True)
    with open(path.join(logger.log_dir, 'job_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=2)

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

def tensorCacheAndClone(*a, **kw):
    '''
    Just like lru_cache, except the returned tensor may be mutated without affecting the cache.
    '''
    try:
        if callable(a[0]):
            raise TypeError('Sorry, the non-calling shorthand is not supported. Use @tensorCacheAndClone(maxsize) instead.')
    except IndexError:
        pass
    lruCache = lru_cache(*a, **kw)
    def cache(func: Callable[..., Tensor]):
        vanilla = lruCache(func)
        def f(*a, **kw):
            return vanilla(*a, **kw).clone()
        return f
    return cache

def testCache():
    from time import sleep

    def slow():
        sleep(1)
        return torch.zeros((2, 3))
    
    for name, subject in (
        ('vanilla', lru_cache()(slow)), 
        ('ours', tensorCacheAndClone()(slow)),
    ):
        print(name)
        input('Enter...')
        a = subject()
        print(a)
        a[0, 0] = 69
        b = subject()
        print(b)

def freeze(module: torch.nn.Module):
    for param in module.parameters():
        param.requires_grad = False

def collateWithNone(datapoints: List[Tuple[Optional[Tensor]]]):
    non_none_indices = [i for (i, x) in enumerate(datapoints[0]) if x is not None]
    squeezed = [[
        x[i] for i in non_none_indices
    ] for x in datapoints]
    squeezed_batch: Iterable[Tensor] = default_collate(squeezed)
    batch = []
    i_squeezed_batch = iter(squeezed_batch)
    for i in range(len(datapoints[0])):
        if i in non_none_indices:
            batch.append(next(i_squeezed_batch))
        else:
            batch.append(None)
    return tuple(batch)

def colorBar(fig: Figure, ax: Axes, im: AxesImage):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

class GeneratorWithLen:
    def __init__(self, g: typing.Generator, size: int):
        self.g = g
        self.size = size
    
    @staticmethod
    def decorate(size: int):
        def decorator(G: Callable[[], typing.Generator]):
            def decorated(*a, **kw):
                return __class__(G(*a, **kw), size)
            return decorated
        return decorator
    
    def __len__(self):
        return self.size
    
    def __next__(self):
        return next(self.g)
    
    def __iter__(self):
        return self

class SingleProcessNewThreadPreFetchDataLoader:
    # supports map-style datasets
    def __init__(
        self, dataset: Dataset, batch_size: int, shuffle: bool, 
        collate_fn: Callable = collateWithNone, prefetch_factor: int = 2, 
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn
        self.prefetch_factor = prefetch_factor

        self.dataset_size = len(self.dataset)    # type: ignore
        self.n_batches = math.ceil(self.dataset_size / batch_size)
    
    def __iter__(self):
        return SingleProcessNewThreadPreFetchDataLoaderIter(self)
    
    def __len__(self):
        return self.n_batches

class SingleProcessNewThreadPreFetchDataLoaderIter:
    def __init__(self, l: SingleProcessNewThreadPreFetchDataLoader):
        self.l = l

        self.q = Queue()
        self.name = uuid.uuid4()
        self.is_deleted = False
        self.thread = Thread(target=self.worker)
        self.thread.start()
        self.g = self.Generator()

    def __len__(self):
        return self.l.n_batches
    
    def __next__(self):
        return next(self.g)
    
    def worker(self):
        try:
            if self.l.shuffle:
                indices = torch.randperm(self.l.dataset_size)
            else:
                indices = torch.arange(self.l.dataset_size)
            cursor = 0
            while cursor < self.l.dataset_size:
                batch_indices = indices[cursor : cursor + self.l.batch_size]
                # len(indices) <= self.batch_size
                batch = self.l.collate_fn([
                    self.l.dataset[i] for i in batch_indices
                ])
                assert batch is not None
                # self.debug('put(batch)...')
                self.q.put(batch)
                # self.debug('put(batch) ok')
                if self.is_deleted:
                    break
                cursor += self.l.batch_size
        finally:
            # self.debug('put(None)...')
            self.q.put(None)
            # self.debug('put(None) ok')
    
    def Generator(self):
        while True:
            # self.debug('get()...')
            batch = self.q.get()
            # self.debug('get() ok')
            if batch is None:
                # self.debug('batch is None')
                break
            time.sleep(0)   # yield control to worker. Query disk ASAP, then hardware blocking will yield control back to main thread.
            yield batch
        # self.debug('join()...')
        self.thread.join()
        # self.debug('join() ok')
    
    def __del__(self):
        # self.debug('__del__()...')
        self.is_deleted = True
        try:
            self.q.get_nowait()
            self.q.get_nowait()
        except Empty:
            pass
        # self.debug('__del__() 1')
        self.thread.join()
        # self.debug('__del__() ok')
    
    def debug(self, *a, **kw):
        print('\n', self.name, *a, **kw)

if __name__ == '__main__':
    __inspectPositionalEncoding()
    __inspectPositionalEncodingAt()
    testCache()
