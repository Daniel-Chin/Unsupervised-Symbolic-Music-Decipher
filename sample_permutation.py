import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx
from torch.distributions.categorical import Categorical

from sample_with_ste_backward import SampleWithSTEBackward

class SamplePermutation(SampleWithSTEBackward):
    @staticmethod
    def forward(_: FunctionCtx, probs: Tensor, n: int):
        '''
        `probs` shape: (n_keys, n_pitches), simplex: over dim 0  
        returns shape: (n_keys, n, n_pitches)  
        where n_keys == n_pitches  
        '''
        # this current implementation is a biased estimate. Awaiting shouchang
        n_keys, n_pitches = probs.shape
        assert n_keys == n_pitches
        entropy = Categorical(probs=probs.T).entropy()
        out = torch.zeros((
            n_keys, n, n_pitches, 
        ), dtype=probs.dtype, device=probs.device)
        for n_i in range(n):
            available_key_mask = torch.ones((n_keys, ), device=probs.device)
            for pitch_i, _ in sorted(enumerate(entropy), key=lambda x: x[1]):
                simplex = probs[:, pitch_i]
                remaining = simplex * available_key_mask
                key_i = Categorical(probs = remaining / remaining.sum()).sample()
                key_int: int = key_i.item() # type: ignore
                available_key_mask[key_int] = 0.0
                out[key_i, n_i, pitch_i] = 1.0
        return out

def samplePermutation(probs: Tensor, n: int) -> Tensor:
    return SamplePermutation.apply(probs, n) # type: ignore
