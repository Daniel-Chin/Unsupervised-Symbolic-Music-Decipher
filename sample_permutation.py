import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx, once_differentiable
from torch.distributions.categorical import Categorical
from matplotlib import pyplot as plt
from tqdm import tqdm

from torchwork_mini import DEVICE

class SamplePermutation(torch.autograd.Function):
    @staticmethod
    def forward(
        functionCtx: FunctionCtx, 
        probs: Tensor, # already detached (by torch)
        n: int, 
        entropy_guided_sampling: bool = True,
    ):
        '''
        Samples a permutation matrix from a "mean permutation matrix" (MPM).  
        MPM insufficiently parametrizes a distribution of permutations.  
        I use an ad hoc method to sample the lowest entropy at each step.  
        This assumes some prior distribution, but I can't define it.  

        `probs` shape: (n_keys, n_pitches), simplex: over dim 0  
        returns shape: (n_keys, n, n_pitches)  
        where n_keys == n_pitches  
        '''
        _ = functionCtx
        n_keys, n_pitches = probs.shape
        assert n_keys == n_pitches
        device = probs.device
        LADDER = torch.arange(n, device=device)

        out = probs.unsqueeze(1).repeat(1, n, 1)
        available_pitch_mask = torch.ones((n, n_pitches), device=device)
        for i in range(n_pitches):
            if entropy_guided_sampling:
                entropy: Tensor = Categorical(probs=out.permute(1, 2, 0)).entropy()
                pitch_i = (entropy * available_pitch_mask).argmin(dim=1)
            else:
                pitch_i = torch.ones((n, ), dtype=torch.long, device=device) * i
            simplex = out[:, LADDER, pitch_i]
            key_i = Categorical(probs = simplex.T).sample()
            out[key_i, LADDER, :] = 0.0
            out[:, LADDER, pitch_i] = 0.0
            out[key_i, LADDER, pitch_i] = 1.0
            out.div_(out.sum(dim=0))
            available_pitch_mask[LADDER, pitch_i] = 0.0
        return out
    
    @once_differentiable
    @staticmethod
    def backward(_: FunctionCtx, grad_output: Tensor):
        return grad_output.sum(dim=1), None, None

def samplePermutation(probs: Tensor, n: int, entropy_guided_sampling: bool = True) -> Tensor:
    return SamplePermutation.apply(probs, n, entropy_guided_sampling) # type: ignore

def test():
    N = 1000
    D = 88
    K = 30
    probs = torch.randn((D, D), device=DEVICE).softmax(dim=0)
    for entropy_guided_sampling in (True, False):
        label = 'entropy guided' if entropy_guided_sampling else 'baseline'
        losses = []
        for _ in tqdm(range(K), desc=label):
            empirical = samplePermutation(
                probs, N, entropy_guided_sampling, 
            ).mean(dim=1)
            losses.append((probs - empirical).square().mean().item())
        plt.hist(losses, label=label)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    test()
