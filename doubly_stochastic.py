import torch
from torch import Tensor

from my import profileDuration

DIMS = (1, 0)   # strictly simplex along dim 0

@torch.no_grad()
def sinkhornKnopp(x: Tensor, /):
    x.clamp_min_(0.0)
    has_converged = [False] * len(DIMS)
    while not all(has_converged):
        for dim in DIMS:
            s = x.sum(dim=dim, keepdim=True)
            x.mul_(1 / s)
            h_c = (s.log().abs() < 1e-3).all().item()
            has_converged[dim] = h_c    # type: ignore

def study():
    # conclusion: doubly stochastic, collapsing one row, is not easily doubly stochastic any more
    # related: a dist over perms is insufficiently parametrized by the frequency matrix
    sinkhornKnopp_ = profileDuration()(sinkhornKnopp)
    d_s = torch.randn((
        3, 3, # minimum case. 2, 2 is parametrized by frequency matrix.
        # 88, 88, # compute time: ms-level
    ))
    d_s = d_s.softmax(dim=0)
    sinkhornKnopp_(d_s)
    def check():
        print(f'{d_s.sum(dim=0) = }')
        print(f'{d_s.sum(dim=1) = }')
    check()
    x, y = 0, 1
    d_s[x, :] = 0.0
    d_s[:, y] = 0.0
    d_s[x, y] = 1.0
    d_s = d_s / d_s.sum(dim=0)
    check()
    sinkhornKnopp_(d_s)

if __name__ == '__main__':
    study()
