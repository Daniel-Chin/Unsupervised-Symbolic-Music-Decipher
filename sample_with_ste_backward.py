import torch
from torch import Tensor
import torch.autograd.gradcheck
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.autograd.function import FunctionCtx, once_differentiable
from torch.autograd.gradcheck import gradcheck

class SampleWithSTEBackward(torch.autograd.Function):
    '''
    Forward: Sample a simplex `probs` Tensor into a one-hot Tensor.  
    Backward: Straight-Through Estimation (STE).  
    A baseline of gumbel_softmax.
    '''
    @staticmethod
    def forward(_: FunctionCtx, probs: Tensor, n: int):
        '''
        `probs` shape: (batch_size, n_classes)
        returns shape: (batch_size, n, n_classes)
        '''
        c = Categorical(probs=probs)
        sampled = c.sample(torch.Size((n, )))
        return (sampled == c.enumerate_support().unsqueeze(1).expand(
            -1, n, -1, 
        )).float().permute(2, 1, 0)

    @once_differentiable
    @staticmethod
    def backward(_: FunctionCtx, grad_output: Tensor):
        return grad_output.sum(dim=1), None

def sampleWithSTEBackward(probs: Tensor, n: int) -> Tensor:
    return SampleWithSTEBackward.apply(probs, n) # type: ignore

def checkAccuracy():
    batch_size = 2
    n_classes = 3
    kk = 1
    while True:
        kk *= 10
        print(f'{kk = }')
        class SampleWithSTEBackwardCheckable(SampleWithSTEBackward):
            @staticmethod
            def forward(ctx: FunctionCtx, probs: Tensor):
                n = 3
                return SampleWithSTEBackward.forward(ctx, probs, n * kk).view(batch_size, n, kk, n_classes).mean(dim=2)
            
        def swsb(logits: Tensor) -> Tensor:
            probs = F.softmax(logits, dim=1)
            return SampleWithSTEBackwardCheckable.apply(probs)  # type: ignore
        try:
            test = gradcheck(
                swsb, torch.randn((batch_size, n_classes), requires_grad=True, dtype=torch.double), 
                eps=1e-2, 
                raise_exception=True,
            )
        except Exception as e:
            print(e)
        else:
            print(f'{test = }')
            break

def checkDim():
    bs = 3
    n = 7
    logits = torch.randn((bs, 6), requires_grad=True)
    probs = F.softmax(logits, dim=-1)
    sampled = sampleWithSTEBackward(probs, n)
    print(f'{sampled.shape = }')
    sampled.sum().backward()
    assert logits.grad is not None
    print(f'{logits.grad.shape = }')

if __name__ == '__main__':
    checkDim()
    checkAccuracy()
