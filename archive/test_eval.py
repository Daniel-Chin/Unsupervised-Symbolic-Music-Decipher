import torch

class A(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class B(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

a = A()
b = B()

def f():
    x = torch.randn(1, 1)
    x = a.forward(x)
    x = b.forward(x)
    x.backward()

f()

f()

b.eval()
f()
b.train()

a.eval()
f()
