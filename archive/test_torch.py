import torch
from torch.nn.modules.transformer import Transformer
import lightning as L
from lightning.pytorch.callbacks import ModelSummary

print(torch.__version__)

class M(L.LightningModule):
    def __init__(self):
        super().__init__()
        tf=Transformer(16, 4, 3, 3, 16, batch_first=True)
        self.tf = tf
    
    def forward(self, src, tgt):
        return self.tf(src, tgt)

trainer = L.Trainer()
ModelSummary().on_fit_start(trainer, M())
