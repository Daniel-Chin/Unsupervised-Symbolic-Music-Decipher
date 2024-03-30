import os
from functools import lru_cache

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import lightning as L
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.profilers import SimpleProfiler

from shared import *
from hparams import HParams
from tf_piano_model import TFPiano, KeyEventEncoder, TransformerPianoModel
from tf_piano_dataset import TransformerPianoDataset, collate

MONKEY_VAL = 'MONKEY_VAL'
ORACLE_VAL = 'ORACLE_VAL'
VAL_CASES = [MONKEY_VAL, ORACLE_VAL]

class LitPiano(L.LightningModule):
    def __init__(self, hParams: HParams) -> None:
        super().__init__()
        self.hP = hParams
        writeLightningHparams(hParams, self)
        self.example_input_array = (
            torch.zeros((hParams.batch_size, 233, 1 + 1 + 88)), 
            [200] * hParams.batch_size, 
        )
    
    def log(self, *a, **kw):
        hParams = self.hP
        return super().log(*a, batch_size=hParams.batch_size, **kw)
    
    def setup(self, stage: str):
        print('lit module setup', stage)
        hParams = self.hP
        keyEventEncoder = KeyEventEncoder(
            hParams.d_model, 
            hParams.key_event_encoder_n_layers,
            hParams.key_event_encoder_d_hidden, 
        )
        transformerPianoModel = TransformerPianoModel(
            hParams.d_model, hParams.tf_piano_n_head,
            hParams.tf_piano_n_encoder_layers, 
            hParams.tf_piano_n_decoder_layers,
            hParams.tf_piano_d_feedforward, 
        )
        self.tfPiano = TFPiano(keyEventEncoder, transformerPianoModel)
    
    def forward(self, x: Tensor, x_lens: List[int]) -> Tensor:
        return self.tfPiano.forward(x, x_lens)
    
    def training_step(
        self, batch: Tuple[Tensor, Tensor, List[int]], batch_idx: int, 
    ):
        x, y, x_lens = batch
        y_hat = self.tfPiano.forward(x, x_lens)
        loss = F.cross_entropy(
            y_hat.view(-1, ENCODEC_N_WORDS_PER_BOOK), 
            y    .view(-1), 
        )
        self.log('train_loss', loss)

        for book_i, accuracy in enumerate(
            (y_hat.argmax(dim=-1) == y).float().mean(dim=2).mean(dim=0), 
        ):
            self.log(f'train_accuracy_book_{book_i}', accuracy, on_step=False, on_epoch=True)

        return loss
    
    def validation_step(
        self, batch: Tuple[Tensor, Tensor, List[int]], 
        batch_idx: int, dataloader_idx: int, 
    ):
        def log(name: str, value: float | int | Tensor):
            self.log(f'{VAL_CASES[dataloader_idx]}_{name}', value)

        x, y, x_lens = batch
        y_hat = self.tfPiano.forward(x, x_lens)

        loss = F.cross_entropy(
            y_hat.view(-1, ENCODEC_N_WORDS_PER_BOOK), 
            y    .view(-1), 
        )
        log('loss', loss)

        for book_i, accuracy in enumerate(
            (y_hat.argmax(dim=-1) == y).float().mean(dim=2).mean(dim=0), 
        ):
            log(f'accuracy_book_{book_i}', accuracy)
    
    def configure_optimizers(self):
        hParams = self.hP
        return torch.optim.Adam(self.tfPiano.parameters(), lr=hParams.lr)

class LitPianoDataModule(L.LightningDataModule):
    def __init__(self, hParams: HParams) -> None:
        super().__init__()
        self.hP = hParams
    
    def setup(self, stage: Optional[str] = None):
        print('data module setup', stage)

        @lru_cache(maxsize=1)
        def monkeyDataset():
            return TransformerPianoDataset(
                'monkey', TRANSFORMER_PIANO_MONKEY_DATASET_DIR, 
            )

        @lru_cache(maxsize=1)
        def oracleDataset():
            return TransformerPianoDataset(
                'oracle', TRANSFORMER_PIANO_ORACLE_DATASET_DIR, 
            )
        
        self.train_dataset, self.val_monkey_dataset = random_split(
            monkeyDataset(), [.8, .2], 
        )
        self.val_oracle_dataset = oracleDataset()
    
    def train_dataloader(self):
        hParams = self.hP
        return DataLoader(
            self.train_dataset, batch_size=hParams.batch_size, 
            collate_fn=collate, shuffle=True, 
            num_workers=2, persistent_workers=True, 
        )
    
    def val_dataloader(self):
        hParams = self.hP
        return [
            DataLoader(
                self.val_monkey_dataset, batch_size=hParams.batch_size, 
                collate_fn=collate, 
                num_workers=2, persistent_workers=True, 
            ),
            DataLoader(
                self.val_oracle_dataset, batch_size=hParams.batch_size, 
                collate_fn=collate, 
                num_workers=2, persistent_workers=True, 
            ),
        ]

def train(hParams: HParams, root_dir: str):
    os.makedirs(path.join(root_dir, 'lightning_logs'))
    if GPU_NAME == 'NVIDIA GeForce RTX 3050 Ti Laptop GPU':
        torch.set_float32_matmul_precision('high')
    litPiano = LitPiano(hParams)
    profiler = SimpleProfiler(filename='profile.txt')
    trainer = L.Trainer(
        devices=[DEVICE.index], max_epochs=hParams.max_epochs, 
        default_root_dir=root_dir,
        profiler=profiler, callbacks=[DeviceStatsMonitor()], 
    )
    trainer.fit(litPiano, LitPianoDataModule(hParams))
    return litPiano
