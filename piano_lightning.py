import os
from functools import lru_cache

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import lightning as L
from lightning.pytorch.callbacks import DeviceStatsMonitor, ModelSummary
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler

from shared import *
from hparams import HParamsPiano
from music import PIANO_RANGE
from piano_model import PianoModel
from piano_dataset import PianoDataset, BatchType

MONKEY_VAL = 'VAL_MONKEY'
ORACLE_VAL = 'VAL_ORACLE'
VAL_CASES = [MONKEY_VAL, ORACLE_VAL]

class LitPiano(L.LightningModule):
    def __init__(self, hParams: HParamsPiano) -> None:
        super().__init__()
        self.hP = hParams
        writeLightningHparams(hParams, self, hParams.require_repo_working_tree_clean)
        example_batch_size = 3
        self.example_input_array = torch.randn(
            (
                example_batch_size, 2, 
                PIANO_RANGE[1] - PIANO_RANGE[0], N_FRAMES_PER_DATAPOINT, 
            ), 
        )

        self.did_setup: bool = False
    
    def log_(self, *a, **kw):
        hParams = self.hP
        return super().log(*a, batch_size=hParams.batch_size, **kw)
    
    def setup(self, stage: str):
        _ = stage
        assert not self.did_setup
        self.did_setup = True

        hParams = self.hP
        self.piano = PianoModel.new(hParams)

        # just for ModelSummary
        # self.convs = self.cnnPiano.convs
        # self.outProjector = self.cnnPiano.outProjector
    
    def forward(
        self, x: Tensor, 
    ):
        return self.piano.forward(x)
    
    def training_step(
        self, batch: BatchType, batch_idx: int, 
    ):
        _ = batch_idx
        x, y, _ = batch

        y_logits = self.forward(x)
        loss = F.cross_entropy(
            y_logits.reshape(-1, ENCODEC_N_WORDS_PER_BOOK), 
            y       .view   (-1).to(torch.long), 
        )
        self.log_('train_loss', loss)

        for book_i, accuracy in enumerate(
            (y_logits.argmax(dim=-1) == y).float().mean(dim=2).mean(dim=0), 
        ):
            self.log_(f'train_accuracy_book_{book_i}', accuracy, on_step=False, on_epoch=True)

        return loss
    
    def validation_step(
        self, batch: BatchType, 
        batch_idx: int, dataloader_idx: int, 
    ):
        if not self.hP.do_validate:
            return

        _ = batch_idx
        x, y, _ = batch

        y_logits = self.forward(x)

        def logName(x: str, /):
            return f'{VAL_CASES[dataloader_idx]}_{x}'

        assert y.shape == y_logits.shape[:-1], (y.shape, y_logits.shape)
        loss = F.cross_entropy(
            y_logits.reshape(-1, ENCODEC_N_WORDS_PER_BOOK), 
            y       .view   (-1).to(torch.long), 
        )
        self.log_(logName('loss'), loss)

        for book_i, accuracy in enumerate(
            (y_logits.argmax(dim=-1) == y).float().mean(dim=2).mean(dim=0), 
        ):
            self.log_(logName(f'accuracy_book_{book_i}'), accuracy)
        
    def configure_optimizers(self):
        hParams = self.hP
        optim = torch.optim.Adam(
            self.piano.parameters(), lr=hParams.lr, 
        )
        sched = torch.optim.lr_scheduler.ExponentialLR(
            optim, gamma=hParams.lr_decay, 
        )
        return [optim], [sched]

    def on_before_optimizer_step(self, _: torch.optim.Optimizer):
        norms = grad_norm(self.piano, norm_type=2)
        key = 'grad_2.0_norm_total'
        self.log_(key, norms[key])

class LitPianoDataModule(L.LightningDataModule):
    def __init__(self, hParams: HParamsPiano) -> None:
        super().__init__()
        self.hP = hParams
        self.did_setup: bool = False

    def setup(self, stage: Optional[str] = None):
        _ = stage
        assert not self.did_setup
        self.did_setup = True

        hParams = self.hP

        @lru_cache(maxsize=1)
        def monkeyDataset():
            return PianoDataset(
                'monkey', PIANO_MONKEY_DATASET_DIR, 
                hParams.train_set_size + hParams.val_monkey_set_size, 
            )

        @lru_cache(maxsize=1)
        def oracleDataset():
            return PianoDataset(
                'oracle', PIANO_ORACLE_DATASET_DIR, 
                hParams.val_oracle_set_size,
            )
        
        self.train_dataset, self.val_monkey_dataset = random_split(
            monkeyDataset(), [
                hParams.train_set_size, 
                hParams.val_monkey_set_size, 
            ], 
        )
        self.val_oracle_dataset = oracleDataset()
        self.val_sets = {
            MONKEY_VAL: self.val_monkey_dataset, 
            ORACLE_VAL: self.val_oracle_dataset,
        }
    
    def train_dataloader(self, batch_size: Optional[int] = None, shuffle: bool = True):
        hParams = self.hP
        bs = batch_size or hParams.batch_size
        return DataLoader(
            self.train_dataset, batch_size=bs, 
            shuffle=shuffle, 
            num_workers=2, persistent_workers=True, 
        )
    
    def val_dataloader(self, batch_size: Optional[int] = None):
        hParams = self.hP
        bs = batch_size or hParams.batch_size
        return [
            DataLoader(
                self.val_sets[x], batch_size=bs, 
                num_workers=2, persistent_workers=True, 
            )
            for x in VAL_CASES
        ]

def train(hParams: HParamsPiano, root_dir: str):
    log_name = '.'
    os.makedirs(path.join(root_dir, log_name))
    litPiano = LitPiano(hParams)
    profiler = SimpleProfiler(filename='profile')
    logger = TensorBoardLogger(root_dir, log_name)
    # torch.cuda.memory._record_memory_history(max_entries=100000)
    trainer = L.Trainer(
        devices=[DEVICE.index], max_epochs=hParams.max_epochs, 
        gradient_clip_val=5.0, 
        default_root_dir=root_dir,
        logger=logger, 
        profiler=profiler, 
        callbacks=[
            # DeviceStatsMonitor(), 
            ModelSummary(max_depth=3), 
        ], 
        log_every_n_steps=min(50, hParams.train_set_size // hParams.batch_size), 
        # overfit_batches=1, 
    )
    dataModule = LitPianoDataModule(hParams)
    trainer.fit(litPiano, dataModule)
    # torch.cuda.memory._dump_snapshot(path.join(root_dir, 'VRAM.pickle'))
    # torch.cuda.memory._record_memory_history(enabled=None) # type: ignore

    return litPiano, dataModule

if __name__ == '__main__':
    import IPython; IPython.embed()
