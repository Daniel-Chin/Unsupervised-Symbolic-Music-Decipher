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
from hparams import HParams
from tf_piano_model import TFPiano, KeyEventEncoder, TransformerPianoModel
from tf_piano_dataset import TransformerPianoDataset, collate, CollateFnOut

MONKEY_VAL = 'MONKEY_VAL'
ORACLE_VAL = 'ORACLE_VAL'
VAL_CASES = [MONKEY_VAL, ORACLE_VAL]

class LitPiano(L.LightningModule):
    def __init__(self, hParams: HParams) -> None:
        super().__init__()
        self.hP = hParams
        writeLightningHparams(hParams, self, hParams.require_repo_working_tree_clean)
        self.example_input_array = (
            torch.zeros((hParams.batch_size, 233, hParams.keyEventFormat().length)), 
            torch.full((hParams.batch_size, 233), False), 
        )

        self.did_setup: bool = False
    
    def log_(self, *a, **kw):
        hParams = self.hP
        return super().log(*a, batch_size=hParams.batch_size, **kw)
    
    def setup(self, stage: str):
        # Because I've no idea lightning's setup stage logic
        assert stage == TrainerFn.FITTING
        # print('lit module setup', stage)
        assert not self.did_setup
        self.did_setup = True

        hParams = self.hP
        keyEventEncoder = KeyEventEncoder(
            hParams.keyEventFormat().length,
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
    
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        return self.tfPiano.forward(x, mask)
    
    def training_step(
        self, batch: CollateFnOut, batch_idx: int, 
    ):
        x, y, mask, _ = batch
        y_hat = self.tfPiano.forward(x, mask)
        loss = F.cross_entropy(
            y_hat.view(-1, ENCODEC_N_WORDS_PER_BOOK), 
            y    .view(-1), 
        )
        self.log_('train_loss', loss)

        for book_i, accuracy in enumerate(
            (y_hat.argmax(dim=-1) == y).float().mean(dim=2).mean(dim=0), 
        ):
            self.log_(f'train_accuracy_book_{book_i}', accuracy, on_step=False, on_epoch=True)

        return loss
    
    def validation_step(
        self, batch: CollateFnOut, 
        batch_idx: int, dataloader_idx: int, 
    ):
        def log(name: str, value: float | int | Tensor):
            self.log_(f'{VAL_CASES[dataloader_idx]}_{name}', value)

        x, y, mask, _ = batch
        y_hat = self.tfPiano.forward(x, mask)

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

    def on_before_optimizer_step(self, _: torch.optim.Optimizer):
        norms = grad_norm(self, norm_type=2)
        key = 'grad_2.0_norm_total'
        self.log_(key, norms[key])

class LitPianoDataModule(L.LightningDataModule):
    def __init__(self, hParams: HParams) -> None:
        super().__init__()
        self.hP = hParams
        self.did_setup: bool = False

    def setup(self, stage: Optional[str] = None):
        # Because I've no idea lightning's setup stage logic
        assert stage == TrainerFn.FITTING
        # print('data module setup', stage)
        assert not self.did_setup
        self.did_setup = True

        hParams = self.hP

        @lru_cache(maxsize=1)
        def monkeyDataset():
            return TransformerPianoDataset(
                'monkey', TRANSFORMER_PIANO_MONKEY_DATASET_DIR, 
                hParams.keyEventFormat(), 
                hParams.tf_piano_train_set_size + hParams.tf_piano_val_monkey_set_size, 
            )

        @lru_cache(maxsize=1)
        def oracleDataset():
            return TransformerPianoDataset(
                'oracle', TRANSFORMER_PIANO_ORACLE_DATASET_DIR, 
                hParams.keyEventFormat(), 
                hParams.tf_piano_val_oracle_set_size,
            )
        
        self.train_dataset, self.val_monkey_dataset = random_split(
            monkeyDataset(), [
                hParams.tf_piano_train_set_size, 
                hParams.tf_piano_val_monkey_set_size, 
            ], 
        )
        self.val_oracle_dataset = oracleDataset()
    
    def train_dataloader(self, shuffle=True):
        hParams = self.hP
        return DataLoader(
            self.train_dataset, batch_size=hParams.batch_size, 
            collate_fn=collate, shuffle=shuffle, 
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
    log_name = '.'
    os.makedirs(path.join(root_dir, log_name))
    if GPU_NAME in (
        'NVIDIA GeForce RTX 3050 Ti Laptop GPU', 
        'NVIDIA GeForce RTX 3090', 
    ):
        torch.set_float32_matmul_precision('high')
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
            # ModelSummary(max_depth=2), # Internal error: NestedTensorImpl doesn't support sizes.
        ], 
        log_every_n_steps=min(50, hParams.tf_piano_train_set_size // hParams.batch_size), 
    )
    dataModule = LitPianoDataModule(hParams)
    trainer.fit(litPiano, dataModule)
    # torch.cuda.memory._dump_snapshot(path.join(root_dir, 'VRAM.pickle'))
    # torch.cuda.memory._record_memory_history(enabled=None) # type: ignore

    return litPiano, dataModule

if __name__ == '__main__':
    import IPython; IPython.embed()
