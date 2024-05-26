import dataclasses

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.distributions.categorical import Categorical
import lightning as L
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.callbacks import DeviceStatsMonitor, ModelSummary
from matplotlib import pyplot as plt

from shared import *
from music import PIANO_RANGE
from hparams import HParamsDecipher
from piano_dataset import PianoDataset, BatchType
from interpreter import Interpreter
from my_musicgen import myMusicGen, LMOutput
from sample_with_ste_backward import sampleWithSTEBackward
from piano_lightning import LitPiano

class LitDecipherDataModule(L.LightningDataModule):
    def __init__(self, hParams: HParamsDecipher) -> None:
        super().__init__()
        self.hP = hParams
        self.did_setup: bool = False

    def setup(self, stage: Optional[str] = None):
        _ = stage
        assert not self.did_setup
        self.did_setup = True

        hParams = self.hP

        dataset = PianoDataset(
            'oracle', PIANO_ORACLE_DATASET_DIR, 
            False, False, 
            hParams.train_set_size + hParams.val_set_size,
        )
        
        self.train_dataset, self.val_dataset = random_split(
            dataset, [
                hParams.train_set_size, 
                hParams.val_set_size, 
            ], 
        )
    
    def train_dataloader(self, batch_size: Optional[int] = None, shuffle: bool = True):
        hParams = self.hP
        bs = batch_size or hParams.batch_size
        return myChosenDataLoader(
            self.train_dataset, batch_size=bs, 
            shuffle=shuffle, 
        )
    
    def val_dataloader(self, batch_size: Optional[int] = None):
        hParams = self.hP
        bs = batch_size or hParams.batch_size
        return myChosenDataLoader(
            self.val_dataset, batch_size=bs, 
            shuffle=False,
        )

class LitDecipher(L.LightningModule):
    def __init__(self, **kw) -> None:
        super().__init__()
        self.save_hyperparameters()
        hParams = HParamsDecipher.fromDict(kw)
        self.hP = hParams
        example_batch_size = 3
        self.example_input_array = torch.randn(
            (
                example_batch_size, 2, 
                PIANO_RANGE[1] - PIANO_RANGE[0], N_FRAMES_PER_DATAPOINT, 
            ), 
        )

        def getPiano():
            checkpoint_path = hParams.getPianoAbsPaths()
            litPiano = LitPiano.load_from_checkpoint(checkpoint_path)
            litPiano.train()
            return litPiano.piano

        self.piano = getPiano()
        freeze(self.piano)
        self.interpreter = Interpreter(hParams)

        self.did_setup: bool = False

    def log_(self, *a, **kw):
        hParams = self.hP
        return super().log(*a, batch_size=hParams.batch_size, **kw)

    def setup(self, stage: str):
        _ = stage
        assert not self.did_setup
        self.did_setup = True

        hParams = self.hP

        assert self.logger is not None
        assert self.logger.log_dir is not None
        self.interpreter_visualized_dir = path.join(
            self.logger.log_dir, 'interpreter_visualized', 
        )
        os.makedirs(self.interpreter_visualized_dir)

    def forward(self, x: Tensor):
        '''
        `x` shape: (batch_size, 2, n_pitches, n_frames)  
        out shape: (batch_size, ENCODEC_N_BOOKS, n_frames, ENCODEC_N_WORDS_PER_BOOK)
        '''
        x = self.interpreter.forward(x)
        if DO_CHECK_NAN:
            assert not x.isnan().any(), pdb.set_trace()
        x = x.contiguous()
        x = self.piano.forward(x)
        return x

    def training_step(self, batch: BatchType, batch_idx: int):
        myMusicGen.train()
        return self.shared_step('train', batch, batch_idx)
    
    def validation_step(self, batch: BatchType, batch_idx: int):
        myMusicGen.eval()
        return self.shared_step('val', batch, batch_idx)
    
    def shared_step(
        self, step_name: str, batch: BatchType, batch_idx: int, 
    ):
        hParams = self.hP
        _ = batch_idx
        x, _, _, _ = batch

        encodec_tokens_logits = self.forward(x)
        (batch_size, n_books, n_frames, n_words_per_book) = encodec_tokens_logits.shape
        assert n_books == ENCODEC_N_BOOKS
        assert n_frames == N_FRAMES_PER_DATAPOINT
        assert n_words_per_book == ENCODEC_N_WORDS_PER_BOOK
        sampled_encodec_onehots = sampleWithSTEBackward(
            encodec_tokens_logits.reshape(
                batch_size * n_books * n_frames, n_words_per_book,
            ).softmax(dim=1), 
            n=1,
        ).unsqueeze(1).view(
            batch_size, n_books, n_frames, n_words_per_book,
        )
        prediction = myMusicGen.lmPredict(sampled_encodec_onehots)

        loss = torch.zeros(( ), device=self.device)
        def logLoss(name: Optional[str], loss: Tensor):
            suffix = '' if name is None else '_' + name
            self.log_(f'{step_name}_loss' + suffix, loss)
        if hParams.loss_weight_left != 0.0:
            loss_left, ce_per_codebook = myMusicGen.lmLoss(
                prediction, 
                sampled_encodec_onehots.argmax(dim=-1), 
            )
            logLoss('left', loss_left)
            loss += hParams.loss_weight_left  * loss_left
            for k, ce_k in enumerate(ce_per_codebook):
                logLoss(f'left_codebook_{k}', ce_k)
        if hParams.loss_weight_right != 0.0:
            loss_right = self.lossRight(encodec_tokens_logits, prediction)
            logLoss('right', loss_right)
            loss += hParams.loss_weight_right * loss_right
        logLoss(None, loss)
        return loss
    
    @staticmethod
    def lossRight(performed: Tensor, lmOutput: LMOutput):
        mask = lmOutput.mask    # mask with False: invalid, True: valid.  
        valid_logits = lmOutput.logits[mask, :] # (B*K*T', ENCODEC_N_WORDS_PER_BOOK)
        valid_performed = performed[mask, :] # (B*K*T', ENCODEC_N_WORDS_PER_BOOK)
        c = Categorical(logits=valid_logits)
        sampled = c.sample()    # grad is lost, and that's just right
        # `sampled` shape: (B*K*T', )
        onehots = (sampled == c.enumerate_support()).float().T
        return F.l1_loss(valid_performed, onehots)

    def configure_optimizers(self):
        hParams = self.hP
        optim = torch.optim.Adam(
            self.interpreter.parameters(), lr=hParams.lr, 
        )
        sched = torch.optim.lr_scheduler.ExponentialLR(
            optim, gamma=hParams.lr_decay, 
        )
        return [optim], [sched]

    def on_before_optimizer_step(self, _: torch.optim.Optimizer):
        norms = grad_norm(self.interpreter, norm_type=2)
        key = 'grad_2.0_norm_total'
        self.log_(key, norms[key])

        interval = os.environ.get('PLOT_INTERPRETER_EVERY_X_STEP')
        assert interval is not None
        if self.global_step % int(interval) == 0:
            self.plotInterpreter()
    
    @torch.no_grad()
    def plotInterpreter(self):
        w = self.interpreter.w.cpu()
        plt.imshow(
            w.numpy(), 
            aspect='auto', interpolation='nearest', 
            origin='lower', 
        )
        plt.colorbar()
        plt.xlabel(f'midi pitch - {PIANO_RANGE[0]}')
        plt.ylabel(f'piano key - {PIANO_RANGE[0]}')
        step = self.hP.formatGlobalStep(self.global_step)
        plt.title(f'{step = }')
        plt.tight_layout()
        plt.savefig(path.join(
            self.interpreter_visualized_dir, 
            step + '.png', 
        ))

def train(hParams: HParamsDecipher, root_dir: str):
    log_name = '.'
    os.makedirs(path.join(root_dir, log_name))
    litDecipher = LitDecipher(**dataclasses.asdict(hParams))
    profiler = SimpleProfiler(filename='profile')
    logger = TensorBoardLogger(root_dir, log_name)
    logJobMeta(logger, hParams.require_repo_working_tree_clean)
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
        overfit_batches=1 if hParams.overfit_first_batch else 0.0, 
    )
    dataModule = LitDecipherDataModule(hParams)
    trainer.fit(litDecipher, dataModule)
    # torch.cuda.memory._dump_snapshot(path.join(root_dir, 'VRAM.pickle'))
    # torch.cuda.memory._record_memory_history(enabled=None) # type: ignore

    return litDecipher, dataModule
