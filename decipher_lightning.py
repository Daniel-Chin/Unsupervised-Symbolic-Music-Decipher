import dataclasses
from functools import cached_property

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.distributions.categorical import Categorical
import lightning as L
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch.utilities.types import _METRIC
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.callbacks import DeviceStatsMonitor, ModelSummary
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from shared import *
from music import PIANO_RANGE
from hparams import HParamsDecipher, NoteIsPianoKeyHParam, FreeHParam, PianoOutType
from piano_dataset import PianoDataset, BatchType
from piano_model import PianoModel
from interpreter import Interpreter
from my_musicgen import MyMusicGen, LMOutput
from sample_with_ste_backward import sampleWithSTEBackward
from piano_lightning import LitPiano

@dataclasses.dataclass(frozen=True)
class LMOutputDistribution:
    lmOutput: LMOutput

    @cached_property
    def categorical(self):
        mask = self.lmOutput.mask    # mask with False: invalid, True: valid.  
        valid_logits = self.lmOutput.logits[mask, :] # (B*K*T', ENCODEC_N_WORDS_PER_BOOK)
        return Categorical(logits=valid_logits)

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

class LitDecipher(TorchworkModule):
    def __init__(self, **kw) -> None:
        hParams = HParamsDecipher.fromDict(kw)
        super().__init__(hParams)
        self.hP: HParamsDecipher

        example_batch_size = 3
        self.example_input_array = torch.randn(
            (
                example_batch_size, 2, 
                PIANO_RANGE[1] - PIANO_RANGE[0], N_FRAMES_PER_DATAPOINT, 
            ), 
        )

        if isinstance(hParams.strategy_hparam, NoteIsPianoKeyHParam):
            def getPiano():
                checkpoint_path = hParams.getPianoAbsPaths()
                litPiano = LitPiano.load_from_checkpoint(checkpoint_path)
                litPiano.train()
                return litPiano.piano

            self.piano = getPiano()
            freeze(self.piano)
            self.interpreter = Interpreter(hParams)
        elif isinstance(hParams.strategy_hparam, FreeHParam):
            self.performer = PianoModel(
                hParams.strategy_hparam.arch, PianoOutType.EncodecTokens, 
                hParams.strategy_hparam.dropout, 
            )
        else:
            raise TypeError(type(hParams.strategy_hparam))

    def log_(
        self, name: str, value: _METRIC, 
        also_average_by_epoch: str | None = None,   # "abe"
        **kw, 
    ):
        hParams = self.hP
        if also_average_by_epoch is not None:
            self.log_(
                name, 
                value, None, **kw, on_step=True, on_epoch=False, 
            )
            self.log_(
                also_average_by_epoch, 
                value, None, **kw, on_step=False, on_epoch=True, 
            )
        else:
            super().log(name, value, batch_size=hParams.batch_size, **kw)

    def setup(self, stage: str):
        super().setup(stage)
        hParams = self.hP

        if isinstance(hParams.strategy_hparam, NoteIsPianoKeyHParam):
            self.interpreter_visualized_dir = path.join(
                getLogDir(self.logger), 'interpreter_visualized', 
            )
            os.makedirs(self.interpreter_visualized_dir)
            plot_interpreter_every_x_step = os.environ.get('PLOT_INTERPRETER_EVERY_X_STEP')
            assert plot_interpreter_every_x_step is not None
            self.plot_interpreter_every_x_step = plot_interpreter_every_x_step
        elif isinstance(hParams.strategy_hparam, FreeHParam):
            pass
        else:
            raise TypeError(type(hParams.strategy_hparam))

    def forward(self, x: Tensor):
        '''
        `x` shape: (batch_size, 2, n_pitches, n_frames)  
        out shape: (batch_size, ENCODEC_N_BOOKS, n_frames, ENCODEC_N_WORDS_PER_BOOK)
        '''
        hParams = self.hP
        if isinstance(hParams.strategy_hparam, NoteIsPianoKeyHParam):
            x = self.interpreter.forward(x)
            if DO_CHECK_NAN:
                assert not x.isnan().any(), pdb.set_trace()
            x = x.contiguous()
            x = self.piano.forward(x)
            return x
        elif isinstance(hParams.strategy_hparam, FreeHParam):
            x = self.performer.forward(x)
            return x
        else:
            raise TypeError(type(hParams.strategy_hparam))

    def training_step(self, batch: BatchType, batch_idx: int):
        MyMusicGen.singleton(self.hP.music_gen_version).train()
        return self.shared_step('train', batch, batch_idx)
    
    def validation_step(self, batch: BatchType, batch_idx: int):
        MyMusicGen.singleton(self.hP.music_gen_version).eval()
        return self.shared_step('val', batch, batch_idx)
    
    def shared_step(
        self, step_name: str, batch: BatchType, batch_idx: int, 
    ):
        hParams = self.hP
        strategy_hP = hParams.strategy_hparam
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
        prediction = LMOutputDistribution(
            MyMusicGen.singleton(self.hP.music_gen_version).lmPredict(sampled_encodec_onehots), 
        )
        entropy: Tensor = prediction.categorical.entropy()
        # measures MusicGen certainty. Low entropy = high certainty.
        self.log_(
            'music_gen_entropy', entropy.mean(dim=0), 
            'fav/music_gen_entropy_abe', 
        )

        loss = torch.zeros(( ), device=self.device)
        def logLoss(name: Optional[str], loss: Tensor, is_fav: bool = False):
            suffix = '' if name is None else '_' + name
            full_name = step_name + '_loss' + suffix
            self.log_(full_name, loss, 'fav/' + full_name + '_abe')
        if hParams.loss_weight_left != 0.0:
            loss_left, ce_per_codebook = MyMusicGen.singleton(
                self.hP.music_gen_version, 
            ).lmLoss(
                prediction.lmOutput, 
                sampled_encodec_onehots.argmax(dim=-1), 
            )
            logLoss('left', loss_left, is_fav=True)
            loss += hParams.loss_weight_left  * loss_left
            for k, ce_k in enumerate(ce_per_codebook):
                logLoss(f'left_codebook_{k}', ce_k)
        if hParams.loss_weight_right != 0.0:
            loss_right = self.lossRight(encodec_tokens_logits, prediction)
            logLoss('right', loss_right, is_fav=True)
            loss += hParams.loss_weight_right * loss_right
        if isinstance(strategy_hP, NoteIsPianoKeyHParam):
            if strategy_hP.loss_weight_anti_collapse != 0.0:
                loss_anti_collapse = self.lossAntiCollapse(self.interpreter.w)
                logLoss('anti_collapse', loss_anti_collapse, is_fav=True)
                loss += strategy_hP.loss_weight_anti_collapse * loss_anti_collapse
        logLoss(None, loss, is_fav=True)
        return loss
    
    def lossRight(self, performed: Tensor, lmOutputDistribution: LMOutputDistribution):
        mask = lmOutputDistribution.lmOutput.mask    # mask with False: invalid, True: valid.  
        valid_performed = performed[mask, :] # (B*K*T', ENCODEC_N_WORDS_PER_BOOK)
        c = lmOutputDistribution.categorical
        sampled = c.sample()    # grad is lost, and that's just right
        # `sampled` shape: (B*K*T', )
        onehots = (sampled == c.enumerate_support()).float().T
        return (onehots * valid_performed.softmax(dim=-1)).sum(dim=-1).mean(dim=0)
    
    def lossAntiCollapse(self, w: Tensor):
        a = w.softmax(dim=0).sum(dim=1)
        return F.mse_loss(a, torch.ones_like(a))

    def theTrainableModule(self):
        hParams = self.hP
        if isinstance(hParams.strategy_hparam, NoteIsPianoKeyHParam):
            return self.interpreter
        elif isinstance(hParams.strategy_hparam, FreeHParam):
            return self.performer
        else:
            raise TypeError(type(hParams.strategy_hparam))
    
    def configure_optimizers(self):
        hParams = self.hP
        optim = torch.optim.Adam(
            self.theTrainableModule().parameters(), lr=hParams.lr, 
        )
        sched = torch.optim.lr_scheduler.ExponentialLR(
            optim, gamma=hParams.lr_decay, 
        )
        return [optim], [sched]

    def on_before_optimizer_step(self, _: torch.optim.Optimizer):
        norms = grad_norm(self.theTrainableModule(), norm_type=2)
        key = 'grad_2.0_norm_total'
        self.log_(key, norms[key])

        if isinstance(self.hP.strategy_hparam, NoteIsPianoKeyHParam):
            self.log_('interpreter_mean', self.interpreter.w.mean(), 'fav/interpreter_mean_abe')
            if self.global_step % int(self.plot_interpreter_every_x_step) < 4:
                self.plotInterpreter()
    
    @torch.no_grad()
    def plotInterpreter(self):
        simplex = self.interpreter.simplex().cpu()
        fig = Figure()
        ax = fig.subplots(1)
        assert isinstance(ax, Axes)
        im = ax.imshow(
            simplex.numpy(), 
            vmin=0.0, vmax=1.0,
            aspect='auto', interpolation='nearest', 
            origin='lower', 
        )
        colorBar(fig, ax, im)
        ax.set_xlabel(f'midi pitch - {PIANO_RANGE[0]}')
        ax.set_ylabel(f'piano key - {PIANO_RANGE[0]}')
        step = self.hP.formatGlobalStep(self.global_step)
        ax.set_title(f'{step = }')
        fig.tight_layout()
        fig.savefig(path.join(
            self.interpreter_visualized_dir, 
            step + '.png', 
        ))

    def optimizer_step(self, *a, **kw):
        result = super().optimizer_step(*a, **kw)
        if self.hP.project_w_to_doubly_stochastic:
            self.interpreter.sinkhornKnopp()
        return result

def train(hParams_or_continue_from: HParamsDecipher | str, root_dir: str):
    litDecipher, continue_from = LitDecipher.new(
        hParams_or_continue_from, HParamsDecipher, 
    )
    assert isinstance(litDecipher, LitDecipher)
    hParams = litDecipher.hP
    hParams.summary()
    log_name = '.'
    os.makedirs(path.join(root_dir, log_name))
    profiler = SimpleProfiler(filename='profile')
    logger = TensorBoardLogger(root_dir, log_name)
    logJobMeta(getLogDir(logger), hParams.require_repo_working_tree_clean)
    # torch.cuda.memory._record_memory_history(max_entries=100000)
    trainer = L.Trainer(
        devices=[DEVICE.index], max_epochs=hParams.max_epochs, 
        gradient_clip_val=5.0, 
        default_root_dir=root_dir,
        logger=logger, 
        # profiler=profiler, 
        callbacks=[
            # DeviceStatsMonitor(), 
            ModelSummary(max_depth=3), 
        ], 
        log_every_n_steps=min(50, hParams.train_set_size // hParams.batch_size), 
        overfit_batches=1 if hParams.overfit_first_batch else 0.0, 
    )
    dataModule = LitDecipherDataModule(hParams)
    trainer.fit(
        litDecipher, dataModule,
        ckpt_path=continue_from, 
    )
    # torch.cuda.memory._dump_snapshot(path.join(root_dir, 'VRAM.pickle'))
    # torch.cuda.memory._record_memory_history(enabled=None) # type: ignore

    return litDecipher, dataModule
