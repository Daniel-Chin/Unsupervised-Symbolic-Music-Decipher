import os
from functools import lru_cache
import shutil

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import lightning as L
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler
import scipy.io.wavfile as wavfile

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
        writeLightningHparams(hParams, self, hParams.require_repo_working_tree_clean)
        self.example_input_array = (
            torch.zeros((hParams.batch_size, 233, 1 + 1 + 88)), 
            [200] * hParams.batch_size, 
        )

        self.did_setup: bool = False
    
    def log(self, *a, **kw):
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
        self, batch: Tuple[Tensor, Tensor, List[int], List[str]], batch_idx: int, 
    ):
        x, y, x_lens, _ = batch
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
        self, batch: Tuple[Tensor, Tensor, List[int], List[str]], 
        batch_idx: int, dataloader_idx: int, 
    ):
        def log(name: str, value: float | int | Tensor):
            self.log(f'{VAL_CASES[dataloader_idx]}_{name}', value)

        x, y, x_lens, _ = batch
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
                hParams.tf_piano_train_set_size + hParams.tf_piano_val_monkey_set_size, 
            )

        @lru_cache(maxsize=1)
        def oracleDataset():
            return TransformerPianoDataset(
                'oracle', TRANSFORMER_PIANO_ORACLE_DATASET_DIR, 
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
    log_name = 'lightning_logs'
    os.makedirs(path.join(root_dir, log_name))
    if GPU_NAME == 'NVIDIA GeForce RTX 3050 Ti Laptop GPU':
        torch.set_float32_matmul_precision('high')
    litPiano = LitPiano(hParams)
    profiler = SimpleProfiler(filename='profile.txt')
    logger = TensorBoardLogger(root_dir, log_name)
    trainer = L.Trainer(
        devices=[DEVICE.index], max_epochs=hParams.max_epochs, 
        default_root_dir=root_dir,
        logger=logger, profiler=profiler, 
        callbacks=[DeviceStatsMonitor()], 
    )
    dataModule = LitPianoDataModule(hParams)
    trainer.fit(litPiano, dataModule)

    litPiano.eval()
    with torch.no_grad():
        evaluateAudio(litPiano, dataModule, root_dir)

    return litPiano

def evaluateAudio(
    litPiano: LitPiano, dataModule: LitPianoDataModule, 
    root_dir: str, 
):
    # to speed up dataloader worker spawning
    from my_encodec import getEncodec
    
    print('eval audio...', flush=True)
    audio_dir = path.join(root_dir, 'audio')
    encodec = getEncodec().to(DEVICE)
    subsets = ['train', 'val_monkey', 'val_oracle']
    loaders = [
        dataModule.train_dataloader(shuffle=False), 
        *dataModule.val_dataloader(),
    ]
    dataset_dirs = [
        TRANSFORMER_PIANO_MONKEY_DATASET_DIR, 
        TRANSFORMER_PIANO_ORACLE_DATASET_DIR, 
        TRANSFORMER_PIANO_ORACLE_DATASET_DIR, 
    ]
    n_evals = []
    for subset in subsets:
        env_var = os.environ.get('N_AUDIO_EVAL_' + subset.upper())
        assert env_var is not None, subset
        n_evals.append(int(env_var))
    max_n = max(n_evals)
    n_digits = len(str(max_n))
    index_format = f'0{n_digits}'
    def filename(subset: str, i: int, task: str):
        return path.join(
            audio_dir, 
            f'{subset}_{i:{index_format}}_{task}.wav',
        )

    for subset, loader, n_eval, dataset_dir in zip(
        subsets, loaders, n_evals, dataset_dirs, 
    ):
        print(f'{subset = }')
        datapoint_i = 0
        for batch in loader:
            print(datapoint_i, '/', n_eval, flush=True)
            x, _, x_lens, stems = batch
            x: Tensor
            x_lens: List[int]
            batch_size = x.shape[0]
            y_hat = litPiano.forward(x, x_lens)
            wave = encodec.decode(y_hat.argmax(dim=-1))
            assert wave.shape[1] == 1
            wave_cpu = wave[:, 0, :].cpu().numpy()
            for i in range(batch_size):
                wavfile.write(
                    filename(subset, datapoint_i, 'predict'), ENCODEC_SR, wave_cpu[i, :],
                )
                src = path.join(dataset_dir, stems[i])
                shutil.copyfile(src + '.mid', filename(subset, datapoint_i, 'reference'))
                shutil.copyfile(src + '_encodec_recon.wav', filename(subset, datapoint_i, 'encodec_recon'))
                datapoint_i += 1
                if datapoint_i == n_eval:
                    break
            else:
                continue
            break
