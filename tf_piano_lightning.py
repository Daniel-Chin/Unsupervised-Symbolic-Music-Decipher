from typing import *

from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L

from shared import *
from hparams import HParams
from tf_piano_model import TFPiano, KeyEventEncoder, TransformerPianoModel
from tf_piano_dataset import TransformerPianoDataset, collate

class LitPiano(L.LightningModule):
    def __init__(self, tfPiano: TFPiano) -> None:
        super().__init__()
        self.tfPiano = tfPiano
    
    def training_step(
        self, batch: Tuple[Tensor, Tensor, List[int]], batch_idx, 
    ):
        x, y, x_lens = batch
        y_hat = self.tfPiano.forward(x, x_lens)
        loss = F.cross_entropy(
            y_hat.view(-1, ENCODEC_N_WORDS_PER_BOOK), 
            y    .view(-1), 
        )
        self.log("batch_idx", batch_idx)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.tfPiano.parameters(), lr=1e-3)

class LitPianoDataModule(L.LightningDataModule):
    def __init__(self) -> None:
        super().__init__()
    
    def setup(self, stage: Optional[str] = None):
        self.dataset = TransformerPianoDataset()
    
    def train_dataloader(self):
        dataset = TransformerPianoDataset()
        loader = DataLoader(
            dataset, batch_size=16, collate_fn=collate, 
            num_workers=2, shuffle=True, 
        )
        return loader

def main(hParams: HParams):
    keyEventEncoder = KeyEventEncoder(
        hParams.d_model, hParams.key_event_encoder_d_hidden, 
        hParams.key_event_encoder_n_layers,
    )
    transformerPianoModel = TransformerPianoModel(
        hParams.d_model, hParams.tf_piano_n_head,
        hParams.tf_piano_n_encoder_layers, 
        hParams.tf_piano_n_decoder_layers,
        hParams.tf_piano_d_feedforward, 
    )
    tfPiano = TFPiano(keyEventEncoder, transformerPianoModel)
    litPiano = LitPiano(tfPiano)
    trainer = L.Trainer(
        devices=[DEVICE.index], max_epochs=3, 
    )
    trainer.fit(litPiano, LitPianoDataModule())
