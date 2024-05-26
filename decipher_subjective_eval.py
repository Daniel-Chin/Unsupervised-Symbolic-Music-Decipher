import os
from os import path
import shutil

import torch
from torch.distributions.categorical import Categorical
from tqdm import tqdm
import pretty_midi

from shared import *
from piano_dataset import BatchType
from decipher_lightning import LitDecipher, LitDecipherDataModule

N_EVALS = 16

@torch.no_grad()
def decipherSubjectiveEval(
    litDecipher: LitDecipher, dataModule: LitDecipherDataModule, 
    root_dir: str, 
):
    # Both `litDecipher` and `dataModule` are already `setup()`-ed.  

    print('decipherSubjectiveEval()...', flush=True)
    litDecipher.eval()
    litDecipher = litDecipher.cpu()
    subjective_dir = path.join(root_dir, 'subjective_eval')
    os.makedirs(subjective_dir)
    batch_size = min(8, dataModule.hP.batch_size)
    n_digits = len(str(N_EVALS))
    index_format = f'0{n_digits}'
    def filename(subset_name:str, i: int, task: str):
        return path.join(
            subjective_dir, 
            f'{subset_name}_{i:{index_format}}_{task}.mid',
        )

    simplex = litDecipher.interpreter.w.softmax(dim=0)
    categorical = Categorical(simplex)
    random_baseline = Categorical(torch.randn((
        PIANO_RANGE[1] - PIANO_RANGE[0], 
        PIANO_RANGE[1] - PIANO_RANGE[0], 
    )).softmax(dim=0))
    for subset_name, loader in dict(
        train = dataModule.train_dataloader(batch_size, shuffle=False), 
        val = dataModule.val_dataloader(batch_size, ),
    ).items():
        data_ids_acc = []
        for batch in loader:
            batch: BatchType
            _, _, _, data_ids = batch
            data_ids_acc.extend(data_ids)
            if len(data_ids_acc) >= N_EVALS:
                data_ids_acc = data_ids_acc[:N_EVALS]
                break

        for i, data_id in enumerate(tqdm(data_ids_acc, desc=subset_name)):
            src = path.join(PIANO_ORACLE_DATASET_DIR, data_id + '.mid')
            shutil.copyfile(src, filename(subset_name, i, 'reference'))

            midi = pretty_midi.PrettyMIDI(src)
            piano, = midi.instruments
            piano: pretty_midi.Instrument
            for task, c in dict(
                random = random_baseline,
                deciphered = categorical,
            ).items():
                switcherboard = c.sample(torch.Size(( )))
                assert switcherboard.shape == (PIANO_RANGE[1] - PIANO_RANGE[0], )
                new_midi = pretty_midi.PrettyMIDI()
                new_piano = pretty_midi.Instrument(0)
                new_midi.instruments.append(new_piano)
                for note in piano.notes:
                    note: pretty_midi.Note
                    new_piano.notes.append(pretty_midi.Note(
                        note.velocity, 
                        switcherboard[note.pitch - PIANO_RANGE[0]].item(), 
                        note.start,
                        note.end,
                    ))
                new_midi.write(filename(subset_name, i, task))
