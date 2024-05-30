import os
from os import path
import shutil

import torch
from torch import Tensor
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
    do_sample_not_polyphonic = litDecipher.hP.interpreter_sample_not_polyphonic
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

    simplex_decipher = litDecipher.interpreter.w.softmax(dim=0)
    simplex_random = torch.randn((
        PIANO_RANGE[1] - PIANO_RANGE[0],
        PIANO_RANGE[1] - PIANO_RANGE[0],
    )).softmax(dim=0)
    if do_sample_not_polyphonic:
        c_decipher = Categorical(simplex_decipher)
        c_random = Categorical(simplex_random)
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
            interpreteMidi(
                midi, do_sample_not_polyphonic, 
                c_decipher if do_sample_not_polyphonic else simplex_decipher, 
            ).write(filename(subset_name, i, 'decipher'))
            interpreteMidi(
                midi, do_sample_not_polyphonic, 
                c_random if do_sample_not_polyphonic else simplex_random, 
            ).write(filename(subset_name, i, 'random'))

def interpreteMidi(
    src: pretty_midi.PrettyMIDI, 
    do_sample_not_polyphonic: bool, 
    interpreter: Tensor | Categorical,
):
    if do_sample_not_polyphonic:
        assert isinstance(interpreter, Categorical)
        switcherboard = interpreter.sample(torch.Size(( )))
        assert switcherboard.shape == (PIANO_RANGE[1] - PIANO_RANGE[0], )
    else:
        assert isinstance(interpreter, Tensor)
        simplex = interpreter
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(0)
    midi.instruments.append(piano)
    src_piano, = src.instruments
    src_piano: pretty_midi.Instrument
    for note in src_piano.notes:
        note: pretty_midi.Note
        key_i = note.pitch - PIANO_RANGE[0]
        if do_sample_not_polyphonic:
            piano.notes.append(pretty_midi.Note(
                note.velocity, 
                switcherboard[key_i].item() + PIANO_RANGE[0], 
                note.start,
                note.end,
            ))
        else:
            for i, p in enumerate(simplex[key_i, :]):
                piano.notes.append(pretty_midi.Note(
                    round(note.velocity * p.sqrt().item()), 
                    i + PIANO_RANGE[0], 
                    note.start,
                    note.end,
                ))
    return midi
