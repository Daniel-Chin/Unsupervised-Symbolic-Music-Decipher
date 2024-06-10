import os
from os import path
import shutil

import torch
from torch import Tensor
from torch.distributions.categorical import Categorical
import scipy.io.wavfile as wavfile
from tqdm import tqdm
import pretty_midi

from piano_lightning import LitPiano
from shared import *
from music import pitch2name
from piano_dataset import BatchType
from decipher_lightning import LitDecipher, LitDecipherDataModule, train
from midi_reasonablizer import MidiReasonablizer
from hparams import HParamsDecipher, DecipherStrategy, NoteIsPianoKeyHParam, FreeHParam
from my_musicgen import MyMusicGen

N_EVALS = 16

@torch.no_grad()
def decipherSubjectiveEval(
    litDecipher: LitDecipher, dataModule: LitDecipherDataModule, 
):
    # Both `litDecipher` and `dataModule` are already `setup()`-ed.  

    print('decipherSubjectiveEval()...', flush=True)
    strategy_hP = litDecipher.hP.strategy_hparam
    if isinstance(strategy_hP, NoteIsPianoKeyHParam):
        do_sample_not_polyphonic = strategy_hP.interpreter_sample_not_polyphonic
    elif isinstance(strategy_hP, FreeHParam):
        encodec = MyMusicGen.singleton(litDecipher.hP.music_gen_version).encodec.to(DEVICE)
        encodec.eval()
    litDecipher.eval()
    subjective_dir = path.join(getLogDir(litDecipher.logger), 'subjective_eval')
    os.makedirs(subjective_dir)
    batch_size = min(8, dataModule.hP.batch_size)
    n_digits = len(str(N_EVALS))
    index_format = f'0{n_digits}'
    def filename(subset_name:str, i: int, task: str, ext: str):
        return path.join(
            subjective_dir, 
            f'{subset_name}_{i:{index_format}}_{task}.{ext}',
        )

    if isinstance(strategy_hP, NoteIsPianoKeyHParam):
        litDecipher = litDecipher.cpu()
        simplex_decipher = litDecipher.interpreter.w.softmax(dim=0)
        simplex_random = torch.randn((
            PIANO_RANGE[1] - PIANO_RANGE[0],
            PIANO_RANGE[1] - PIANO_RANGE[0],
        )).softmax(dim=0)
        if do_sample_not_polyphonic:
            c_decipher = Categorical(simplex_decipher.T)
            c_random = Categorical(simplex_random.T)
    elif isinstance(strategy_hP, FreeHParam):
        litDecipher = litDecipher.to(DEVICE)
    else:
        raise TypeError(type(strategy_hP))
    for subset_name, loader in dict(
        train = dataModule.train_dataloader(batch_size, shuffle=False), 
        val = dataModule.val_dataloader(batch_size, ),
    ).items():
        data_ids_acc = []
        performed_waves: List[np.ndarray] = []
        for batch in loader:
            batch: BatchType
            score, _, _, data_ids = batch
            data_ids_acc.extend(data_ids)
            if isinstance(strategy_hP, FreeHParam):
                performed = litDecipher.forward(score.to(DEVICE)).argmax(dim=-1)
                wave_gpu = encodec.decode(performed).squeeze(1)
                assert wave_gpu.shape == (batch_size, wave_gpu.shape[1]), wave_gpu.shape
                wave_cpu = wave_gpu.cpu().numpy()
                performed_waves.extend(wave_cpu)

            if len(data_ids_acc) >= N_EVALS:
                data_ids_acc = data_ids_acc[:N_EVALS]
                break

        for i, data_id in enumerate(tqdm(data_ids_acc, desc=subset_name)):
            src = path.join(PIANO_ORACLE_DATASET_DIR, data_id + '.mid')
            shutil.copyfile(src, filename(subset_name, i, 'reference', 'mid'))

            if isinstance(strategy_hP, NoteIsPianoKeyHParam):
                midi = pretty_midi.PrettyMIDI(src)
                interpreteMidi(
                    midi, do_sample_not_polyphonic, 
                    c_decipher if do_sample_not_polyphonic else simplex_decipher, 
                ).write(filename(subset_name, i, 'decipher', 'mid'))
                interpreteMidi(
                    midi, do_sample_not_polyphonic, 
                    c_random if do_sample_not_polyphonic else simplex_random, 
                ).write(filename(subset_name, i, 'random', 'mid'))
            elif isinstance(strategy_hP, FreeHParam):
                wavfile.write(
                    filename(subset_name, i, 'performed', 'wav'), ENCODEC_SR, 
                    performed_waves[i],
                )

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
    if not do_sample_not_polyphonic:
        midiReasonablizer = MidiReasonablizer(piano)
    src_piano, = src.instruments
    src_piano: pretty_midi.Instrument
    sortByNoteOn(src_piano)
    # print('Original:')
    # printMidi(src_piano)
    for note in src_piano.notes:
        note: pretty_midi.Note
        if do_sample_not_polyphonic:
            piano.notes.append(pretty_midi.Note(
                note.velocity, 
                switcherboard[note.pitch - PIANO_RANGE[0]].item() + PIANO_RANGE[0], 
                note.start,
                note.end,
            ))
        else:
            energy = note.velocity ** 2
            for i, l in enumerate(simplex[:, note.pitch - PIANO_RANGE[0]]):
                loading = l.item()
                if loading >= 1e-2:
                    midiReasonablizer.add(pretty_midi.Note(
                        round((loading * energy) ** 0.5), 
                        i + PIANO_RANGE[0], 
                        note.start,
                        note.end,
                    ))
        # print('added note', pitch2name(note.pitch), note.start, '-', note.end)
        # printMidi(piano)
        # input('Enter...')
    return midi

def testReasonablizer():
    initMainProcess()
    hParams = HParamsDecipher(
        strategy = DecipherStrategy.NoteIsPianoKey,
        strategy_hparam = NoteIsPianoKeyHParam(
            using_piano='2024_m06_d03@14_52_28_p_tea/version_0/checkpoints/epoch=49-step=70350.ckpt', 
            interpreter_sample_not_polyphonic = False,
            init_oracle_w_offset = None, 
            loss_weight_anti_collapse = 0.0, 
        ), 

        music_gen_version = 'small',

        loss_weight_left = 0.0, 
        loss_weight_right = 1.0, 

        train_set_size = 2, 
        val_set_size = 2,

        lr = 1e-3, 
        lr_decay = 1.0, 
        batch_size = 2, 
        max_epochs = 0, 
        overfit_first_batch = False, 
        
        require_repo_working_tree_clean = False, 
    )
    exp_name = currentTimeDirName() + '_d_test_reasonablizer'
    root_dir = path.join(EXPERIMENTS_DIR, exp_name)
    litDecipher, dataModule = train(hParams, root_dir)
    decipherSubjectiveEval(litDecipher, dataModule)
    print('OK')

if __name__ == '__main__':
    testReasonablizer()
