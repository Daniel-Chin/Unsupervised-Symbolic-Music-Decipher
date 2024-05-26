import os
from os import path
import shutil

import torch
import scipy.io.wavfile as wavfile
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from shared import *
from hparams import PianoOutType
from piano_dataset import BatchType
from piano_lightning import LitPiano, LitPianoDataModule, VAL_CASES

@torch.no_grad()
def pianoSubjectiveEval(
    litPiano: LitPiano, dataModule: LitPianoDataModule, 
    root_dir: str, 
):
    # Both `litPiano` and `dataModule` are already `setup()`-ed.  

    # to speed up dataloader worker spawning
    from my_musicgen import myMusicGen
    
    print('pianoSubjectiveEval()...', flush=True)
    hParams = litPiano.hP
    litPiano.eval()
    subjective_dir = path.join(root_dir, 'subjective_eval')
    os.makedirs(subjective_dir)
    encodec = myMusicGen.encodec.to(DEVICE)
    encodec.eval()
    subsets = ['train', *VAL_CASES]
    batch_size = min(8, dataModule.hP.batch_size)
    loaders = [
        dataModule.train_dataloader(batch_size, shuffle=False), 
        *dataModule.val_dataloader(batch_size, ),
    ]
    dataset_dirs = [    # violates DRY w/ VAL_CASES
        PIANO_MONKEY_DATASET_DIR, 
        PIANO_MONKEY_DATASET_DIR, 
        PIANO_ORACLE_DATASET_DIR, 
    ]
    n_evals = []
    for subset in subsets:
        env_var = os.environ.get('N_SUBJECTIVE_EVAL_' + subset.upper())
        assert env_var is not None, subset
        n_evals.append(int(env_var))
    max_n = max(n_evals)
    n_digits = len(str(max_n))
    index_format = f'0{n_digits}'
    def filename(subset: str, i: int, task: str, ext: str):
        return path.join(
            subjective_dir, 
            f'{subset}_{i:{index_format}}_{task}.{ext}',
        )

    def doSet(subset, loader, n_eval, dataset_dir):
        print(f'{subset = }', flush=True)
        for batch_i, batch in enumerate(loader):
            batch: BatchType
            score_cpu, encodec_tokens_cpu, _, data_ids = batch
            score = score_cpu.to(DEVICE)
            encodec_tokens = encodec_tokens_cpu.to(DEVICE)
            batch_size = score.shape[0]

            if batch_i * batch_size >= n_eval:
                return
            
            y_hat = litPiano.forward(score)

            if hParams.out_type == PianoOutType.Score:
                score_hat = y_hat
            else:
                if hParams.out_type == PianoOutType.EncodecTokens:
                    wave_hat = encodec.decode(
                        y_hat.argmax(dim=-1), 
                    ).squeeze(1)
                    wave = encodec.decode(encodec_tokens).squeeze(1)
                if hParams.out_type == PianoOutType.LogSpectrogram:
                    _, griffinLim, _= fftTools()
                    wave_hat: torch.Tensor = griffinLim(y_hat)
                assert wave_hat.shape == (batch_size, wave_hat.shape[1]), wave_hat.shape
                wave_hat_cpu = wave_hat.cpu().numpy()
                wave_cpu = wave.cpu().numpy()

            for i in range(batch_size):
                datapoint_i = batch_i * batch_size + i
                if datapoint_i == n_eval:
                    break
                src = path.join(dataset_dir, data_ids[i])
                shutil.copyfile(src + '.mid', filename(
                    subset, datapoint_i, 'reference', 'mid', 
                ))
                # shutil.copyfile(src + '_synthed.wav', filename(
                #     subset, datapoint_i, 'reference', 'wav', 
                # ))
                if hParams.out_type == PianoOutType.EncodecTokens:
                    # shutil.copyfile(src + '_encodec_recon.wav', filename(
                    #     subset, datapoint_i, 'encodec_recon', 'wav', 
                    # ))
                    wavfile.write(
                        filename(subset, datapoint_i, 'encodec_recon', 'wav'), ENCODEC_SR, 
                        wave_cpu[i, :],
                    )
                if hParams.out_type == PianoOutType.LogSpectrogram:
                    shutil.copyfile(src + '_griffin_lim.wav', filename(
                        subset, datapoint_i, 'griffin_lim', 'wav', 
                    ))
                if hParams.out_type == PianoOutType.Score:
                    for name, s in [
                        ('reference', score), 
                        ('predict', score_hat),
                    ]:
                        s: Tensor
                        fig = Figure()
                        ax = fig.subplots(1)
                        assert isinstance(ax, Axes)
                        im = plotScore(s[i, :, :, :].cpu(), ax)
                        colorBar(fig, ax, im)
                        fig.savefig(filename(
                            subset, datapoint_i, name, 'png', 
                        ))
                else:
                    wavfile.write(
                        filename(subset, datapoint_i, 'predict', 'wav'), ENCODEC_SR, 
                        wave_hat_cpu[i, :],
                    )
            print(datapoint_i, '/', n_eval, flush=True)

    [doSet(*x) for x in zip(
        subsets, loaders, n_evals, dataset_dirs, 
    )]
