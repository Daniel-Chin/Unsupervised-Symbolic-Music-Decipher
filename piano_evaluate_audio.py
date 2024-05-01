import os
from os import path
import shutil

import torch
import scipy.io.wavfile as wavfile

from shared import *
from piano_dataset import BatchType
from piano_lightning import LitPiano, LitPianoDataModule, VAL_CASES

@torch.no_grad()
def evaluateAudio(
    litPiano: LitPiano, dataModule: LitPianoDataModule, 
    root_dir: str, 
):
    # Both `litPiano` and `dataModule` are already `setup()`-ed.  

    # to speed up dataloader worker spawning
    from my_musicgen import getEncodec
    
    print('eval audio...', flush=True)
    litPiano.eval()
    audio_dir = path.join(root_dir, 'audio')
    os.makedirs(audio_dir)
    encodec = getEncodec().to(DEVICE)
    subsets = ['train', *VAL_CASES]
    batch_size = min(8, dataModule.hP.piano_batch_size)
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
        env_var = os.environ.get('N_AUDIO_EVAL_' + subset.upper())
        assert env_var is not None, subset
        n_evals.append(int(env_var))
    max_n = max(n_evals)
    n_digits = len(str(max_n))
    index_format = f'0{n_digits}'
    def filename(subset: str, i: int, task: str, ext: str):
        return path.join(
            audio_dir, 
            f'{subset}_{i:{index_format}}_{task}.{ext}',
        )

    def doSet(subset, loader, n_eval, dataset_dir):
        print(f'{subset = }', flush=True)
        for batch_i, batch in enumerate(loader):
            batch: BatchType
            x_cpu, _, data_ids = batch
            x = x_cpu.to(DEVICE)
            batch_size = x.shape[0]

            if batch_i * batch_size >= n_eval:
                return
            
            y_logits = litPiano.forward(x)
            wave: np.ndarray = encodec.decode(
                y_logits.argmax(dim=-1), 
            ).cpu().numpy()
            assert wave.shape[1] == 1

            for i in range(batch_size):
                datapoint_i = batch_i * batch_size + i
                if datapoint_i == n_eval:
                    break
                src = path.join(dataset_dir, data_ids[i])
                shutil.copyfile(src + '_synthed.wav', filename(
                    subset, datapoint_i, 'reference', 'wav', 
                ))
                shutil.copyfile(src + '_encodec_recon.wav', filename(
                    subset, datapoint_i, 'encodec_recon', 'wav', 
                ))
                wavfile.write(
                    filename(subset, datapoint_i, 'predict', 'wav'), ENCODEC_SR, wave[i, 0, :],
                )
            print(datapoint_i, '/', n_eval, flush=True)

    [doSet(*x) for x in zip(
        subsets, loaders, n_evals, dataset_dirs, 
    )]
