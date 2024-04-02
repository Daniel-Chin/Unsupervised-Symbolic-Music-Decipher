import os
from os import path
import shutil

from torch import Tensor
import scipy.io.wavfile as wavfile

from shared import *
from tf_piano_lightning import LitPiano, LitPianoDataModule

def evaluateAudio(
    litPiano: LitPiano, dataModule: LitPianoDataModule, 
    root_dir: str, 
):
    # dataModule: an already-setup LitPianoDataModule

    # to speed up dataloader worker spawning
    from my_encodec import getEncodec
    
    print('eval audio...', flush=True)
    audio_dir = path.join(root_dir, 'audio')
    os.makedirs(audio_dir)
    encodec = getEncodec().to(DEVICE)
    subsets = ['train', 'val_monkey', 'val_oracle']
    loaders = [
        dataModule.train_dataloader(shuffle=False), 
        *dataModule.val_dataloader(),
    ]
    dataset_dirs = [
        TRANSFORMER_PIANO_MONKEY_DATASET_DIR, 
        TRANSFORMER_PIANO_MONKEY_DATASET_DIR, 
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
    def filename(subset: str, i: int, task: str, ext: str):
        return path.join(
            audio_dir, 
            f'{subset}_{i:{index_format}}_{task}.{ext}',
        )

    for subset, loader, n_eval, dataset_dir in zip(
        subsets, loaders, n_evals, dataset_dirs, 
    ):
        print(f'{subset = }', flush=True)
        datapoint_i = 0
        for batch in loader:
            try:
                x, _, mask, stems = batch
                x: Tensor
                mask: Tensor
                batch_size = x.shape[0]
                y_hat = litPiano.forward(x.to(DEVICE), mask)
                wave = encodec.decode(y_hat.argmax(dim=-1))
                assert wave.shape[1] == 1
                wave_cpu = wave[:, 0, :].cpu().numpy()
                for i in range(batch_size):
                    wavfile.write(
                        filename(subset, datapoint_i, 'predict', 'wav'), ENCODEC_SR, wave_cpu[i, :],
                    )
                    src = path.join(dataset_dir, stems[i])
                    shutil.copyfile(src + '_synthed.wav', filename(subset, datapoint_i, 'reference', 'wav'))
                    shutil.copyfile(src + '_encodec_recon.wav', filename(subset, datapoint_i, 'encodec_recon', 'wav'))
                    datapoint_i += 1
                    if datapoint_i == n_eval:
                        break
                else:
                    continue
                break
            finally:
                print(datapoint_i, '/', n_eval, flush=True)
