import os
from os import path
import shutil

from torch import Tensor
import scipy.io.wavfile as wavfile

from shared import *
from tf_piano_dataset import CollateFnOut
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

    def doSet(subset, loader, n_eval, dataset_dir):
        print(f'{subset = }', flush=True)
        for batch_i, batch in enumerate(loader):
            batch: CollateFnOut
            x_cpu, y_cpu, mask_cpu, data_ids = batch
            x = x_cpu.to(DEVICE)
            y = y_cpu.to(DEVICE)
            mask = mask_cpu.to(DEVICE)
            batch_size = x.shape[0]

            if batch_i * batch_size >= n_eval:
                return
            
            tasks: List[Tuple[str, Tensor]] = []
            if litPiano.hP.tf_piano_decoder_auto_regressive:
                tasks.append((
                    'teacher_forcing', 
                    litPiano.tfPiano.forward(x, mask, y), 
                ))
                tasks.append((
                    'auto_regressive', 
                    litPiano.tfPiano.autoRegress(x, mask), 
                ))
            else:
                tasks.append((
                    '', 
                    litPiano.tfPiano.forward(x, mask, None), 
                ))
            waves = []
            for task_name, y_logits in tasks:
                wave = encodec.decode(y_logits.argmax(dim=-1))
                assert wave.shape[1] == 1
                waves.append(wave[:, 0, :].cpu().numpy())

            for i in range(batch_size):
                datapoint_i = batch_i * batch_size + i
                if datapoint_i == n_eval:
                    break
                src = path.join(dataset_dir, data_ids[i])
                shutil.copyfile(src + '_synthed.wav', filename(subset, datapoint_i, 'reference', 'wav'))
                shutil.copyfile(src + '_encodec_recon.wav', filename(subset, datapoint_i, 'encodec_recon', 'wav'))
                for (task_name, _), wave_cpu in zip(tasks, waves):
                    wavfile.write(
                        filename(subset, datapoint_i, 'predict_' + task_name, 'wav'), ENCODEC_SR, wave_cpu[i, :],
                    )
            print(datapoint_i, '/', n_eval, flush=True)

    [doSet(*x) for x in zip(
        subsets, loaders, n_evals, dataset_dirs, 
    )]
