from os import path

import torch

from shared import *
from hparams import HParams
from tf_piano_lightning import train
from tf_piano_evaluate_audio import evaluateAudio

def main():
    initMainProcess()
    hParams = HParams(
        tf_piano_d_model = 512,
        key_event_encoder_n_layers = 1,
        key_event_encoder_d_hidden = None,
        key_event_onset_as_positional_encoding = True, 
        key_event_key_as_modular_encoding = True, 
        key_event_velocity_as_modular_encoding = True, 
        is_modular_encoding_soft = False,
        tf_piano_n_head = 8,
        tf_piano_n_encoder_layers = 3,
        tf_piano_n_decoder_layers = 3,
        tf_piano_d_feedforward = 1024,
        tf_piano_decoder_auto_regressive = True,

        tf_piano_train_set_size = 8000, 
        tf_piano_val_monkey_set_size = 2000, 
        tf_piano_val_oracle_set_size = 128, 

        tf_piano_lr = 2e-4,
        tf_piano_batch_size = 8,
        tf_piano_max_epochs = 100,
        require_repo_working_tree_clean = True, 
    )
    exp_name = currentTimeDirName() + '_p_big_tf'
    print(f'{exp_name = }', flush=True)
    if not hParams.require_repo_working_tree_clean:
        exp_name += '_dirty_working_tree'
    root_dir = path.join(EXPERIMENTS_DIR, exp_name)
    litPiano, dataModule = train(hParams, root_dir)
    litPiano.eval()
    with torch.no_grad():
        evaluateAudio(litPiano.to(DEVICE), dataModule, root_dir)
    print('OK')

if __name__ == '__main__':
    main()
