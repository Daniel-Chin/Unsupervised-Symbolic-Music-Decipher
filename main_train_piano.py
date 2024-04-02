from os import path

import torch

from shared import *
from hparams import HParams
from tf_piano_lightning import train
from tf_piano_evaluate_audio import evaluateAudio

def main():
    initMainProcess()
    hParams = HParams(
        d_model = 512,
        key_event_encoder_n_layers = 1,
        key_event_encoder_d_hidden = None,
        key_event_onset_as_positional_encoding = True, 
        tf_piano_n_head = 8,
        tf_piano_n_encoder_layers = 3,
        tf_piano_n_decoder_layers = 3,
        tf_piano_d_feedforward = 1024,

        tf_piano_train_set_size = 1024, 
        tf_piano_val_monkey_set_size = 128, 
        tf_piano_val_oracle_set_size = 128, 

        lr = 1e-4,
        batch_size = 8,
        max_epochs = 10,
        require_repo_working_tree_clean = True, 
    )
    exp_name = currentTimeDirName() + '_piano_pe_key_event'
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
