from os import path

from shared import *
from hparams import HParams
from tf_piano_lightning import train

def main():
    initMainProcess()
    hParams = HParams(
        d_model = 512,
        key_event_encoder_n_layers = 2,
        key_event_encoder_d_hidden = 1024,
        tf_piano_n_head = 8,
        tf_piano_n_encoder_layers = 3,
        tf_piano_n_decoder_layers = 3,
        tf_piano_d_feedforward = 1024,

        tf_piano_train_set_size = 1024, 
        tf_piano_val_monkey_set_size = 128, 
        tf_piano_val_oracle_set_size = 128, 

        lr = 1e-3,
        batch_size = 16,
        max_epochs = 10,
        require_repo_working_tree_clean = True, 
    )
    exp_name = currentTimeDirName() + '_piano_wide'
    if not hParams.require_repo_working_tree_clean:
        exp_name += '_dirty_working_tree'
    train(hParams, path.join(EXPERIMENTS_DIR, exp_name))
    print('OK')

if __name__ == '__main__':
    main()
