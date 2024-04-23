from os import path

from shared import *
from hparams import HParams
from gru_piano_lightning import train
from gru_piano_evaluate_audio import evaluateAudio

def main():
    initMainProcess()
    hParams = HParams(
        gru_piano_hidden_size = 512, 
        gru_piano_n_layers = 2, 
        gru_drop_out = 0.2,

        gru_piano_train_set_size = 8000, 
        gru_piano_val_monkey_set_size = 2000, 
        gru_piano_val_oracle_set_size = 128, 
        gru_piano_do_validate = True,

        gru_piano_lr = 1e-3,
        gru_piano_lr_decay = 0.999, 
        gru_piano_batch_size = 64,
        gru_piano_max_epochs = 1370,
        require_repo_working_tree_clean = True, 
    )
    exp_name = currentTimeDirName() + '_p_gru'
    if not hParams.require_repo_working_tree_clean:
        exp_name += '_dirty_working_tree'
    print(f'{exp_name = }', flush=True)
    hParams.summary()
    root_dir = path.join(EXPERIMENTS_DIR, exp_name)
    litPiano, dataModule = train(hParams, root_dir)
    evaluateAudio(litPiano.to(DEVICE), dataModule, root_dir)
    print('OK')

if __name__ == '__main__':
    main()
