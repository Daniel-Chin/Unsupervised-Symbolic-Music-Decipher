from os import path

from shared import *
from hparams import HParams
from cnn_piano_lightning import train
from cnn_piano_evaluate_audio import evaluateAudio

def main():
    initMainProcess()
    hParams = HParams(
        cnn_piano_architecture = (512, [
            [
                (2, 512), 
                (2, 512), 
            ], 
            [
                (2, 512), 
                (2, 512), 
            ], 
            [
                (2, 512), 
                (2, 512), 
            ], 
            [
                (0, 512), 
            ], 
        ]), 

        cnn_piano_train_set_size = 8000, 
        cnn_piano_val_monkey_set_size = 2000, 
        cnn_piano_val_oracle_set_size = 128, 
        cnn_piano_do_validate = True,

        cnn_piano_lr = 1e-3,
        cnn_piano_lr_decay = 0.999, 
        cnn_piano_batch_size = 64,
        cnn_piano_max_epochs = 2000,
        require_repo_working_tree_clean = True, 
    )
    exp_name = currentTimeDirName() + '_p_rec_field'
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
