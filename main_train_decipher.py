from os import path

from shared import *
from hparams import HParamsDecipher
from decipher_lightning import train

def main():
    initMainProcess()
    hParams = HParamsDecipher(
        using_piano='banana palace', 

        interpreter_sample_not_polyphonic = False,

        loss_weight_left = 1.0, 
        loss_weight_right = 1.0, 

        train_set_size = 800, 
        val_set_size = 200,

        lr = 1e-3, 
        lr_decay = 1.0, 
        batch_size = 64, 
        max_epochs = 300, 
        overfit_first_batch = False, 

        require_repo_working_tree_clean = True, 
    )
    exp_name = currentTimeDirName() + '_decipher_first'
    if not hParams.require_repo_working_tree_clean:
        exp_name += '_dirty_working_tree'
    print(f'{exp_name = }', flush=True)
    hParams.summary()
    root_dir = path.join(EXPERIMENTS_DIR, exp_name)
    litDecipher, dataModule = train(hParams, root_dir)
    # evaluateAudio?
    print('OK')

if __name__ == '__main__':
    main()
