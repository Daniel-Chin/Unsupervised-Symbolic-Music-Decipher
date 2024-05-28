from os import path

from shared import *
from hparams import HParamsDecipher
from decipher_lightning import train
from decipher_subjective_eval import decipherSubjectiveEval

def main():
    initMainProcess()
    hParams = HParamsDecipher(
        using_piano='2024_m05_d23@22_07_23_p_tofu/version_0/checkpoints/epoch=169-step=212500.ckpt', 

        interpreter_sample_not_polyphonic = True,

        loss_weight_left = 0.0, 
        loss_weight_right = 1.0, 

        train_set_size = 8000, 
        val_set_size = 2000,
        # train_set_size = 16, 
        # val_set_size = 16,

        lr = 1e-3, 
        lr_decay = 1.0, 
        batch_size = 32, 
        # batch_size = 8, 
        max_epochs = 30, 
        # max_epochs = 3, 
        overfit_first_batch = False, 

        require_repo_working_tree_clean = True, 
        # require_repo_working_tree_clean = False, 
    )
    exp_name = currentTimeDirName() + '_d_right_only'
    if not hParams.require_repo_working_tree_clean:
        exp_name += '_dirty_working_tree'
    print(f'{exp_name = }', flush=True)
    hParams.summary()
    root_dir = path.join(EXPERIMENTS_DIR, exp_name)
    litDecipher, dataModule = train(hParams, root_dir)
    decipherSubjectiveEval(litDecipher, dataModule, root_dir)
    print('OK')

if __name__ == '__main__':
    main()
