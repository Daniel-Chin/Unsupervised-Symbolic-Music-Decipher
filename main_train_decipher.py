from os import path

from shared import *
from hparams import HParamsDecipher, DecipherStrategy, NoteIsPianoKeyHParam, FreeHParam, CNN_LSTM_HParam
from decipher_lightning import train
from decipher_subjective_eval import decipherSubjectiveEval

def main():
    initMainProcess()
    hParams = HParamsDecipher(
        # strategy = DecipherStrategy.NoteIsPianoKey,
        # strategy_hparam = NoteIsPianoKeyHParam(
        #     using_piano='2024_m06_d03@14_52_28_p_tea/version_0/checkpoints/epoch=49-step=70350.ckpt', 
        #     interpreter_sample_not_polyphonic = False,
        #     init_oracle_w_offset = None, 
        #     loss_weight_anti_collapse = 0.0, 
        # ), 

        strategy = DecipherStrategy.Free,
        strategy_hparam = FreeHParam(
            arch = CNN_LSTM_HParam(
                entrance_n_channel = 512, 
                blocks = [
                ], 
                lstm_hidden_size = 512,
                lstm_n_layers = 2,
                last_conv_kernel_radius = 3, 
                last_conv_n_channel = 512,
            ), 
            dropout = 0.0, 
        ),

        music_gen_version = 'small',

        loss_weight_left = 0.0, 
        loss_weight_right = 1.0, 

        train_set_size = 8000, 
        val_set_size = 2000,
        # train_set_size = 8, 
        # val_set_size = 8,

        lr = 1e-3, 
        lr_decay = 1.0, 
        batch_size = 32, 
        # batch_size = 8, 
        max_epochs = 30, 
        # max_epochs = 2, 
        overfit_first_batch = False, 

        continue_from = None, 
        # WARNING: using `continue_from` has a bug: the validation set is newly split, so data leak.
        
        require_repo_working_tree_clean = True, 
        # require_repo_working_tree_clean = False, 
    )
    exp_name = currentTimeDirName() + '_d_free_r'
    if not hParams.require_repo_working_tree_clean:
        exp_name += '_dirty_working_tree'
    print(f'{exp_name = }', flush=True)
    hParams.summary()
    root_dir = path.join(EXPERIMENTS_DIR, exp_name)
    litDecipher, dataModule = train(hParams, root_dir)
    decipherSubjectiveEval(litDecipher, dataModule)
    print('OK')

if __name__ == '__main__':
    main()
