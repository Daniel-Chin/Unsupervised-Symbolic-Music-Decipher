from os import path

from shared import *
from hparams import HParamsDecipher, DecipherStrategy, NoteIsPianoKeyHParam, FreeHParam, CNN_LSTM_HParam
from decipher_lightning import train
from decipher_subjective_eval import decipherSubjectiveEval

def main():
    initMainProcess()
    continue_from = None
    hParams = HParamsDecipher(
        # strategy = DecipherStrategy.NoteIsPianoKey,
        # strategy_hparam = NoteIsPianoKeyHParam(
        #     using_piano='2024_m06_d03@14_52_12_p_slow/version_0/checkpoints/epoch=149-step=211050.ckpt', 
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
        batch_size = 32, 
        # music_gen_version = 'medium',
        # batch_size = 16, 
        # batch_size = 8, 

        loss_weight_left = 1.0, 
        loss_weight_right = 0.0, 

        train_set_size = 8000, 
        val_set_size = 2000,
        # train_set_size = 8, 
        # val_set_size = 8,

        lr = 1e-3, 
        lr_decay = 1.0, 
        max_epochs = 30, 
        # max_epochs = 2, 
        overfit_first_batch = False, 
        
        require_repo_working_tree_clean = True, 
        # require_repo_working_tree_clean = False, 
    )
    hParams = None
    continue_from = path.join(
        EXPERIMENTS_DIR, 
        "2024_m06_d06@05_47_08_d_s_free_l/version_0/checkpoints/epoch=29-step=7500.ckpt", 
    )
    exp_name = currentTimeDirName() + '_d_s_free_l_cont'
    if hParams is not None and not hParams.require_repo_working_tree_clean:
        exp_name += '_dirty_working_tree'
    print(f'{exp_name = }', flush=True)
    root_dir = path.join(EXPERIMENTS_DIR, exp_name)
    litDecipher, dataModule = train(hParams or continue_from, root_dir)
    decipherSubjectiveEval(litDecipher, dataModule)
    print('OK')

if __name__ == '__main__':
    main()
