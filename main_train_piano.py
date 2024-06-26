from os import path

from shared import *
from music import PIANO_RANGE
from hparams import (
    HParamsPiano, PianoOutType, PianoArchType, CNNHParam, TransformerHParam, 
    GRUHParam, PerformanceNetHParam, CNN_LSTM_HParam, 
)
from piano_lightning import train
from piano_subjective_eval import pianoSubjectiveEval

def main():
    initMainProcess()
    continue_from = None
    hParams = HParamsPiano(
        # arch_type = PianoArchType.CNN, 
        # arch_hparam = CNNHParam(512, [
        #     [
        #         (1, 512), 
        #         (1, 512), 
        #     ], 
        #     [
        #         (1, 512), 
        #     ], 
        # ]), 

        # arch_type = PianoArchType.Transformer,
        # arch_hparam = TransformerHParam(
        #     d_model = 1024, 
        #     n_heads = 8, 
        #     d_feedforward = 2048, 
        #     n_layers = 6, 
        #     attn_radius = None, 
        # ),

        # arch_type = PianoArchType.GRU, 
        # arch_hparam = GRUHParam(
        #     n_hidden = 512, 
        #     n_layers = 4, 
        # ),

        # arch_type = PianoArchType.PerformanceNet,
        # arch_hparam = PerformanceNetHParam(
        #     depth = 5, 
        #     start_channels = 128, 
        #     end_channels = 3201, 
        # ),

        arch_type = PianoArchType.CNN_LSTM, 
        arch_hparam = CNN_LSTM_HParam(
            entrance_n_channel = 512, 
            blocks = [
            ], 
            lstm_hidden_size = 512,
            lstm_n_layers = 1,
            last_conv_kernel_radius = 3, 
            last_conv_n_channel = 512,
        ), 

        music_gen_version = 'small',
        dropout = 0.0, 

        out_type = PianoOutType.EncodecTokens,

        train_set_size = 90000, 
        val_monkey_set_size = 512, 
        val_oracle_set_size = 512, 
        # train_set_size = 16, 
        # val_monkey_set_size = 16, 
        # val_oracle_set_size = 16, 
        do_validate = True,

        lr = 1e-3,
        lr_decay = .99, 
        batch_size = 64,
        # batch_size = 8,
        max_epochs = 150,
        # max_epochs = 3,
        overfit_first_batch = False, 

        require_repo_working_tree_clean = True, 
        # require_repo_working_tree_clean = False, 
        random_seed = 16, 
    )
    # hParams = None
    # continue_from = path.join(
    #     EXPERIMENTS_DIR, 
    #     "???", 
    # )
    exp_name = currentTimeDirName() + '_p_slow'
    if hParams is not None and not hParams.require_repo_working_tree_clean:
        exp_name += '_dirty_working_tree'
    print(f'{exp_name = }', flush=True)
    root_dir = path.join(EXPERIMENTS_DIR, exp_name)
    litPiano, dataModule = train(hParams or continue_from, root_dir)
    pianoSubjectiveEval(litPiano.to(DEVICE), dataModule)
    print('OK')

if __name__ == '__main__':
    main()
