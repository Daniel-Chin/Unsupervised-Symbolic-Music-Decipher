from os import path

from shared import *
from hparams import HParams, PianoArchType, CNNHParam, TransformerHParam
from piano_lightning import train
from piano_evaluate_audio import evaluateAudio

def main():
    initMainProcess()
    hParams = HParams(
        # piano_arch_type = PianoArchType.CNN, 
        # piano_arch_hparam = CNNHParam(1024, [
        #     [
        #         (1, 1024), 
        #         (1, 1024), 
        #     ], 
        #     [
        #         (1, 1024), 
        #         (1, 1024), 
        #     ], 
        #     [
        #         (1, 1024), 
        #         (1, 1024), 
        #     ], 
        #     [
        #         (0, 1024), 
        #     ], 
        # ]), 

        piano_arch_type = PianoArchType.Transformer,
        piano_arch_hparam = TransformerHParam(
            d_model = 512, 
            n_heads = 4, 
            d_feedforward = 1024, 
            n_layers = 3, 
            attn_radius = None, 
        ),

        piano_dropout = 0.0, 

        piano_train_set_size = 8000, 
        piano_val_monkey_set_size = 2000, 
        piano_val_oracle_set_size = 128, 
        piano_do_validate = True,

        piano_lr = 1e-3,
        piano_lr_decay = 0.999, 
        piano_batch_size = 32,
        piano_max_epochs = 300,
        require_repo_working_tree_clean = True, 
    )
    exp_name = currentTimeDirName() + '_p_tf_1batch'
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
