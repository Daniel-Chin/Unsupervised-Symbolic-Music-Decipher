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
            d_model = 1024, 
            n_heads = 8, 
            d_feedforward = 2048, 
            n_layers = 6, 
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

        interpreter_sample_not_polyphonic = False,
        decipher_loss_weight_left = 1.0, 
        decipher_loss_weight_right = 1.0, 
        decipher_train_set_size = 800, 
        decipher_val_set_size = 200,

        decipher_lr = 1e-3, 
        decipher_lr_decay = 1.0, 
        decipher_batch_size = 64, 
        decipher_max_epochs = 300, 

        require_repo_working_tree_clean = True, 
    )
    exp_name = currentTimeDirName() + '_p_tf_1b_ori_tf'
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
