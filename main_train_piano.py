from os import path

from shared import *
from hparams import HParamsPiano, PianoArchType, CNNHParam, TransformerHParam, PianoOutType
from piano_lightning import train
from piano_evaluate_audio import evaluateAudio

def main():
    initMainProcess()
    hParams = HParamsPiano(
        arch_type = PianoArchType.CNN, 
        arch_hparam = CNNHParam(512, [
            [
                (1, 512), 
                (1, 512), 
            ], 
            [
                (1, 512), 
                (1, 512), 
            ], 
            [
                (1, 512), 
                (1, 512), 
            ], 
            [
                (0, 512), 
            ], 
        ]), 

        # arch_type = PianoArchType.Transformer,
        # arch_hparam = TransformerHParam(
        #     d_model = 1024, 
        #     n_heads = 8, 
        #     d_feedforward = 2048, 
        #     n_layers = 6, 
        #     attn_radius = None, 
        # ),

        dropout = 0.0, 

        out_type = PianoOutType.LogSpectrogram,

        train_set_size = 800, 
        val_monkey_set_size = 200, 
        val_oracle_set_size = 128, 
        do_validate = True,

        lr = 1e-3,
        lr_decay = 0.999, 
        batch_size = 32,
        max_epochs = 30,
        overfit_first_batch = False, 

        require_repo_working_tree_clean = True, 
    )
    exp_name = currentTimeDirName() + '_p_spec'
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
