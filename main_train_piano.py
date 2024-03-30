from os import path

from shared import *
from hparams import HParams
from tf_piano_lightning import train

def main():
    hParams = HParams(
        d_model = 256,
        key_event_encoder_n_layers = 1,
        key_event_encoder_d_hidden = None,
        tf_piano_n_head = 4,
        tf_piano_n_encoder_layers = 3,
        tf_piano_n_decoder_layers = 3,
        tf_piano_d_feedforward = 1024,
        lr = 1e-3,
        batch_size = 16,
        max_epochs = 10,
    )
    train(hParams, path.join(
        EXPERIMENTS_DIR, currentTimeDirName() + '_piano_first', 
    ))

if __name__ == '__main__':
    main()