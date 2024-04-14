from dataclasses import dataclass, asdict

from shared import *

@dataclass(frozen=True)
class HParams:
    cnn_piano_cnn: List[Tuple[int, int]]
    cnn_piano_fc: List[List[int]]

    cnn_piano_train_set_size: int
    cnn_piano_val_monkey_set_size: int
    cnn_piano_val_oracle_set_size: int
    cnn_piano_do_validate: bool

    cnn_piano_lr: float
    cnn_piano_lr_decay: float
    cnn_piano_batch_size: int
    cnn_piano_max_epochs: int

    require_repo_working_tree_clean: bool

    def __post_init__(self):
        assert len(self.cnn_piano_fc) == ENCODEC_N_BOOKS

    def summary(self):
        print('HParams:')
        for k, v in asdict(self).items():
            print(' ', k, '=', v)
        print(' ')
        print(' ', 'Ending lr =', self.cnn_piano_lr * (self.cnn_piano_lr_decay ** self.cnn_piano_max_epochs))
