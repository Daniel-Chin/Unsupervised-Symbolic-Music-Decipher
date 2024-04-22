from dataclasses import dataclass, asdict

from shared import *

@dataclass(frozen=True)
class HParams:
    gru_piano_hidden_size: int
    gru_piano_n_layers: int
    gru_drop_out: float

    gru_piano_train_set_size: int
    gru_piano_val_monkey_set_size: int
    gru_piano_val_oracle_set_size: int
    gru_piano_do_validate: bool

    gru_piano_lr: float
    gru_piano_lr_decay: float
    gru_piano_batch_size: int
    gru_piano_max_epochs: int

    require_repo_working_tree_clean: bool

    def __post_init__(self):
        pass    # put validation here

    def summary(self):
        print('HParams:')
        for k, v in asdict(self).items():
            print(' ', k, '=', v)
        print(' ')
        print(' ', 'Ending lr =', self.gru_piano_lr * (self.gru_piano_lr_decay ** self.gru_piano_max_epochs))
