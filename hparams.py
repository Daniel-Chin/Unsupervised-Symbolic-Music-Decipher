from dataclasses import dataclass

from shared import *

@dataclass(frozen=True)
class HParams:
    d_model: int
    key_event_encoder_n_layers: int
    key_event_encoder_d_hidden: Optional[int]
    tf_piano_n_head: int
    tf_piano_n_encoder_layers: int
    tf_piano_n_decoder_layers: int
    tf_piano_d_feedforward: int

    tf_piano_train_set_size: int
    tf_piano_val_monkey_set_size: int
    tf_piano_val_oracle_set_size: int

    lr: float
    batch_size: int
    max_epochs: int

    require_repo_working_tree_clean: bool

    def __post_init__(self):
        assert (
            self.key_event_encoder_n_layers == 1
        ) == (
            self.key_event_encoder_d_hidden is None
        )
