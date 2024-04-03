from dataclasses import dataclass
from functools import lru_cache

from shared import *
from key_event_format import KeyEventFormat

@dataclass(frozen=True)
class HParams:
    d_model: int
    key_event_encoder_n_layers: int
    key_event_encoder_d_hidden: Optional[int]
    key_event_onset_as_positional_encoding: bool
    key_event_key_as_modular_encoding: bool
    key_event_velocity_as_modular_encoding: bool
    is_modular_encoding_soft: Optional[bool]
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
        assert (
            self.key_event_key_as_modular_encoding
            or
            self.key_event_velocity_as_modular_encoding
        ) == (
            self.is_modular_encoding_soft is not None
        )
    
    @lru_cache()
    def keyEventFormat(self):
        return KeyEventFormat(
            self.key_event_onset_as_positional_encoding, 
            self.d_model,
            self.key_event_key_as_modular_encoding,
            self.key_event_velocity_as_modular_encoding,
        )
