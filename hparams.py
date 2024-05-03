from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum

from shared import *

class PianoArchType(Enum):
    CNN = 'CNN'
    Transformer = 'Transformer'

arch_types = {}
def registerArchType(x: PianoArchType, /):
    def decorator(cls):
        arch_types[x] = cls
        return cls
    return decorator

class PianoArchHParam: pass

ConvLayerHParam = Tuple[int, int]    # (kernel_radius, out_channels)
CNNResidualBlockHParam = List[ConvLayerHParam]

@registerArchType(PianoArchType.CNN)
@dataclass(frozen=True)
class CNNHParam(PianoArchHParam):
    entrance_n_channel: int
    blocks: List[CNNResidualBlockHParam]

@registerArchType(PianoArchType.Transformer)
@dataclass(frozen=True)
class TransformerHParam(PianoArchHParam):
    d_model: int
    n_heads: int
    d_feedforward: int
    n_layers: int
    attn_radius: Optional[int]

@dataclass(frozen=True)
class HParams:
    piano_arch_type: PianoArchType
    piano_arch_hparam: PianoArchHParam
    piano_dropout: float

    piano_train_set_size: int
    piano_val_monkey_set_size: int
    piano_val_oracle_set_size: int
    piano_do_validate: bool

    piano_lr: float
    piano_lr_decay: float
    piano_batch_size: int
    piano_max_epochs: int

    interpreter_sample_not_polyphonic: bool
    decipher_loss_weight_left: float
    decipher_loss_weight_right: float
    decipher_train_set_size: int
    decipher_val_set_size: int

    decipher_lr: float
    decipher_lr_decay: float
    decipher_batch_size: int
    decipher_max_epochs: int

    require_repo_working_tree_clean: bool

    def __post_init__(self):
        assert isinstance(
            self.piano_arch_hparam, arch_types[self.piano_arch_type], 
        )

    def summary(self):
        print('HParams:')
        for k, v in asdict(self).items():
            print(' ', k, '=', v)
        print(' ')

        piano_total_decay = self.piano_lr_decay ** self.piano_max_epochs
        print(' ', f'{piano_total_decay = :.2e}')
        piano_ending_lr = self.piano_lr * piano_total_decay
        print(' ', f'{piano_ending_lr = :.2e}')

        decipher_total_decay = self.decipher_lr_decay ** self.decipher_max_epochs
        print(' ', f'{decipher_total_decay = :.2e}')
        decipher_ending_lr = self.piano_lr * decipher_total_decay
        print(' ', f'{decipher_ending_lr = :.2e}')
