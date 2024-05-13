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
    lr: float
    lr_decay: float
    batch_size: int
    max_epochs: int
    overfit_first_batch: bool

    require_repo_working_tree_clean: bool

    def summary(self):
        print('HParams:')
        for k, v in asdict(self).items():
            print(' ', k, '=', v)
        print(' ')

        total_decay = self.lr_decay ** self.max_epochs
        print(' ', f'{total_decay = :.2e}')
        ending_lr = self.lr * total_decay
        print(' ', f'{ending_lr = :.2e}')

class PianoOutType(Enum):
    EncodecTokens = 'EncodecTokens'
    LogSpectrogram = 'LogSpectrogram'

@dataclass(frozen=True)
class HParamsPiano(HParams):
    arch_type: PianoArchType
    arch_hparam: PianoArchHParam
    dropout: float

    out_type: PianoOutType
    
    train_set_size: int
    val_monkey_set_size: int
    val_oracle_set_size: int
    do_validate: bool

    def __post_init__(self):
        assert isinstance(
            self.arch_hparam, arch_types[self.arch_type], 
        )
    
    def outShape(self):
        if self.out_type == PianoOutType.EncodecTokens:
            return (ENCODEC_N_BOOKS, ENCODEC_N_WORDS_PER_BOOK)
        if self.out_type == PianoOutType.LogSpectrogram:
            _, _, n_bins = fftTools()
            return (n_bins, )
        raise ValueError(self.out_type)

@dataclass(frozen=True)
class HParamsDecipher(HParams):
    using_piano: str

    interpreter_sample_not_polyphonic: bool

    loss_weight_left: float
    loss_weight_right: float
    
    train_set_size: int
    val_set_size: int
