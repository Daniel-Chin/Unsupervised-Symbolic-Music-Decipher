from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from functools import cached_property
import math

import dacite

from shared import *
from music import PIANO_RANGE

class PianoArchType(Enum):
    CNN = 'CNN'
    Transformer = 'Transformer'
    GRU = 'GRU'
    PerformanceNet = 'PerformanceNet'
    CNN_LSTM = 'CNN_LSTM'

arch_types: Dict[PianoArchType, type] = {}
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

@registerArchType(PianoArchType.GRU)
@dataclass(frozen=True)
class GRUHParam(PianoArchHParam):
    n_hidden: int
    n_layers: int

@registerArchType(PianoArchType.PerformanceNet)
@dataclass(frozen=True)
class PerformanceNetHParam(PianoArchHParam):
    depth: int
    start_channels: int
    end_channels: int

@registerArchType(PianoArchType.CNN_LSTM)
@dataclass(frozen=True)
class CNN_LSTM_HParam(PianoArchHParam):
    entrance_n_channel: int
    blocks: List[CNNResidualBlockHParam]
    lstm_hidden_size: int
    lstm_n_layers: int
    last_conv_kernel_radius: int
    last_conv_n_channel: int

@dataclass(frozen=True)
class HParams:
    lr: float
    lr_decay: float
    batch_size: int
    train_set_size: int
    max_epochs: int
    overfit_first_batch: bool

    continue_from: Optional[str]

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
    
    @cached_property
    def n_total_steps(self):
        return math.ceil(self.train_set_size / self.batch_size) * self.max_epochs 
    
    @cached_property
    def global_step_f_string(self):
        return f'0{len(str(self.n_total_steps))}d'
    
    def formatGlobalStep(self, global_step: int):
        return format(global_step, self.global_step_f_string)
    
    def getContinueFromAbsPath(self):
        if self.continue_from is None:
            return None
        return path.join(EXPERIMENTS_DIR, self.continue_from)

class PianoOutType(Enum):
    EncodecTokens = 'EncodecTokens'
    LogSpectrogram = 'LogSpectrogram'
    Score = 'PianoRoll' # identity mapping, for debugging

@dataclass(frozen=True)
class HParamsPiano(HParams):
    arch_type: PianoArchType
    arch_hparam: PianoArchHParam
    dropout: float

    out_type: PianoOutType
    
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
        if self.out_type == PianoOutType.Score:
            return (2, PIANO_RANGE[1] - PIANO_RANGE[0])
        raise ValueError(self.out_type)
    
    @staticmethod
    def fromDict(d: Dict, /):
        t = arch_types[d['arch_type']]
        def f(x):
            return dacite.from_dict(t, x)
        return dacite.from_dict(__class__, d, config=dacite.Config(type_hooks={
            PianoArchHParam: f, 
        }))

@dataclass(frozen=True)
class HParamsDecipher(HParams):
    using_piano: str

    interpreter_sample_not_polyphonic: bool
    init_oracle_w_offset: Optional[int]

    loss_weight_left: float
    loss_weight_right: float
    
    val_set_size: int

    def getPianoAbsPaths(self):
        return path.join(EXPERIMENTS_DIR, self.using_piano)

    @staticmethod
    def fromDict(d: Dict, /):
        return dacite.from_dict(__class__, d)
