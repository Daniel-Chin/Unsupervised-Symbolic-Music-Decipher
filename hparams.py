from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import cached_property

import dacite

from shared import *
from music import PIANO_RANGE

@dataclass(frozen=True)
class AVHHParams(BaseHParams):
    music_gen_version: str

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

class PianoOutType(Enum):
    EncodecTokens = 'EncodecTokens'
    LogSpectrogram = 'LogSpectrogram'
    Score = 'PianoRoll' # identity mapping, for debugging

    @cached_property
    def shape(self):
        if self == PianoOutType.EncodecTokens:
            return (ENCODEC_N_BOOKS, ENCODEC_N_WORDS_PER_BOOK)
        if self == PianoOutType.LogSpectrogram:
            _, _, n_bins = fftTools()
            return (n_bins, )
        if self == PianoOutType.Score:
            return (2, PIANO_RANGE[1] - PIANO_RANGE[0])
        raise ValueError(self)

@dataclass(frozen=True)
class HParamsPiano(AVHHParams):
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

    @staticmethod
    def fromDict(d: Dict, /):
        # backward-compatible defaults
        if 'random_seed' not in d:
            d['random_seed'] = 16

        t = arch_types[d['arch_type']]
        def f(x):
            return dacite.from_dict(t, x)
        return dacite.from_dict(__class__, d, config=dacite.Config(type_hooks={
            PianoArchHParam: f, 
        }))

class DecipherStrategy(Enum):
    NoteIsPianoKey = 'NoteIsPianoKey'
    Free = 'Free'

strategy_types: Dict[DecipherStrategy, type] = {}
def registerStrategy(x: DecipherStrategy, /):
    def decorator(cls):
        strategy_types[x] = cls
        return cls
    return decorator

class StrategyHParam: pass

@registerStrategy(DecipherStrategy.NoteIsPianoKey)
@dataclass(frozen=True)
class NoteIsPianoKeyHParam(StrategyHParam):
    using_piano: str
    interpreter_sample_not_polyphonic: bool
    init_oracle_w_offset: Optional[int]

    loss_weight_anti_collapse: float

@registerStrategy(DecipherStrategy.Free)
@dataclass(frozen=True)
class FreeHParam(StrategyHParam):
    arch: CNN_LSTM_HParam
    dropout: float

@dataclass(frozen=True)
class HParamsDecipher(AVHHParams):
    strategy: DecipherStrategy
    strategy_hparam: StrategyHParam
    project_w_to_doubly_stochastic: bool

    loss_weight_left: float
    loss_weight_right: float
    
    val_set_size: int

    def __post_init__(self):
        assert isinstance(
            self.strategy_hparam, strategy_types[self.strategy], 
        )
        if isinstance(self.strategy_hparam, NoteIsPianoKeyHParam):
            if self.strategy_hparam.loss_weight_anti_collapse > 0:
                assert self.project_w_to_doubly_stochastic is False, 'mutually exclusive'

    def getPianoAbsPaths(self):
        assert isinstance(self.strategy_hparam, NoteIsPianoKeyHParam)
        return path.join(EXPERIMENTS_DIR, self.strategy_hparam.using_piano)

    @staticmethod
    def fromDict(d: Dict, /):
        t = strategy_types[d['strategy']]
        def f(x):
            return dacite.from_dict(t, x)
        return dacite.from_dict(__class__, d, config=dacite.Config(type_hooks={
            StrategyHParam: f, 
        }))
