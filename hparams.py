from dataclasses import dataclass, asdict

@dataclass
class HParams:
    d_model: int
    key_event_encoder_d_hidden: int
    key_event_encoder_n_layers: int
    tf_piano_n_head: int
    tf_piano_n_encoder_layers: int
    tf_piano_n_decoder_layers: int
    tf_piano_d_feedforward: int

    def asDict(self) -> dict:
        return asdict(self)
