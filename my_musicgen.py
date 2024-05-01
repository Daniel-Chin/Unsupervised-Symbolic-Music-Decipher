from functools import lru_cache

import torch
from torch import Tensor
from audiocraft.models.musicgen import MusicGen
from audiocraft.models.lm import ConditionTensors, LMOutput
from audiocraft.models.encodec import EncodecModel
from audiocraft.modules.conditioners import ConditioningAttributes, WavCondition, ClassifierFreeGuidanceDropout
from audiocraft.solvers.musicgen import MusicGenSolver

from shared import *

class MyMusicGen:
    def __init__(self, version: str = 'small') -> None:
        self.musicGen = MusicGen.get_pretrained('facebook/musicgen-' + version, device='cuda' if HAS_CUDA else 'cpu')
        encodec = self.musicGen.compression_model
        assert isinstance(encodec, EncodecModel)
        assert encodec.sample_rate == ENCODEC_SR
        encodec.eval()
        self.encodec = encodec
        self.lm = self.musicGen.lm
        self.lm.eval()

    @lru_cache()
    def blankCondition(self, batch_size: int):
        attr = ConditioningAttributes(text={'description': None})
        attr.wav['self_wav'] = WavCondition(
            torch.zeros((1, 1, 1), device=DEVICE),
            torch.tensor([0], device=DEVICE),
            sample_rate=[ENCODEC_SR],
            path=[None], 
        )
        
        attrs: List[ConditioningAttributes] = ClassifierFreeGuidanceDropout(p=1.0)([attr] * batch_size)
        # Conceptually this does nothing. In reality `seek_time` changes from `[]` to `[None]`, which I hope does nothing.

        tokenized = self.lm.condition_provider.tokenize(attrs)
        condition_tensors: ConditionTensors = self.lm.condition_provider(tokenized)
        return condition_tensors

    def lmForward(self, audio_tokens: Tensor):
        with self.musicGen.autocast:
            batch_size, K, T = audio_tokens.shape
            assert K == ENCODEC_N_BOOKS
            assert T == N_TOKENS_PER_DATAPOINT
            return self.lm.compute_predictions(audio_tokens, [], self.blankCondition(batch_size))

    def lmLoss(self, model_output: LMOutput, audio_tokens: Tensor):
        with self.musicGen.autocast:
            loss, ce_per_codebook = self.cELoss(
                model_output.logits, audio_tokens, model_output.mask, 
            )
            return loss, ce_per_codebook
    
    def cELoss(self, logits: Tensor, targets: Tensor, mask: Tensor):
        return MusicGenSolver._compute_cross_entropy(
            self=None, # type: ignore
            # Look, they could have made it static! 
            logits=logits, targets=targets, mask=mask, 
        )

# @lru_cache(1)
# def getSolver():
#     return MusicGenSolver(omegaconf.DictConfig(dict(
#         # generate = None, 
#     )))

MY_MUSICGEN = MyMusicGen()

if __name__ == '__main__':
    import pdb; pdb.set_trace()
