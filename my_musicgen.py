from __future__ import annotations

from functools import lru_cache

import torch
from torch import Tensor
from audiocraft.models.musicgen import MusicGen
from audiocraft.models.lm import ConditionTensors, LMOutput
from audiocraft.models.encodec import EncodecModel
from audiocraft.modules.conditioners import ConditioningAttributes, WavCondition, ClassifierFreeGuidanceDropout
from audiocraft.modules.codebooks_patterns import Pattern, DelayedPatternProvider
from audiocraft.solvers.musicgen import MusicGenSolver

from shared import *

class PatternOnehot(Pattern):
    def build_pattern_sequence_onehot(self, z: Tensor):
        keep_only_valid_steps = True
        special_token = float('nan')
        B, K, T, n_word_per_book = z.shape
        indexes, mask = self._build_pattern_sequence_scatter_indexes(
            T, K, keep_only_valid_steps=keep_only_valid_steps, device=str(z.device)
        )
        z = z.view(B, K * T, n_word_per_book)
        # we append the special token as the last index of our flattened z tensor
        z = torch.cat([z, torch.zeros_like(z[:, :1, :]) + special_token], dim=1)
        values = z[:, indexes.view(-1), :]
        values = values.view(B, K, indexes.shape[-1], n_word_per_book)
        return values, indexes, mask
    
    def revert_pattern_sequence_onehot(self, s: Tensor):
        return self.revert_pattern_logits(
            s, float('nan'), keep_only_valid_steps=True, 
        )

class DelayedPatternProviderOnehot(DelayedPatternProvider):
    def get_pattern(self, timesteps: int):
        pattern = super().get_pattern(timesteps)
        return PatternOnehot(pattern.layout, pattern.timesteps, pattern.n_q)

class MyMusicGen:
    def __init__(self, version: str = 'small') -> None:
        self.musicGen = MusicGen.get_pretrained('facebook/musicgen-' + version, device='cuda' if HAS_CUDA else 'cpu')
        encodec = self.musicGen.compression_model
        assert isinstance(encodec, EncodecModel)
        assert encodec.sample_rate == ENCODEC_SR
        encodec.eval()
        freeze(encodec)
        self.encodec = encodec
        self.lm = self.musicGen.lm
        self.lm.eval()
        freeze(self.lm)
        assert isinstance(self.lm.pattern_provider, DelayedPatternProvider)
        self.patternProvider = DelayedPatternProviderOnehot(
            ENCODEC_N_BOOKS, 
        )
        def dModel():
            emb = self.lm.emb[0]
            assert isinstance(emb, torch.nn.Embedding)
            return emb.weight.shape[1]
        self.lm_emb_w = torch.zeros((
            1, ENCODEC_N_BOOKS, dModel(), ENCODEC_N_WORDS_PER_BOOK, 
        ), device=DEVICE)
        for k in range(ENCODEC_N_BOOKS):
            emb = self.lm.emb[k]
            assert isinstance(emb, torch.nn.Embedding)
            self.lm_emb_w[0, k, :, :] = emb.weight[:ENCODEC_N_WORDS_PER_BOOK, :].T
        self.lm_emb_w = self.lm_emb_w.contiguous()

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

    def lmPredict(self, encodec_onehots: Tensor) -> LMOutput:
        '''
        `audio_tokens`: one-hot
        '''
        with self.musicGen.autocast:
            B, K, T, card = encodec_onehots.shape
            assert K == ENCODEC_N_BOOKS
            assert T == N_FRAMES_PER_DATAPOINT
            assert card == ENCODEC_N_WORDS_PER_BOOK
            condition_tensors = self.blankCondition(B)
            encodec_onehots = encodec_onehots.contiguous()
            # map codes [B, K, T, card] into pattern sequence [B, K, S, card] using special_token_id for masked tokens
            pattern = self.patternProvider.get_pattern(T)
            sequence_onehots, _, _ = pattern.build_pattern_sequence_onehot(
                encodec_onehots, 
            )
            # apply model on pattern sequence
            assert self.lm._fsdp is None
            logits = self.lmForward(sequence_onehots, [], condition_tensors)  # [B, K, S, card]
            # map back the logits on pattern sequence to logits on original codes: [B, K, S, card] -> [B, K, T, card]
            # and provide the corresponding mask over invalid positions of tokens
            logits = logits.permute(0, 3, 1, 2)  # [B, card, K, S]
            # note: we use nans as special token to make it obvious if we feed unexpected logits
            logits, _, logits_mask = pattern.revert_pattern_sequence_onehot(
                logits, 
            )
            logits = logits.permute(0, 2, 3, 1)  # [B, K, T, card]
            logits_mask = logits_mask[None, :, :].expand(B, -1, -1)  # [K, T] -> [B, K, T]
            return LMOutput(logits, logits_mask)
    
    def lmForward(
        self, sequence_onehots: torch.Tensor,
        conditions: List[ConditioningAttributes],
        condition_tensors: Optional[ConditionTensors] = None, 
    ) -> torch.Tensor:
        B, K, S, card = sequence_onehots.shape
        assert K == self.lm.num_codebooks, "Sequence shape must match the specified number of codebooks"
        emb = self.lm_emb_w @ sequence_onehots.permute(
            0, 1, 3, 2, 
            # (B, K, card, S)
        )
        # (B, K, d_model, S)
        input_ = emb.sum(dim=3).permute(2, 1, 0)
        # (B, S, d_model)
        assert condition_tensors is not None
        assert not conditions

        input_, cross_attention_input = self.lm.fuser(input_, condition_tensors)

        out = self.lm.transformer(input_, cross_attention_src=cross_attention_input)
        if self.lm.out_norm:
            out = self.lm.out_norm(out)
        logits = torch.stack([self.lm.linears[k](out) for k in range(K)], dim=1)  # [B, K, S, card]

        # remove the prefix from the model outputs
        if len(self.lm.fuser.fuse2cond['prepend']) > 0:
            logits = logits[:, :, -S:]

        return logits  # [B, K, S, card]

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

myMusicGen = MyMusicGen()

if __name__ == '__main__':
    def testPattern():
        provider = DelayedPatternProviderOnehot(ENCODEC_N_BOOKS)
        pattern = provider.get_pattern(N_FRAMES_PER_DATAPOINT)
        batch_size = 3
        audio_tokens = torch.randn((
            batch_size, ENCODEC_N_BOOKS, N_FRAMES_PER_DATAPOINT, ENCODEC_N_WORDS_PER_BOOK, 
        ), requires_grad=True)
        seq, _, _ = pattern.build_pattern_sequence_onehot(audio_tokens)
        print(f'{seq.shape = }')
        reverted, _, mask = pattern.revert_pattern_sequence_onehot(seq.permute(0, 3, 1, 2))
        print(f'{reverted.shape = }')
        print(f'{reverted.isnan().any() = }')
        print(f'{mask = }')
        print(f'{mask.all() = }')

    testPattern()
    import pdb; pdb.set_trace()
