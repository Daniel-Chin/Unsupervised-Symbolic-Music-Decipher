from functools import lru_cache

from audiocraft.models.musicgen import MusicGen

from shared import *

VERSION = 'small'

@lru_cache(1)
def getMusicGen():
    musicGen = MusicGen.get_pretrained('facebook/musicgen-' + VERSION, device='cuda' if HAS_CUDA else 'cpu')
    return musicGen

@lru_cache(1)
def getEncodec():
    encodec = getMusicGen().compression_model
    # encodec = CompressionModel.get_pretrained(
    #     'facebook/encodec_32khz', DEVICE, 
    # )
    encodec.eval()
    return encodec

@lru_cache(1)
def getLM():
    lm = getMusicGen().lm
    lm.eval()
    return lm
