from audiocraft.models.encodec import CompressionModel, HFEncodecCompressionModel

from shared import *

def getEncodec():
    encodec = CompressionModel.get_pretrained(
        'facebook/encodec_32khz', DEVICE, 
    )
    assert isinstance(encodec, HFEncodecCompressionModel)
    encodec.eval()
    return encodec
