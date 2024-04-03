import torch
from torch import Tensor
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

N_OCTAVES = 12
MODULAR_ENCODING_LEN = N_OCTAVES + 12

def modularEncode(key: Tensor, is_soft_not_hard: bool):
    # element in key: [0, 127]
    k = key.detach().view(-1).to(torch.int)
    length, = k.shape
    ladder = torch.arange(length)
    result = torch.zeros((length, MODULAR_ENCODING_LEN))
    result[ladder, N_OCTAVES + torch.remainder(k, 12)] = 1.0
    octave = k / 12.0
    left = torch.floor(octave).to(torch.int)
    if is_soft_not_hard:
        right = left + 1
        assert (right < N_OCTAVES).all(), f'{key.max() = } must < 128'
        left_k = right - octave
        right_k = octave - left
        result[ladder, left] = left_k
        result[ladder, right] = right_k
    else:
        result[ladder, left] = 1.0
    return result.view(*key.shape, MODULAR_ENCODING_LEN)

def inspect():
    n_keys = 128
    keys = torch.arange(n_keys).view(4, -1)
    fig, axes = plt.subplots(2, 1)
    for is_soft, ax in zip((False, True), axes):
        ax: Axes
        m = modularEncode(keys, is_soft)
        print(f'{m.shape = }')
        t = m.view(-1, MODULAR_ENCODING_LEN).numpy()
        ax.imshow(t, aspect='auto', interpolation='nearest')
        ax.set_xlabel('Encoding dim')
        ax.set_ylabel('Pitch')
        ax.set_title('Soft' if is_soft else 'Hard')
    plt.show()

if __name__ == '__main__':
    inspect()
