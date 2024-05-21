import os
from os import path
import json

from tqdm import tqdm

from shared import *

TRIM_TO_SIZE = 10000

def main():
    for set_dir in (PIANO_MONKEY_DATASET_DIR, PIANO_ORACLE_DATASET_DIR):
        n_dirs_left = len(LA_DATASET_DIRS)
        n_datapoints_left = TRIM_TO_SIZE
        for d in tqdm(LA_DATASET_DIRS):
            with open(path.join(set_dir, d, 'index.json'), 'r', encoding='utf-8') as f:
                original = json.load(f)
            n_keep = n_datapoints_left // n_dirs_left
            n_datapoints_left -= n_keep
            n_dirs_left -= 1
            
            assert n_keep <= len(original), (n_keep, len(original))
            to_keep, to_del = original[:n_keep], original[n_keep:]
            for i in tqdm(to_del, desc=d):
                for suffix in (
                    '.mid', '_encodec_recon.wav', '_encodec_tokens.pt', 
                    '_griffin_lim.wav', '_log_spectrogram.pt', 
                    '_score.pt', '_synthed.wav', 
                ):
                    os.remove(path.join(set_dir, d, i + suffix))
            with open(path.join(set_dir, d, 'index.json'), 'w', encoding='utf-8') as f:
                json.dump(to_keep, f)

if __name__ == '__main__':
    main()
