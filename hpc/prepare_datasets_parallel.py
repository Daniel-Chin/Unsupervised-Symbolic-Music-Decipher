#!/usr/bin/env python3
import argparse

from parallel_sbatch import SbatchContext

DIRS = '0123456789abcdef'
TO_PREPARE = [
    ('monkey', 10000 // len(DIRS)), 
    ('oracle', 128 // len(DIRS)),
]

def main(stage: str):
    with open('./prepare_datasets_templare.sbatch', 'r', encoding='utf-8') as f:
        template = f.read()
    
    with SbatchContext('./scancel_all.sh') as sbatch:
        for which_set, n_datapoints in TO_PREPARE:
            for select_dir in DIRS:
                job_identifier = which_set + '_' + select_dir
                auto_sb_filename = f'./auto_prep_dataset_{job_identifier}.sbatch'
                with open(auto_sb_filename, 'w', encoding='utf-8') as f:
                    f.write(template.replace(
                        '{N_GPU}', '0' if stage == 'cpu' else '1', 
                    ).replace(
                        '{LOG_ID}', job_identifier, 
                    ).replace(
                        '{PARTITION}', 'parallel' if stage == 'cpu' else 'aquila,gpu', 
                    ).replace(
                        '{CONSTRAINT}', 'cpu' if stage == 'cpu' else '3090', 
                    ).replace(
                        '{WHICH_SET}', which_set, 
                    ).replace(
                        '{STAGE}', stage, 
                    ).replace(
                        '{SELECT_DIR}', select_dir, 
                    ).replace(
                        '{N_DATAPOINTS}', str(n_datapoints), 
                    ))

                sbatch(auto_sb_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--stage', type=str, required=True, choices=['cpu', 'gpu'],
    )
    args = parser.parse_args()
    main(
        args.stage, 
    )
