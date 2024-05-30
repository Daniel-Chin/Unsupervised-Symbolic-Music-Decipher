#!/usr/bin/env python3

from meta_shared import *
from parallel_sbatch import SbatchContext

DIRS = '0123456789abcdef'
TO_PREPARE = [
    ('monkey', 100000 // len(DIRS)), 
    ('oracle', 100000 // len(DIRS)),
]

def main():
    with open('./prepare_datasets_template.sbatch', 'r', encoding='utf-8') as f:
        template = f.read()
    
    with SbatchContext('./scancel_all.sh') as sbatch:
        for which_set, n_datapoints in TO_PREPARE:
            for select_dir in DIRS:
                job_identifier = which_set + '_' + select_dir
                auto_sb_filename = f'./auto_prep_dataset_{job_identifier}.sbatch'
                with open(auto_sb_filename, 'w', encoding='utf-8') as f:
                    f.write(template.replace(
                        '{N_GPU}', '1', 
                    ).replace(
                        '{LOG_ID}', job_identifier, 
                    ).replace(
                        '{PARTITION}', 'aquila,gpu' if on_low_not_high else 'sfscai', 
                    ).replace(
                        '{CONSTRAINT}', '3090' if on_low_not_high else 'a800', 
                    ).replace(
                        '{WHICH_SET}', which_set, 
                    ).replace(
                        '{SELECT_DIR}', select_dir, 
                    ).replace(
                        '{N_DATAPOINTS}', str(n_datapoints), 
                    ))

                sbatch(auto_sb_filename)

if __name__ == '__main__':
    main()
