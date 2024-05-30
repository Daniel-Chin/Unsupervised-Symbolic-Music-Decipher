from os import path
import shutil
import argparse
import uuid
import socket

from meta_shared import *
from parallel_sbatch import SbatchContext

def main(stage: AVH_Stage):
    stage_ = stage.value.lower()
    job_id = uuid.uuid4()
    py_filename = f'auto_train_{stage_}_{job_id}.py'
    shutil.copyfile(f'../main_train_{stage_}.py', path.join('..', py_filename))
    with open('./train_template.sbatch', 'r', encoding='utf-8') as f:
        template = f.read()
    auto_sb_filename = f'./auto_train_{stage_}_{job_id}.sbatch'
    
    with open(auto_sb_filename, 'w', encoding='utf-8') as f:
        f.write(template.replace(
            '{PY_FILENAME}', py_filename, 
        ).replace(
            '{STAGE}', stage.name,
        ).replace(
            '{PARTITION}', 'aquila,gpu' if on_low_not_high else 'sfscai', 
        ).replace(
            '{CONSTRAINT}', '3090' if on_low_not_high else 'a800', 
        ))
    slurm_ids = []
    with SbatchContext('./scancel_all.sh', slurm_ids) as sbatch:
        sbatch(auto_sb_filename)
    slurm_id, = slurm_ids
    print(f'{slurm_id = }')
    print('To monitor logs:')
    print(f'tail -f *{slurm_id}*')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=AVH_Stage, required=True, choices=[*AVH_Stage])
    args = parser.parse_args()
    main(args.stage)
