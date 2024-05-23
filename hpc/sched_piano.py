from os import path
import shutil
import uuid

from parallel_sbatch import SbatchContext

def main():
    job_id = uuid.uuid4()
    py_filename = f'auto_train_piano_{job_id}.py'
    shutil.copyfile('../main_train_piano.py', path.join('..', py_filename))
    with open('./train_piano_template.sbatch', 'r', encoding='utf-8') as f:
        template = f.read()
    auto_sb_filename = f'./auto_train_piano_{job_id}.sbatch'
    with open(auto_sb_filename, 'w', encoding='utf-8') as f:
        f.write(template.replace('{PY_FILENAME}', py_filename))
    slurm_ids = []
    with SbatchContext('./scancel_all.sh', slurm_ids) as sbatch:
        sbatch(auto_sb_filename)
    slurm_id, = slurm_ids
    print(f'{slurm_id = }')

if __name__ == '__main__':
    main()
