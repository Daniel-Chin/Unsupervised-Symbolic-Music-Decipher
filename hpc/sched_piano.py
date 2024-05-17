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
    with SbatchContext('./scancel_all.sh') as sbatch:
        sbatch(auto_sb_filename)

if __name__ == '__main__':
    main()
