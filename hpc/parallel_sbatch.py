from typing import List
from contextlib import contextmanager
from subprocess import Popen, PIPE

@contextmanager
def SbatchContext(rm_script_filename: str):
    job_ids: List[str] = []

    def callback(auto_sbatch_filename: str):
        with Popen(['sbatch', auto_sbatch_filename], stdout=PIPE) as p:
            p.wait()
            assert p.stdout is not None
            output = p.stdout.read().decode('utf-8')
        job_id = output.split('Submitted batch job ', 1)[1].strip()
        job_ids.append(job_id)
    
    try:
        yield callback
    finally:
        with open(rm_script_filename, 'w', encoding='utf-8') as f:
            for job_id in job_ids:
                print(f'scancel {job_id}', file=f)
