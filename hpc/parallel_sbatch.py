from typing import List
from contextlib import contextmanager
from io import StringIO
from subprocess import Popen

@contextmanager
def SbatchContext(rm_script_filename: str):
    job_ids: List[str] = []

    def callback(auto_sbatch_filename: str):
        io = StringIO()
        with Popen(['sbatch', auto_sbatch_filename], stdout=io) as p:
            p.wait()
        io.seek(0)
        job_id = io.read().split('Submitted batch job ', 1)[1].strip()
        job_ids.append(job_id)
    
    try:
        yield callback
    finally:
        with open(rm_script_filename, 'w', encoding='utf-8') as f:
            for job_id in job_ids:
                print(f'scancel {job_id}', file=f)
