#!/usr/bin/env python3
from subprocess import Popen
from itertools import count

def main():
    with open('./midi_process_template.sbatch', 'r', encoding='utf-8') as f:
        template = f.read()
    
    all_dirs = set('0123456789abcdef')
    n_dirs_per_process = 1
    for i in count():
        dirs = []
        for _ in range(n_dirs_per_process):
            try:
                dirs.append(all_dirs.pop())
            except KeyError:
                break
        if not dirs:
            break
        auto_sb_filename = f'./auto_midi_process_{i}.sbatch'
        with open(auto_sb_filename, 'w', encoding='utf-8') as f:
            f.write(template.replace(
                '{JOB_NAME}', str(i), 
            ).replace(
                '{SELECT_DIRS}', ' '.join(dirs),
            ))
    
        with Popen(['sbatch', auto_sb_filename]) as p:
            p.wait()

if __name__ == '__main__':
    main()
