# create tar.gz for experiments

import os
from os import path
import subprocess as sp
import typing as tp

def tar(src: str, dest: str):
    print('taring', src, '...')
    with sp.Popen([
        'tar', '-zcf', dest, src, 
    ]) as p:
        # print('  waiting...')
        p.wait()
    # print('  exit')

def main():
    os.chdir('../experiments')

    list_dir = os.listdir()
    # print(f'{list_dir = }')
    all_gz : tp.Set[str] = set()
    all_dir: tp.Set[str] = set()
    IGNORE = ['.gitignore', '.', '..']
    for node in list_dir:
        # print(node)
        if node in IGNORE:
            continue
        base, ext = path.splitext(node)
        # print(base, ext)
        if path.isdir(node):
            all_dir.add(path.normpath(node))
        elif ext.lower() == '.gz':
            base_, tar_ = path.splitext(base)
            assert tar_.lower() == '.tar'
            all_gz.add(path.normpath(base_))
        else:
            print('Warning: unknown file:', node)
    # print(all_dir)
    # print(all_gz)
    available = [x for x in all_dir if x not in all_gz]
    print('available:', *available, sep='\n')
    print()
    print('Select. Enter empty string to tar all.')
    selected: tp.List[str] = []
    while True:
        op = input('>').strip()
        if not op:
            break
        if op not in available:
            print('Invalid selection.')
            continue
        selected.append(op)
    if not selected:
        selected = available
    jobs = [
        (x, x + '.tar.gz') for x in selected
    ]
    for job in jobs:
        tar(*job)
    print('For your convenience to copy:')
    names = ','.join([dest for src, dest in jobs])
    print(
        'scpdan /scratch/nq285/usmd/experiments/'
        '\\{' + names + '\\}'
        ' ~/neuralAVH/unsupervised_symbolic_music_decipher/experiments', 
    )

if __name__ == '__main__':
    main()
