# create tar.gz for experiments

import os
from os import path
import subprocess as sp

def main():
    os.chdir('../experiments')
    doAll()

def tar(exp_dir_name: str):
    print('taring', exp_dir_name, '...')
    with sp.Popen([
        'tar', '-zcf', exp_dir_name + '.tar.gz', exp_dir_name, 
    ]) as p:
        # print('  waiting...')
        p.wait()
    # print('  exit')

def doAll():
    list_dir = os.listdir()
    # print(f'{list_dir = }')
    all_gz = set()
    all_dir = set()
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
    selected = []
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
    for dir in selected:
        tar(dir)

if __name__ == '__main__':
    main()
