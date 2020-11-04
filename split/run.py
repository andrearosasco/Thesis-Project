import os
import subprocess


configs = [f'split_config{x}.py' for x in range(1, 3)]

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    for c in configs:
        print(f'Executing {c}')
        proc = subprocess.Popen(['python', c])
        proc.wait()