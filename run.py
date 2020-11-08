import os
import subprocess


configs = ['./split/cifar100_1.py']

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    for c in configs:
        print(f'Executing {c}')
        proc = subprocess.Popen(['python', c])
        proc.wait()