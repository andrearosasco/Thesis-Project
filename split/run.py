import subprocess


configs = [f'split_config{x}.py' for x in range(6, 7)]
configs.append('pretrain_config.py')
configs.append('split_config7.py')

if __name__ == '__main__':
    for c in configs:
        proc = subprocess.Popen(['python', c])
        proc.wait()