from pathlib import Path
from torch.utils.data import Dataset
from mnist import MNIST
from functools import reduce
import pickle
import numpy as np
import requests
import gzip
import os


class SplitMNIST(Dataset):
    """Split MNIST"""

    urls = ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']
    fnames = ['train-data', 'train-labels', 'test-data', 'test-labels']

    def __init__(self, root='.', meta=None, type='train', valid=0.0, tasks=None, transform=None):
        """
        Args:
            root (string): Directory with containing the cifar-100-python directory.
            meta (bool): True - returns the meta-training dataset, False - returns the meta-test dataset
            train (bool): True - returns the training set, False - returns the test set.
                Training and test sets are internal to the meta-trainiing and meta-test dataset.
            tasks (int): Select the tasks to keep in the dataset. If None all the tasks are used.
        """
        root = Path(root)
        self.transform = transform

        # download and uncompress dataset if not present
        if not (root/'mnist-python').is_dir():
            print('Downloading dataset...')
            (root/'mnist-python').mkdir()

            for url, fname in zip(self.urls, self.fnames):
                r = requests.get(url)
                fn = url.split('/')[-1]

                with (root/'mnist-python'/fn).open('wb') as f:
                    f.write(r.content)
                with gzip.open(str(root/'mnist-python'/fn), 'rb') as f:
                    data = f.read()
                with (root/'mnist-python'/fn[:-3]).open('wb') as f:
                    f.write(data)
                (root/'mnist-python'/fn).unlink()
            print('Done!')

        # open and unpickle train or test set
        mndata = MNIST(str(root / 'mnist-python'))
        f = mndata.load_testing() if type == 'test' else mndata.load_training()
        imgs, labels = f

        # transform dictionary in list of (x, y) pairs
        data = np.array([np.array([np.array(d), np.array(l)]) for d, l in zip(imgs, labels)])

        split = []
        for i in range(0, 10, 2):
            split.append(list((filter(lambda x: x[1] in [i, i + 1], data))))
        split = np.array(split)

        if type != 'test':
            split = np.array(list(map(lambda x: x[:len(x) - int(len(x)*valid)] if type == 'train' else x[-int(valid*len(x)):], split)))

        if meta is not None:
            if meta:
                split = split[:-2]
            else:
                split = split[-2:]

        # select the specified tasks
        if tasks != None and max(tasks) >= len(split):
            print('Error: task index higher then number of tasks (#tasks=' + str(len(split) - 1) + ')')
        # select all the tasks (joint training)
        if tasks == None:
            tasks = range(len(split))

        self.t = reduce(lambda x, y: np.concatenate((x, y)), split[tasks])
        self.l = len(self.t)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        (x, y) = self.t[idx]

        if self.transform:
            x = self.transform(x)

        return (x, y)

    def add(self, buffer, l):
        b = list(buffer)

        for i in range(l):
            self.t = self.t + b

if __name__ == '__main__':
    ds = SplitMNIST(type='train', valid=0.2, tasks=[0, 1, 3])
    print(len(ds))

