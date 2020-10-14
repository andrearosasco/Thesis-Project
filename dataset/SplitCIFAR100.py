from pathlib import Path
from torch.utils.data import Dataset
from functools import reduce
import pickle
import numpy as np
import requests
import tarfile
import os


class SplitCIFAR100(Dataset):
    """Split CIFAR-100 dataset."""

    def __init__(self, root='.', meta=False, train=False, tasks=None, transform=None):
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
        if not (root / 'cifar-100-python').is_dir():
            print('Downloading dataset...')
            url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
            r = requests.get(url)

            with (root/'cifar-100-python.tar.gz').open('wb') as f:
                f.write(r.content)

            tarfile.open(str(root / 'cifar-100-python.tar.gz'), "r:gz").extractall()
            (root / "cifar-100-python.tar.gz").unlink()
            print('Done!')

        # open and unpickle train or test set
        if train:
            with (root / 'cifar-100-python/train').open('rb') as fo:
                data = pickle.load(fo)
        else:
            with (root / 'cifar-100-python/test').open('rb') as fo:
                data = pickle.load(fo)

        # transform dictionary in list of (x, y) pairs
        data = np.array([[d, l] for d, l in zip(data[b'data'], data[b'fine_labels'])])

        split = []
        for i in range(0, 100, 2):
            split.append(list(filter(lambda x: x[1] in [i, i + 1], data)))
        split = np.array(split)

        if meta:
            split = split[:-20]
        else:
            split = split[-20:]

        if tasks != None and max(tasks) >= len(split):
            print('Error: task index higher then number of tasks (#tasks=' + str(len(split) - 1) + ')')
        # select the required tasks
        if tasks == None:
            tasks = range(len(split))
        self.t = reduce(lambda x, y: np.concatenate((x, y)), split[tasks])

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        (x, y) = self.t[idx]

        x = x.reshape((32, 32, 3))
        if self.transform:
            x = self.transform(x)

        return (x, y)

if __name__ == '__main__':
    ds = SplitCIFAR100(meta=True, train=True)
    print(next(iter(ds)))

