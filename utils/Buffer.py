from __future__ import print_function
import random
from dataset.SplitMNIST import SplitMNIST


class Buffer:

    def __init__(self, ds, dim):
        l = len(ds)
        r = []

        for i in range(dim):
            r.append(ds[i])

        for i in range(dim, l):
            h = random.randint(0, i)
            if h < dim:
                r[h] = ds[i]
        self.r = r

    def __getitem__(self, item):
        return self.r[item]

    def __len__(self):
        return len(self.r)

    def add(self, buffer, l):
        b = list(buffer)

        for i in range(l):
            self.r = self.r + b



if __name__ == '__main__':
    ds = SplitMNIST(type='train', valid=0.2, tasks=[0])
    buff = Buffer(ds, 10)