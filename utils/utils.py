import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

def print_img(data):
    data = np.array(data)
    data = data.reshape((3, 32, 32)).transpose((1, 2, 0))
    plt.imshow(data)
    plt.show()

import torch

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)