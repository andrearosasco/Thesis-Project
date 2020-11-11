import importlib
import os
from collections import OrderedDict

import torch
from PIL import Image
import numpy as np
from torchvision.transforms import transforms

model_config = OrderedDict([
    ('arch', 'cnn1'),
    # ('depth', 28),
    # ('base_channels', 16),
    # ('widening_factor', 10),
    # ('drop_rate', 0.0),
    # ('input_shape', (1, 28, 28)),
    ('n_classes', 10),
])

optim_config = OrderedDict([
    ('base_lr', 0.1),
    ('weight_decay', 0.0),
    ('momentum', 0.0),
    ('nesterov', False),
    ('milestones', []),
    ('lr_decay', 0.0),
])

data_config = OrderedDict([
    ('dataset', 'SplitMNIST'),
    ('batch_size', 128),
    ('valid', 0.2),
    ('num_workers', 4),
    ('train_transform', transforms.Compose([
            lambda x: np.array(x).reshape((1, 28, 28)),
            lambda x: torch.FloatTensor(x),
            lambda x: x / 255.0,
            transforms.Normalize(np.array([0.1307]), np.array([0.3081]))
        ])),
    ('test_transform', transforms.Compose([
            lambda x: np.array(x).reshape((1, 28, 28)),
            lambda x: torch.FloatTensor(x),
            lambda x: x / 255.0,
            transforms.Normalize(np.array([0.1307]), np.array([0.3081]))
        ]))
])

k = 2
t = 1
run_config = OrderedDict([
    ('experiment', 'main'),
    ('wandb_name', 'mnist.mod.buffer1'),
    ('checkpoint', None),
    ('epochs', 3),
    ('tasks', [list(range(k*x, k*(x + 1))) for x in range(t)]),
    ('buffer_size', 1),
    ('seed', 1234),
    ('wandb', False),
])


config = OrderedDict([
    ('model_config', model_config),
    ('optim_config', optim_config),
    ('data_config', data_config),
    ('run_config', run_config),
])

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    experiment = importlib.import_module(config['run_config']['experiment'])
    experiment.run(config)