import importlib
import os
import sys
from collections import OrderedDict

model_config = OrderedDict([
    ('arch', 'wide_resnet'),
    ('depth', 28),
    ('base_channels', 16),
    ('widening_factor', 10),
    ('drop_rate', 0.0),
    ('input_shape', (1, 3, 32, 32)),
    ('n_classes', 100),
])

optim_config = OrderedDict([
    ('base_lr', 0.001),
    ('weight_decay', 0.0005),
    ('momentum', 0.9),
    ('nesterov', True),
    ('milestones', []),
    ('lr_decay', 0.0),
])

data_config = OrderedDict([
    ('dataset', 'CIFAR10'),
    ('batch_size', 128),
    ('valid', 0.2),
    ('num_workers', 4),
])
k = 2
t = 5
run_config = OrderedDict([
    ('experiment', 'split'),
    ('wandb_name', 'new_pt-lower_lr'),
    ('checkpoint', 'model_state.ptc'),
    ('epochs', 80),
    ('tasks', [list(range(k*x, k*(x + 1))) for x in range(t)]),
    ('buffer_size', 100),
    ('seed', 1234),
    ('wandb', True),
])


config = OrderedDict([
    ('model_config', model_config),
    ('optim_config', optim_config),
    ('data_config', data_config),
    ('run_config', run_config),
])

if __name__ == '__main__':

    experiment = importlib.import_module(config['run_config']['experiment'])
    experiment.run(config)