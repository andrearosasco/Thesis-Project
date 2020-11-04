import importlib
import os
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
    ('base_lr', 0.1),
    ('weight_decay', 0.0005),
    ('momentum', 0.9),
    ('nesterov', True),
    ('milestones', [60, 70]),
    ('lr_decay', 0.2),
])

data_config = OrderedDict([
    ('dataset', 'CIFAR100'),
    ('batch_size', 128),
    ('valid', 0.2),
    ('num_workers', 4),
])

run_config = OrderedDict([
    ('experiment', 'pretrain'),
    ('wandb_name', 'new_pretrain'),
    ('save', 'model_state.ptc'),
    ('epochs', 80),
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