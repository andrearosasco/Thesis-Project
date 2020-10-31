import importlib
import os
from collections import OrderedDict

from PIL import Image
from torchvision.transforms import transforms
import numpy as np

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
    ('epochs', 80),
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
    ('train_transform', transforms.Compose(
        [

            lambda x: Image.fromarray(x.reshape((3, 32, 32)).transpose((1, 2, 0))),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0, np.array([63.0, 62.1, 66.7]) / 255.0)
        ])),
    ('test_transform', transforms.Compose(
        [
            lambda x: Image.fromarray(x.reshape((3, 32, 32)).transpose((1, 2, 0))),
            transforms.ToTensor(),
            transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0, np.array([63.0, 62.1, 66.7]) / 255.0)
        ]))
])

run_config = OrderedDict([
    ('experiment', 'regular'),
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
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    experiment = importlib.import_module(config['run_config']['experiment'])
    experiment.run(config)