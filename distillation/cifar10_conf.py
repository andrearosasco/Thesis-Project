import importlib
import os
from collections import OrderedDict

import torch
import numpy as np
from PIL import Image
from torchvision.transforms import transforms

model_config = OrderedDict([
    ('arch', 'lenet5'),
    ('n_classes', 10),
    # Next entries are required for using the Wide-ResNet
    # ('depth', 28),
    # ('base_channels', 16),
    # ('widening_factor', 10),
    # ('drop_rate', 0.0),
    ('input_shape', (3, 32, 32)),
])

data_config = OrderedDict([
    ('dataset', 'CIFAR10'),
    ('valid', 0.2),
    ('num_workers', 4),
    ('train_transform', transforms.Compose([
        lambda x: Image.fromarray(x.reshape((3, 32, 32)).transpose((1, 2, 0))),
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(np.array([0.5]), np.array( [0.5]))
    ])),
    ('test_transform', transforms.Compose([
        lambda x: Image.fromarray(x.reshape((3, 32, 32)).transpose((1, 2, 0))),
        transforms.ToTensor(),
        transforms.Normalize(np.array([0.5]), np.array([0.5]))
        ]))
])

run_config = OrderedDict([
    ('experiment', 'distill'),  # This configuration will be executed by distill.py
    ('device', 'cuda'),
    ('save', 'distill6'),  # Path for the distilled dataset
    ('seed', 1234),
])

log_config = OrderedDict([
    ('wandb', True),
    ('wandb_name', 'cifar10.distill.mb128,outer640'),
    ('print', True),
    ('images', True), # Save the distilled images
])

param_config = OrderedDict([
    ('epochs', 3),  # Training epoch performed by the model on the distilled dataset
    ('meta_lr', 0.1),  # Learning rate for distilling images
    ('model_lr', 0.01),  # Base learning rate for the model
    ('lr_lr', 0.1),  # Learning rate for the lrs of the model at each optimization step
    ('outer_steps', 980),  # Distillation epochs
    ('inner_steps', 3),  # Optimization steps of the model
    ('batch_size', 128),  # Minibatch size used during distillation
    ('buffer_size', 1),  # Number of examples per class kept in the buffer
])

config = OrderedDict([
    ('model_config', model_config),
    ('param_config', param_config),
    ('data_config', data_config),
    ('run_config', run_config),
    ('log_config', log_config),
])

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    experiment = importlib.import_module(config['run_config']['experiment'])
    experiment.run(config)