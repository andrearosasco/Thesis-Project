import time
import random
import torch
import wandb
from contflame.data.datasets import SplitCIFAR100, SplitCIFAR10
from contflame.data.utils import MultiLoader, Buffer
from torch import nn
import numpy as np
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, CIFAR10
from torchvision.transforms import transforms
from tqdm import tqdm

import model


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def train(task_id, epoch, model, optimizer, criterion, train_loader, run_config):
    model.train()

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    start = time.time()

    for step, (data, targets) in enumerate(train_loader):
        data = data.cuda()
        targets = targets.cuda()

        optimizer.zero_grad()

        with autocast():
            outputs = model(data)
            loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        _, preds = torch.max(outputs, dim=1)

        loss_ = loss.item()
        correct_ = preds.eq(targets).sum().item()
        num = data.size(0)

        accuracy = correct_ / num

        loss_meter.update(loss_, num)
        accuracy_meter.update(accuracy, num)

    elapsed = time.time() - start

    if run_config['wandb']:
        wandb.log({f'Train loss {task_id}': loss_meter.avg,
                   f'Train accuracy {task_id}': accuracy_meter.avg,
                   f'Epoch {task_id}': epoch})


def test(task_id, i, epoch, model, criterion, test_loader, run_config):

    model.eval()

    loss_meter = AverageMeter()
    correct_meter = AverageMeter()
    start = time.time()

    for step, (data, targets) in enumerate(test_loader):
        data = data.cuda()
        targets = targets.cuda()


        with torch.no_grad() and autocast():
            outputs = model(data)
            loss = criterion(outputs, targets)

        _, preds = torch.max(outputs, dim=1)

        loss_ = loss.item()
        correct_ = preds.eq(targets).sum().item()
        num = data.size(0)

        loss_meter.update(loss_, num)
        correct_meter.update(correct_, 1)

    accuracy = correct_meter.sum / len(test_loader.dataset)

    elapsed = time.time() - start

    if run_config['wandb']:
        wandb.log({f'Test loss {task_id}-{i}': loss_meter.avg,
                   f'Test accuracy {task_id}-{i}': accuracy,
                   f'Epoch {task_id}': epoch})

    return accuracy


def run(config):
    run_config = config['run_config']
    optim_config = config['optim_config']
    model_config = config['model_config']
    data_config = config['data_config']

    if run_config['wandb']:
        wandb.init(project="meta-cl")
        wandb.config.update(config)
    # Reproducibility
    seed = config['run_config']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Model
    net = getattr(model, model_config['arch']).Model(model_config)
    net.cuda()

    if run_config['wandb']:
        wandb.watch(net, log='gradients')

    # Optim
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=optim_config['base_lr'],
        momentum=optim_config['momentum'],
        weight_decay=optim_config['weight_decay'],
        nesterov=optim_config['nesterov']
        )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=optim_config['milestones'],
        gamma=optim_config['lr_decay'])

    # # Pretraining on CIFAR10
    #
    #
    # pretrainset = SplitCIFAR100(dset='train', valid=data_config['valid'], transform=data_config['train_transform'],
    #                              classes=list(range(10, 100)))
    # pretrainloader = DataLoader(pretrainset, batch_size=data_config['batch_size'], shuffle=True,
    #                             pin_memory=True, num_workers=data_config['num_workers'])
    # prevalidset = SplitCIFAR100(dset='valid', valid=data_config['valid'], transform=data_config['test_transform'],
    #                              classes=list(range(10, 100)))
    # prevalidloader = DataLoader(prevalidset, batch_size=data_config['batch_size'], shuffle=False,
    #                             pin_memory=True, num_workers=data_config['num_workers'])
    #
    # for epoch in tqdm(range(1, optim_config['epochs'] + 1)):
    #     scheduler.step()
    #
    #     train(0, epoch, net, optimizer, criterion, pretrainloader, run_config)
    #     test(0, 0, epoch, net, criterion, prevalidloader, run_config)
    #
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer,
    #     milestones=optim_config['milestones'],
    #     gamma=optim_config['lr_decay'])
    #
    # net.freeze()

    # Training
    memories = []
    validloaders = []
    for task_id, task in enumerate(run_config['tasks'], 1):
        # Data
        trainset = SplitCIFAR10(dset='train', valid=data_config['valid'], transform=data_config['train_transform'],
                                 classes=task)
        validset = SplitCIFAR10(dset='valid', valid=data_config['valid'], transform=data_config['test_transform'],
                                 classes=task)

        trainloader = MultiLoader([trainset] + memories, batch_size=data_config['batch_size'])
        validloaders.append(DataLoader(validset, batch_size=data_config['batch_size'], shuffle=False,
                                       pin_memory=True, num_workers=data_config['num_workers']))

        memories.append(Buffer(SplitCIFAR10(dset='train', valid=data_config['valid'],
                                 classes=task),
                                run_config['buffer_size'], transform=data_config['train_transform']))

        for epoch in tqdm(range(1, optim_config['epochs'] + 1)):
            scheduler.step()

            train(task_id, epoch, net, optimizer, criterion, trainloader, run_config)
            for i, vl in enumerate(validloaders, 1):
                test(task_id, i, epoch, net, criterion, vl, run_config)

        # state = OrderedDict([
        #     ('config', config),
        #     ('state_dict', model.state_dict()),
        #     ('optimizer', optimizer.state_dict()),
        #     ('epoch', epoch),
        #     ('accuracy', accuracy),
        # ])
        # model_path = os.path.join(outdir, 'model_state.pth')
        # torch.save(state, model_path)