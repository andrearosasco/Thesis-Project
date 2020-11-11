import copy
import os
import time
import random
from collections import OrderedDict
import matplotlib.pyplot as plt

import torch
import wandb
from PIL import Image
import contflame.data.datasets as datasets
from contflame.data.utils import MultiLoader, Buffer
from torch import nn
import numpy as np
from torch.cuda.amp import autocast
from torch import autograd
from torch.utils.data import DataLoader
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

w = 0
def print_mnist(imgs, trgs):
    global w
    for img, trg in zip(imgs, trgs):
        print(trg)


        img = img * 0.3081 + 0.1307  # unnormalize
        img = img * 255
        npimg = img.cpu().squeeze().detach().numpy()
        npimg = npimg.astype(np.uint8)
        # npimg = np.transpose(npimg, (1, 2, 0))

        plt.imsave(f'./img{w}.png', npimg)
        w += 1

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
    print(correct_meter.sum)
    print(len(test_loader.dataset))

    elapsed = time.time() - start

    if run_config['wandb']:
        wandb.log({f'Test loss {task_id}-{i}': loss_meter.avg,
                   f'Test accuracy {task_id}-{i}': accuracy,
                   f'Epoch {task_id}': epoch})

        wandb.log({f'Test loss {i}': loss_meter.avg,
                   f'Test accuracy {i}': accuracy,
                   f'Epoch': epoch + task_id * run_config['epochs']})

    return accuracy

import torch.nn.functional as F


def run(config):
    run_config = config['run_config']
    optim_config = config['optim_config']
    model_config = config['model_config']
    data_config = config['data_config']

    if run_config['wandb']:
        wandb.init(project="meta-cl", name=run_config['wandb_name'])
        wandb.config.update(config)
    # Reproducibility
    seed = 1243#config['run_config']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Model
    net = getattr(model, model_config['arch']).Model(model_config)
    net.cuda()

    init_config = copy.deepcopy(net.state_dict())

    # if run_config['wandb']:
    #     wandb.watch(net, log='gradients')


    # Training
    memories = []
    validloaders = []
    Dataset = getattr(datasets, data_config['dataset'])
    for task_id, task in enumerate(run_config['tasks'], 1):
        # Data
        validset = Dataset(dset='test', valid=data_config['valid'], transform=data_config['test_transform'],
                                 classes=task)
        validloaders.append(DataLoader(validset, batch_size=data_config['batch_size'], shuffle=False,
                                       pin_memory=True, num_workers=data_config['num_workers']))

        trainset = Dataset(dset='train', valid=data_config['valid'], transform=data_config['train_transform'],
                     classes=task)
        trainloader = DataLoader(trainset, batch_size=data_config['batch_size'], shuffle=True,
                                       pin_memory=True, num_workers=data_config['num_workers'])
        buffer = None
        for t in task:
            aux = Dataset(dset='train', valid=data_config['valid'], transform=data_config['train_transform'],
                               classes=[t])
            buffer = Buffer(aux, run_config['buffer_size']) if buffer is None else buffer + Buffer(aux, run_config['buffer_size'])

        # for img_lr in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]:
        # buffer, lr = distill(net, init_config, buffer, 0.1, criterion, trainloader)
        memories.append(buffer)

        bufferloader = MultiLoader(memories, batch_size=data_config['batch_size'])

        # Optim
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
        net.load_state_dict(init_config)

        for epoch in range(60):
            train(task_id, epoch, net, optimizer, criterion, bufferloader, run_config)
            for i, vl in enumerate(validloaders, 1):
                accuracy = test(task_id, i, epoch, net, criterion, vl, run_config)
            print(f'epoch {epoch} -> accuracy {accuracy}')


def distill(model, init_config, buffer, img_lr, criterion, train_loader):
    model = copy.deepcopy(model)
    model.train()

    buff_imgs, buff_trgs = next(iter(DataLoader(buffer, batch_size=len(buffer))))
    buff_imgs, buff_trgs = buff_imgs.cuda(), buff_trgs.cuda()
    buff_imgs.requires_grad = True

    buff_opt = torch.optim.SGD([buff_imgs], lr=img_lr)
    model_lr = 0.1

    torch.set_printoptions(precision=10)
    # init_param = copy.deepcopy(init_config)

    for step, (ds_imgs, ds_trgs) in enumerate(train_loader):
        model.load_state_dict(init_config)
        model_opt = torch.optim.SGD(model.parameters(), lr=model_lr)

        ds_imgs = ds_imgs.cuda()
        ds_trgs = ds_trgs.cuda()

        # First step modifies the model
        with autograd.detect_anomaly():
            buff_opt.zero_grad()
            buff_out = model(buff_imgs)
            buff_loss = criterion(buff_out, buff_trgs)
            buff_loss.backward()
            model_opt.step()
            model_opt.zero_grad()
            # Second step modifies the buffer
            ds_out = model(ds_imgs)
            ds_loss = criterion(ds_out, ds_trgs)
            ds_loss.backward()
            buff_opt.step()

    # print(buff_imgs.grad)
    print_mnist(buff_imgs, buff_trgs)
    # exit(0)
        # wandb.log({'Full Train Accuracy': torch.max(ds_out, dim=1)[1].eq(ds_trgs).sum().item() / ds_trgs.size(0)})
    # correct = 0
    # tot = 0
    # with torch.no_grad():
    #     for step, (ds_imgs, ds_trgs) in enumerate(train_loader):
    #         correct += torch.max(ds_out.cuda(), dim=1)[1].eq(ds_trgs.cuda()).sum().item()
    #         tot += ds_imgs.size(0)
        # print(f'lr {img_lr} -> train accuracy {correct / tot}')


    # with torch.no_grad():
    #     torch.set_printoptions(precision=10)
    #     final_param = model.state_dict()
    #     for k, v in final_param.items():
    #         print(f'{k}: {(init_param[k] - v).float().abs().mean([x for x in range(len(v.shape))])}')

    aux = []
    buff_imgs, buff_trgs = buff_imgs.cpu(), buff_trgs.cpu()
    for i in range(buff_imgs.size(0)):
        aux.append([buff_imgs[i], buff_trgs[i]])

    return Buffer(aux, len(aux)), model_lr