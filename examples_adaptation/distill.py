import copy
import os
import time
import random
from collections import OrderedDict
import matplotlib.pyplot as plt

import torch
import higher
import wandb
from torch.optim import Optimizer
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

def test(model, criterion, test_loader):
    model.eval()

    loss_meter = AverageMeter()
    correct_meter = AverageMeter()

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

    return loss_meter.avg, accuracy


def run(config):
    run_config = config['run_config']
    distill_config = config['distill_config']
    model_config = config['model_config']
    data_config = config['data_config']

    if run_config['wandb']:
        wandb.init(project="meta-cl", name=run_config['wandb_name'])
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


    # if run_config['wandb']:
    #     wandb.watch(net, log='gradients')


    # Training
    memories = []

    Dataset = getattr(datasets, data_config['dataset'])
    for task_id, task in enumerate(run_config['tasks'], 1):
        # Data
        validset = Dataset(dset='test', valid=data_config['valid'], transform=data_config['test_transform'],
                                 classes=task)
        validloader = DataLoader(validset, batch_size=data_config['batch_size'], shuffle=False,
                                       pin_memory=True, num_workers=data_config['num_workers'])

        trainset = Dataset(dset='train', valid=data_config['valid'], transform=data_config['train_transform'],
                     classes=task)
        trainloader = DataLoader(trainset, batch_size=data_config['batch_size'], shuffle=True,
                                       pin_memory=True, num_workers=data_config['num_workers'])

        buffer = None
        for t in task:
            aux = Dataset(dset='train', valid=data_config['valid'], transform=data_config['train_transform'],
                               classes=[t])
            buffer = Buffer(aux, run_config['buffer_size']) if buffer is None else buffer + Buffer(aux, run_config['buffer_size'])

        buffer, lr = distill(net, buffer, distill_config, criterion, trainloader)

        bufferloader = MultiLoader([buffer], batch_size=len(buffer))

        #Test
        for x, y in bufferloader:
            print_mnist(x, y)

        # Optim
        optimizer = torch.optim.SGD(net.parameters(), lr=lr,) #momentum=0.4)


        for epoch in range(run_config['epochs']):
            train(task_id, epoch, net, optimizer, criterion, bufferloader, run_config)

            test_loss, test_accuracy = test(net, criterion, validloader)
            train_loss, train_accuracy = test(net, criterion, trainloader)

            if run_config['wandb']:
                wandb.log({f'Test loss': test_loss,
                           f'Test accuracy': test_accuracy,
                           f'Train loss': train_loss,
                           f'Train accuracy': train_accuracy,
                           f'Epoch': epoch})


def distill(model, buffer, config, criterion, train_loader):
    model.train()
    eval_trainloader = copy.deepcopy(train_loader)

    buff_imgs, buff_trgs = next(iter(DataLoader(buffer, batch_size=len(buffer))))
    # buff_imgs = torch.normal(mean=0.1307, std=0.3081, size=buff_imgs.shape)
    buff_imgs, buff_trgs = buff_imgs.cuda(), buff_trgs.cuda()
    buff_imgs.requires_grad = True

    model_lr = config['model_lr']
    buff_opt = torch.optim.SGD([buff_imgs], lr=config['meta_lr'],) #momentum=0.9)
    model_opt = torch.optim.SGD(model.parameters(), lr=model_lr,) #momentum=0.4)


    for i in range(config['outer_steps']):
        for step, (ds_imgs, ds_trgs) in enumerate(train_loader):
            ds_imgs = ds_imgs.cuda()
            ds_trgs = ds_trgs.cuda()

            with higher.innerloop_ctx(model, model_opt) as (fmodel, diffopt):
                acc_loss = None
                for j in range(config['inner_steps']):
                    # First step modifies the model
                    buff_out = fmodel(buff_imgs)
                    buff_loss = criterion(buff_out, buff_trgs)
                    diffopt.step(buff_loss)
                    # Second step modifies the buffer
                    ds_out = fmodel(ds_imgs)
                    ds_loss = criterion(ds_out, ds_trgs)
                    acc_loss = acc_loss + ds_loss if acc_loss is not None else ds_loss

                    # Metrics
                    if (step + i * len(train_loader)) % int(round(len(train_loader) * config['outer_steps'] * 0.05)) == int(round(len(train_loader) * config['outer_steps'] * 0.05)) - 1 \
                            and j == config['inner_steps'] - 1:
                        test_loss, test_accuracy = test(fmodel, criterion, eval_trainloader)
                        metrics = {f'Distill train loss': test_loss, f'Distill train accuracy': test_accuracy, f'Distill step': step + i * len(train_loader)}
                        wandb.log(metrics)
                        print(metrics)

                acc_loss.backward()
                buff_opt.step()
                buff_opt.zero_grad()

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