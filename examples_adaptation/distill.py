import copy
import os
import time
import random
from collections import OrderedDict
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import higher
import wandb
from torch.optim import Optimizer
from PIL import Image
from torchviz import make_dot

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

def train(model, optimizer, criterion, train_loader, config):
    model.train()

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()

    for step, (data, targets) in enumerate(train_loader):
        data = data.to(config['device'])
        targets = targets.to(config['device'])
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

    return loss_meter.avg, accuracy_meter.avg

def test(model, criterion, test_loader, config):
    model.eval()

    loss_meter = AverageMeter()
    correct_meter = AverageMeter()

    for step, (data, targets) in enumerate(test_loader):
        data = data.to(config['device'])
        targets = targets.to(config['device'])

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
    net.to(run_config['device'])

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

        buffer, lrs = distill(net, buffer, config, criterion, trainloader)

        bufferloader = MultiLoader([buffer], batch_size=len(buffer))

        #Test
        for x, y in bufferloader:
            print_mnist(x, y)


        for epoch in range(run_config['epochs']):
            # Optim
            optimizer = torch.optim.SGD(net.parameters(), lr=lrs[epoch] if epoch < len(lrs) else lrs[-1], )  # momentum=0.4)

            buffer_loss, buffer_accuracy = train(net, optimizer, criterion, bufferloader, run_config)

            test_loss, test_accuracy = test(net, criterion, validloader, run_config)
            train_loss, train_accuracy = test(net, criterion, trainloader, run_config)

            metrics = {f'Test loss': test_loss,
                       f'Test accuracy': test_accuracy,
                       f'Train loss': train_loss,
                       f'Train accuracy': train_accuracy,
                       f'Buffer loss': buffer_loss,
                       f'Buffer accuracy': buffer_accuracy,
                       f'Epoch': epoch}

            print(metrics)
            if run_config['wandb']:
                wandb.log(metrics)


def distill(model, buffer, config, criterion, train_loader):
    run_config = config['run_config']
    distill_config = config['distill_config']

    model.train()
    eval_trainloader = copy.deepcopy(train_loader)

    buff_imgs, buff_trgs = next(iter(DataLoader(buffer, batch_size=len(buffer))))
    # buff_imgs = torch.normal(mean=0.1307, std=0.3081, size=buff_imgs.shape)
    buff_imgs, buff_trgs = buff_imgs.to(run_config['device']), buff_trgs.to(run_config['device'])
    buff_imgs.requires_grad = True

    buff_opt = torch.optim.SGD([buff_imgs], lr=distill_config['meta_lr'],) #momentum=0.9)
    model_opt = torch.optim.SGD(model.parameters(), lr=1,) #momentum=0.4)
    lr_list = []
    lr_opts = []
    for _ in range(distill_config['inner_steps']):
        lr = torch.tensor([distill_config['model_lr']], requires_grad=True, device=run_config['device'])
        lr_list.append(lr)
        lr_opts.append(torch.optim.SGD([lr], distill_config['lr_lr'],))

    for i in range(distill_config['outer_steps']):
        for step, (ds_imgs, ds_trgs) in enumerate(train_loader):
            ds_imgs = ds_imgs.to(run_config['device'])
            ds_trgs = ds_trgs.to(run_config['device'])

            with higher.innerloop_ctx(model, model_opt) as (fmodel, diffopt):
                acc_loss = None
                for j in range(distill_config['inner_steps']):
                    # First step modifies the model
                    buff_out = fmodel(buff_imgs)
                    buff_loss = criterion(buff_out, buff_trgs)
                    buff_loss = buff_loss * F.softplus(lr_list[j])
                    diffopt.step(buff_loss)
                    # Second step modifies the buffer
                    ds_out = fmodel(ds_imgs)
                    ds_loss = criterion(ds_out, ds_trgs)
                    acc_loss = acc_loss + ds_loss if acc_loss is not None else ds_loss

                    # make_dot(ds_loss,).render("attached", format="png") #params={**{f'lr {i}': lr for (i, lr) in enumerate(lr_list)}, **{'buffer images': buff_imgs}, **dict(fmodel.named_parameters())})

                    lr_opts[j].zero_grad()
                    grad, = autograd.grad(ds_loss, lr_list[j], retain_graph=True)
                    lr_list[j].grad = grad
                    lr_opts[j].step()

                    # Metrics
                    if (step + i * len(train_loader)) % int(round(len(train_loader) * distill_config['outer_steps'] * 0.05)) == \
                            int(round(len(train_loader) * distill_config['outer_steps'] * 0.05)) - 1 \
                            and j == distill_config['inner_steps'] - 1:

                        lrs = {f'Learning rate {i}': lr.item() for (i, lr) in enumerate(lr_list)}
                        test_loss, test_accuracy = test(fmodel, criterion, eval_trainloader, run_config)
                        metrics = {f'Distill train loss': test_loss, f'Distill train accuracy': test_accuracy, f'Distill step': step + i * len(train_loader)}

                        if run_config['wandb']:
                            wandb.log({**metrics, **lrs})

                        print(metrics)

                buff_opt.zero_grad()
                acc_loss.backward()
                buff_opt.step()

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
    lr_list = [lr.item() for lr in lr_list]

    return Buffer(aux, len(aux)), lr_list