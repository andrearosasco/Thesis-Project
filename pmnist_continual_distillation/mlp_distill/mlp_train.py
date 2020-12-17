import copy
import random
import sys
from collections import OrderedDict
import matplotlib.pyplot as plt
import dill

import torch
import higher
import wandb

import contflame.data.datasets as datasets
from contflame.data.utils import MultiLoader, Buffer
from torch import nn
import numpy as np
from torch.cuda.amp import autocast
from torch import autograd
from torch.utils.data import DataLoader
import model


def print_images(imgs, trgs, mean, std, name, perm=None):
    imgs = copy.deepcopy(imgs)

    for img, trg in zip(imgs, trgs):
        label = trg.item()
        print(label)

        img = img.cpu().detach().numpy()

        if perm is not None: img = perm.unpermute(img)

        img = img.reshape((1, 28, 28))

        std = [std[0] for _ in range(img.shape[0])] if len(std) == 1 else std
        mean = [mean[0] for _ in range(img.shape[0])] if len(mean) == 1 else mean

        for i in range(img.shape[0]):
            img[i] = img[i] * std[i] + mean[i]

        img = img * 255
        img = np.transpose(img, (1, 2, 0))
        img = np.squeeze(img)
        img = img.astype(np.uint8)

        wandb.log({f'{name}_{label}':[wandb.Image(img, caption=f"{label}")]})

        # plt.imsave(f'./img{w}_{label}.png', img)

def initialize_weights(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)

class Train:

    def __init__(self, optimizer, criterion, train_loader, config):
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.train_loader = train_loader
        self.iter = enumerate(train_loader)

    def __call__(self, model):
        model.train()

        run_config = self.config['run_config']

        correct = 0
        loss_sum = 0
        tot = 0

        try:
            step, (data, targets) = next(self.iter)
        except StopIteration:
            self.iter = enumerate(self.train_loader)
            step, (data, targets) = next(self.iter)

        # for step, (data, targets) in enumerate(self.train_loader):

        data = data.to(run_config['device'])
        targets = targets.to(run_config['device'])
        self.optimizer.zero_grad()

        # with autocast():
        outputs = model(data)
        loss = self.criterion(outputs, targets)
        loss.backward()

        self.optimizer.step()

        _, preds = torch.max(outputs, dim=1)

        loss_sum += loss.item() * data.size(0)
        correct += preds.eq(targets).sum().item()
        tot += data.size(0)

        accuracy = correct / tot
        loss = loss_sum / tot

        return loss, accuracy

def test(model, criterion, test_loader, config):
    model.eval()

    correct = 0
    loss_sum = 0
    tot = 0

    for step, (data, targets) in enumerate(test_loader):
        data = data.to(config['device'])
        targets = targets.to(config['device'])

        with torch.no_grad(): #  and autocast():
            outputs = model(data)
            loss = criterion(outputs, targets)

        _, preds = torch.max(outputs, dim=1)

        loss_sum += loss.item() * data.size(0)
        correct += preds.eq(targets).sum().item()
        tot += data.size(0)

    accuracy = correct / tot
    loss = loss_sum / tot

    return loss, accuracy


def run(config):

    run_config = config['run_config']
    model_config = config['model_config']
    param_config = config['param_config']
    data_config = config['data_config']
    log_config = config['log_config']

    if log_config['wandb']:
        wandb.init(project="distill_mlp", name=log_config['wandb_name'])
        wandb.config.update(config)

    # Reproducibility
    seed = run_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Model
    net = getattr(model, model_config['arch']).Model(model_config)
    net.to(run_config['device'])
    net.apply(initialize_weights)

    # Data
    Dataset = getattr(datasets, data_config['dataset'])

    # Training
    memories = []
    validloaders = []

    for task_id, task in enumerate(run_config['tasks'], 0):

        validset = Dataset(dset='valid', valid=data_config['valid'], transform=data_config['test_transform'], task=task)
        validloaders.append(DataLoader(validset, batch_size=param_config['batch_size'], shuffle=False, pin_memory=True, num_workers=data_config['num_workers']))
        trainset = Dataset(dset='train', valid=data_config['valid'], transform=data_config['train_transform'], task=task)
        trainloader = DataLoader(trainset, batch_size=param_config['batch_size'], shuffle=True, pin_memory=True, num_workers=data_config['num_workers'])

        optimizer = torch.optim.SGD(net.parameters(), lr=param_config['model_lr'], )

        buffer = None
        for c in range(model_config['n_classes']):
            ds = list(filter(lambda x: x[1] == c,
                             Dataset(dset='train', valid=data_config['valid'],
                                     transform=data_config['train_transform'], task=task)))
            buffer = Buffer(ds, param_config['buffer_size']) if buffer is None else buffer + Buffer(ds,
                                                                                                    param_config[
                                                                                                        'buffer_size'])
        buffer, lrs = distill(net, buffer, config, criterion, trainloader, task_id)
        bufferloader = MultiLoader([buffer], len(buffer))

        train = Train(optimizer, criterion, bufferloader, config)

        steps = len(bufferloader) * param_config['epochs']
        for step in range(steps):

            buffer_loss, buffer_accuracy = train(net)

            if step % int(round(steps * 0.05)) == int(round(steps * 0.05)) - 1 or step == 0:
                valid_m = {}
                for i, vl in enumerate(validloaders):

                    test_loss, test_accuracy = test(net, criterion, vl, run_config)
                    valid_m = {**valid_m, **{f'Test loss {i}': test_loss,
                               f'Test accuracy {i}': test_accuracy,}}

                train_m = {f'Buffer loss': buffer_loss,
                           f'Buffer accuracy': buffer_accuracy,
                           f'Step': step + steps * task_id}
                if log_config['print']:
                    print({**valid_m, **train_m})
                if log_config['wandb']:
                    wandb.log({**valid_m, **train_m})


            # mean, std = [0.1307], [0.3081]
            # for x, y in MultiLoader([buffer], batch_size=len(buffer)):
            #     print_images(x, y, mean, std)

def distill(model, buffer, config, criterion, train_loader, id):
    model = copy.deepcopy(model) # to avoid all the re-initializations don't affect the real model

    run_config = config['run_config']
    param_config = config['param_config']
    log_config = config['log_config']

    model.train()
    eval_trainloader = copy.deepcopy(train_loader)

    buff_imgs, buff_trgs = next(iter(DataLoader(buffer, batch_size=len(buffer))))
    # Uncomment to use random noise instead of real images. The results are simillar
    # buff_imgs = torch.normal(mean=0.1307, std=0.3081, size=buff_imgs.shape)
    buff_imgs, buff_trgs = buff_imgs.to(run_config['device']), buff_trgs.to(run_config['device'])
    buff_imgs = buff_imgs
    buff_imgs.requires_grad = True

    buff_opt = torch.optim.SGD([buff_imgs], lr=param_config['meta_lr'],)
    model_opt = torch.optim.SGD(model.parameters(), lr=1,)
    lr_list = []
    lr_opts = []
    for _ in range(param_config['inner_steps']):
        lr = np.log(np.exp([param_config['model_lr']]) - 1)  # Inverse of softplus (so that the starting learning rate is actually the specified one)
        lr = torch.tensor(lr, requires_grad=True, device=run_config['device'])
        lr_list.append(lr)
        lr_opts.append(torch.optim.SGD([lr], param_config['lr_lr'],))

    for i in range(param_config['outer_steps']):
        for step, (ds_imgs, ds_trgs) in enumerate(train_loader):
            ds_imgs = ds_imgs.to(run_config['device'])
            ds_trgs = ds_trgs.to(run_config['device'])

            init_batch = get_batch(model, config)
            acc_loss = None
            epoch_loss = [None for _ in range(param_config['inner_steps'])]

            for r, sigma in enumerate(init_batch):
                model.load_state_dict(sigma)
                with higher.innerloop_ctx(model, model_opt) as (fmodel, diffopt):
                    for j in range(param_config['inner_steps']):
                        # Update the model
                        # with autocast():
                        buff_out = fmodel(buff_imgs)
                        buff_loss = criterion(buff_out, buff_trgs)
                        buff_loss = buff_loss * torch.log(1 + torch.exp(lr_list[j]))
                        diffopt.step(buff_loss)

                        # Update the buffer (actually we just record the loss and update it outside the inner loop)
                        # with autocast():
                        ds_out = fmodel(ds_imgs)
                        ds_loss = criterion(ds_out, ds_trgs)

                        epoch_loss[j] = epoch_loss[j] + ds_loss if epoch_loss[j] is not None else ds_loss
                        acc_loss = acc_loss + ds_loss if acc_loss is not None else ds_loss

                        # Metrics (20 samples of loss and accuracy at the last inner step)
                        if (step + i * len(train_loader)) % int(round(len(train_loader) * param_config['outer_steps'] * 0.05)) == \
                                int(round(len(train_loader) * param_config['outer_steps'] * 0.05)) - 1 \
                                and j == param_config['inner_steps'] - 1:# and r == 0:

                            lrs = {f'Learning rate {i} - {id}': np.log(np.exp(lr.item()) + 1) for (i, lr) in enumerate(lr_list)}
                            test_loss, test_accuracy = test(fmodel, criterion, eval_trainloader, run_config)
                            metrics = {f'Distill train loss {id}': test_loss, f'Distill train accuracy {id}': test_accuracy, f'Distill step {id}': step + i * len(train_loader)}

                            if log_config['wandb']:
                                wandb.log({**metrics, **lrs})

                            if log_config['print']:
                                print(metrics)

            # Update the lrs
            for j in range(param_config['inner_steps']):
                lr_opts[j].zero_grad()
                grad, = autograd.grad(epoch_loss[j], lr_list[j], retain_graph=True)
                lr_list[j].grad = grad
                lr_opts[j].step()

            buff_opt.zero_grad()
            acc_loss.backward()
            buff_opt.step()

    aux = []
    buff_imgs, buff_trgs = buff_imgs.cpu(), buff_trgs.cpu()
    for i in range(buff_imgs.size(0)):
        aux.append([buff_imgs[i], buff_trgs[i]])
    lr_list = [np.log(1 + np.exp(lr.item())) for lr in lr_list]

    return Buffer(aux, len(aux)), lr_list

def get_batch(model, config):
    model = copy.deepcopy(model)
    # model.train()
    #
    # data_config = config['data_config']
    # param_config = config['param_config']
    # run_config = config['run_config']
    #
    # Dataset = getattr(datasets, data_config['dataset'])
    # optimizer = torch.optim.SGD(model.parameters(), lr=param_config['model_lr'], )

    param_batch = []
    param_batch.append(copy.deepcopy(model.apply(initialize_weights).state_dict()))
    # Without deepcopy every parametrization in param_batch is the same...
    # If it doesn't work, use rehearsal

    # for task in run_config['tasks']:
    #     ds = Dataset(dset='train', valid=data_config['valid'], transform=data_config['train_transform'], classes=task)
    #     tl = DataLoader(ds, batch_size=param_config['batch_size'], shuffle=True, pin_memory=True,
    #                              num_workers=data_config['num_workers'])
    #
    #     train(model, optimizer, nn.CrossEntropyLoss(), tl, run_config)
    #     param_batch.append(copy.deepcopy(model.state_dict()))

    return param_batch