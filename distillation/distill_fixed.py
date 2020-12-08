import copy
import random
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

w = 0
def print_images(imgs, trgs, mean, std):
    global w
    for img, trg in zip(imgs, trgs):
        label = trg.item()

        std = [std[0] for _ in range(img.size(0))] if len(std) == 1 else std
        mean = [mean[0] for _ in range(img.size(0))] if len(mean) == 1 else mean

        img = img.cpu().detach().numpy()

        for i in range(img.shape[0]):
            img[i] = img[i] * std[i] + mean[i]

        img = img * 255
        img = np.transpose(img, (1, 2, 0))
        img = np.squeeze(img)
        img = img.astype(np.uint8)

        plt.imsave(f'./img{label}.png', img)
        w += 1

def initialize_weights(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)

def train(model, optimizer, criterion, train_loader, config):
    model.train()

    correct = 0
    loss_sum = 0
    tot = 0

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

        with torch.no_grad() and autocast():
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
        wandb.init(project="PROJECT-NAME", name=log_config['wandb_name'])
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
    # print(net.feature_extractor[0].weight)
    #
    # net.apply(initialize_weights)
    # print(net.feature_extractor[0].weight)
    #
    # exit()

    # Data
    Dataset = getattr(datasets, data_config['dataset'])

    validset = Dataset(dset='valid', valid=data_config['valid'], transform=data_config['test_transform'], classes=run_config['task'])
    validloader = DataLoader(validset, batch_size=param_config['batch_size'], shuffle=False, pin_memory=True, num_workers=data_config['num_workers'])

    trainset = Dataset(dset='train', valid=data_config['valid'], transform=data_config['train_transform'], classes=run_config['task'])
    trainloader = DataLoader(trainset, batch_size=param_config['batch_size'], shuffle=True, pin_memory=True, num_workers=data_config['num_workers'])

    buffer = None
    for t in run_config['task']:
        aux = Dataset(dset='train', valid=data_config['valid'], transform=data_config['train_transform'], classes=[t])
        buffer = Buffer(aux, param_config['buffer_size']) if buffer is None else buffer + Buffer(aux, param_config['buffer_size'])

    buffer, lrs = distill(net, buffer, config, criterion, trainloader)

    if log_config['print']:
        print(np.log(1 + np.exp(lrs)))

    if run_config['save'] is not None:
        with open(run_config['save'], 'wb') as file:
            dill.dump(OrderedDict([
                ('config', config),
                ('dataset', buffer),
                ('lrs', lrs),
                ('init', net.state_dict())
            ]), file)

    bufferloader = MultiLoader([buffer], batch_size=len(buffer))

    if log_config['images']:
        mean, std = data_config['test_transform'].transforms[-1].mean, data_config['test_transform'].transforms[-1].std
        for x, y in bufferloader:
            print_images(x, y, mean, std)

    for _ in range(5):
        print()
        net.apply(initialize_weights)

        for epoch in range(param_config['epochs']):
            lr = lrs[epoch] if epoch < len(lrs) else lrs[-1]
            optimizer = torch.optim.SGD(net.parameters(), lr=np.log(1 + np.exp(lr)), )

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
            if log_config['print']:
                print(metrics)

            if log_config['wandb']:
                wandb.log(metrics)


def distill(model, buffer, config, criterion, train_loader):
    run_config = config['run_config']
    param_config = config['param_config']
    log_config = config['log_config']

    model.train()
    eval_trainloader = copy.deepcopy(train_loader)

    buff_imgs, buff_trgs = next(iter(DataLoader(buffer, batch_size=len(buffer))))
    # De-comment to use random noise instead of real images. The results are simillar
    # buff_imgs = torch.normal(mean=0.1307, std=0.3081, size=buff_imgs.shape)
    buff_imgs, buff_trgs = buff_imgs.to(run_config['device']), buff_trgs.to(run_config['device'])
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

            with higher.innerloop_ctx(model, model_opt) as (fmodel, diffopt):
                acc_loss = None
                for j in range(param_config['inner_steps']):
                    # Update the model
                    buff_out = fmodel(buff_imgs)
                    buff_loss = criterion(buff_out, buff_trgs)
                    buff_loss = buff_loss * torch.log(1 + torch.exp(lr_list[j]))
                    diffopt.step(buff_loss)

                    # Update the buffer (actually we just record the loss and update it outside the inner loop)
                    ds_out = fmodel(ds_imgs)
                    ds_loss = criterion(ds_out, ds_trgs)
                    acc_loss = acc_loss + ds_loss if acc_loss is not None else ds_loss

                    # Update the lrs
                    lr_opts[j].zero_grad()
                    grad, = autograd.grad(ds_loss, lr_list[j], retain_graph=True)
                    lr_list[j].grad = grad
                    lr_opts[j].step()

                    # Metrics (20 samples of loss and accuracy at the last inner step)
                    if (step + i * len(train_loader)) % int(round(len(train_loader) * param_config['outer_steps'] * 0.05)) == \
                            int(round(len(train_loader) * param_config['outer_steps'] * 0.05)) - 1 \
                            and j == param_config['inner_steps'] - 1:

                        lrs = {f'Learning rate {i}': np.log(1 + np.exp(lr.item())) for (i, lr) in enumerate(lr_list)}
                        test_loss, test_accuracy = test(fmodel, criterion, eval_trainloader, run_config)
                        metrics = {f'Distill train loss': test_loss, f'Distill train accuracy': test_accuracy, f'Distill step': step + i * len(train_loader)}

                        if log_config['wandb']:
                            wandb.log({**metrics, **lrs})

                        if log_config['print']:
                            print(metrics)

                buff_opt.zero_grad()
                acc_loss.backward()
                buff_opt.step()

    aux = []
    buff_imgs, buff_trgs = buff_imgs.cpu(), buff_trgs.cpu()
    for i in range(buff_imgs.size(0)):
        aux.append([buff_imgs[i], buff_trgs[i]])
    lr_list = [lr.item() for lr in lr_list]

    return Buffer(aux, len(aux)), lr_list