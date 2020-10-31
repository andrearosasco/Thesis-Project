from __future__ import print_function
import torch.nn as nn
import torch.optim as optim
import time
import torchvision.transforms as transforms
import torch
import wandb
from tqdm import tqdm
import pandas as pd
# user-defined modules
from configs.config4 import conf
import model
from contflame.data.datasets import PermutedMNIST
from contflame.data.utils import Buffer, MultiLoader
from valid import valid
import os

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

os.environ['CUDA_VISIBLE_DEVICES']='1'
# Check config
if __file__.split('/')[-1] != conf['exp'] + '.py':
    print('Warning: conf[exp] doesn\'t match the executed file name')
# WandB
wandb.init(project="meta-cl")
wandb.config.update(conf)
# Model
net = getattr(model, conf['model']).Model()
net.apply(init_weights)
net.to('cuda')
net.train()
wandb.watch(net, log='all')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=conf['lr'])

# Dataset
transform = transforms.Compose([
        lambda x: torch.FloatTensor(x),
        lambda x: x.reshape((28, 28)),
        lambda x: x.unsqueeze(0)
     ])
memories = []
validsets = []

for t in conf['tasks']:
    trainset = PermutedMNIST(root='./', dset='train', valid=conf['valid'], transform=transform, task=t, tile=conf['tile'])
    validsets.append(
        PermutedMNIST(root='./', dset='valid', valid=conf['valid'], transform=transform, task=t, tile=conf['tile'])
    )

    f = len(memories) + 1
    decay = [len(memories)/(x) for x in range(f, 0, -1)]
    trainloader = MultiLoader([trainset] + memories, batch_size=conf['mb'])
    memories.append(Buffer(trainset, conf['buffer']))

    # Training
    for epoch in range(conf['epochs']):
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs.to('cuda'))
            loss = criterion(outputs, labels.to('cuda'))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % int(len(trainloader)*0.05) == int(len(trainloader)*0.05) - 1 or i == 0:
                running_loss = running_loss / len(trainloader)
                mean_acc = 0
                for s, v in enumerate(validsets):
                    valid_acc = valid(net, v)
                    mean_acc += valid_acc
                    wandb.log({'Valid accuracy '+str(t+1)+'-'+str(s+1): valid_acc,
                               'Mini batch ' + str(t + 1): i + len(trainloader) * epoch})
                wandb.log({'Mean Valid accuracy '+str(t+1): mean_acc / (s+1)})

        # print((time.time() - start))