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
from configs.config2 import conf
import model
from dataset.SplitMNIST import SplitMNIST
from utils import utils
from utils.Buffer import Buffer
from valid import valid
import os

os.environ['CUDA_VISIBLE_DEVICES']='2'
# Check config
if __file__.split('/')[-1] != conf['exp'] + '.py':
    print('Warning: conf[exp] doesn\'t match the executed file name')
# WandB
wandb.init(project="meta-cl")
wandb.config.update(conf)
# Model
net = getattr(model, conf['model']).Model()
net.apply(utils.init_weights)
net.to('cuda')
net.train()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=conf['lr'])

# Dataset
transform = transforms.Compose(
    [
     lambda x: x.reshape((28, 28)),
     lambda x: torch.FloatTensor(x),
     lambda x: x.unsqueeze(0)
     ])
memories = []
validsets = []
for t in conf['tasks']:

    trainset = SplitMNIST(type='train', valid=conf['valid'], transform=transform, tasks=[t])
    validsets.append(SplitMNIST(type='valid', valid=conf['valid'], transform=transform, tasks=[t]))

    new_mem = Buffer(trainset, conf['buffer'])

    l = len(trainset)
    for m in memories:
        trainset.add(m, int(l / len(m)))
    memories.insert(0, new_mem[:])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=conf['mb'], shuffle=True, pin_memory=True)

    # Training
    for epoch in tqdm(range(conf['epochs'])):  # loop over the dataset multiple times
        # start = time.time()
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
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

    print('Finished Training ' + str(t+1))