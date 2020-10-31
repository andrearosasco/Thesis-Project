from __future__ import print_function
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch
import wandb
from tqdm import tqdm
# user-defined modules
from configs.config1 import conf
import model
from contflame.data.datasets import SplitMNIST
from contflame.data.utils import Buffer, MultiLoader
from torch.utils.data import DataLoader
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
transform = transforms.Compose(
    [
        lambda x: torch.FloatTensor(x),
        lambda x: x.reshape((1, 28, 28)),
])

trainset = SplitMNIST(dset='train', valid=conf['valid'], transform=transform, classes=None)
validset = SplitMNIST(dset='valid', valid=conf['valid'], transform=transform, classes=None)

trainloader = DataLoader(trainset, batch_size=conf['mb'], shuffle=True, pin_memory=True)
# Training
for epoch in tqdm(range(conf['epochs'])):
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

            valid_acc = valid(net, validset)
            mean_acc += valid_acc
            wandb.log({'Valid accuracy': valid_acc,
                       'Mini batch': i + len(trainloader) * epoch})
        #endif

    # print((time.time() - start))

print('Finished Training')