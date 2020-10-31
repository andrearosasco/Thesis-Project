from __future__ import print_function
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch
import wandb
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# user-defined modules
from configs.config6 import conf
import model
from contflame.data.datasets import SplitCIFAR100
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.models as models
from contflame.data.utils import Buffer, MultiLoader
from torch.utils.data import DataLoader
from valid import valid
import os

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

os.environ['CUDA_VISIBLE_DEVICES']='1'
# os.environ['WANDB_MODE']='dryrun'
# Check config
if __file__.split('/')[-1] != conf['exp'] + '.py':
    print('Warning: conf[exp] doesn\'t match the executed file name')
# WandB
wandb.init(project="meta-cl")
wandb.config.update(conf)
# Model
net = model.resnet.Model()#getattr(model, conf['model']).Model() # #
# net.apply(init_weights)
net.to('cuda')
net.train()
wandb.watch(net, log='gradients')

criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=conf['lr'])
optimizer = torch.optim.SGD(
        net.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0005,
        nesterov=True)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[20, 30, 50],
        gamma=0.2)
# Dataset
# t = transforms.Compose([transforms.ToTensor()])
test_transform = transforms.Compose(
    [
        # lambda x: torch.FloatTensor(x),
        # lambda x: x.reshape((3, 32, 32)),
        # lambda x: x / 255.,
        transforms.ToTensor(),
        transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                    np.array([63.0, 62.1, 66.7]) / 255.0)
])


train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                    np.array([63.0, 62.1, 66.7]) / 255.0)
])

classes = []
# for x in range(100):
#     classes.append(x)

# warmup = CIFAR10(root="./", train=True, download=True, transform=t)
# warmup_valid = CIFAR10(root="./", train=False, download=True, transform=t)

# trainset = SplitCIFAR100(dset='train', valid=conf['valid'], transform=transform, classes=None)
# validset = SplitCIFAR100(dset='valid', valid=conf['valid'], transform=transform, classes=None)
trainset = CIFAR100(root="./", train=True, download=True, transform=train_transform)
validset = CIFAR100(root="./", train=False, download=True, transform=test_transform)


# warmuploader = DataLoader(warmup, batch_size=conf['mb'], shuffle=True, pin_memory=True)
trainloader = DataLoader(trainset, batch_size=conf['mb'], shuffle=True, pin_memory=True)

# batch, _ = next(iter(trainloader))
# x = batch[0].numpy()
# plt.imsave('image.png', np.transpose(x, (1, 2, 0)))
# exit(0)

# batch = None
# for x, y, in tqdm(trainloader):
#     x = x.reshape(-1, 3, 32*32).std(2)
#     batch = x if batch == None else torch.cat((batch, x), dim=0)
#     # print(batch.shape)
#
# print(batch.mean(0))
# Training
# for epoch in tqdm(range(5)):  # loop over the dataset multiple times
#     # start = time.time()
#     running_loss = 0.0
#
#     for i, data in enumerate(warmuploader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         outputs = net(inputs.to('cuda'))
#
#         loss = criterion(outputs, labels.to('cuda'))
#         loss.backward()
#         optimizer.step()
#
#         # print statistics
#         running_loss += loss.item()
#
#         if i % int(len(warmuploader)*0.05) == int(len(warmuploader)*0.05) - 1 or i == 0:
#             running_loss = running_loss / len(warmuploader)
#             mean_acc = 0
#
#             valid_acc = valid(net, warmup_valid)
#             mean_acc += valid_acc
#             wandb.log({'Wu accuracy': valid_acc,
#                        'Wu Mini batch': i + len(warmuploader) * epoch})
#         #endif
#
#     # print((time.time() - start))

# print(f'Finished warm up')
# net.freeze_conv()

for epoch in tqdm(range(conf['epochs'])):
    # start = time.time()
    training_loss = []
    scheduler.step()

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
        training_loss += [loss.item()]

        if i % int(len(trainloader)*0.1) == int(len(trainloader)*0.1) - 1 or i == 0:
            valid_acc = valid(net, validset)
            wandb.log({'Valid accuracy': valid_acc,
                       'Train log loss': sum(training_loss) / len(training_loss),
                       'Minibatch': i + len(trainloader) * epoch,
                       'Epoch': epoch})
            training_loss = []
        # endif

    # print((time.time() - start))

print(f'Finished Training')