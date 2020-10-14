import torch.nn as nn
import torch.optim as optim
import time
import torchvision.transforms as transforms
import torch
import wandb
from tqdm import tqdm
# user-defined modules
from configs.config1 import conf
import model
from dataset.SplitMNIST import SplitMNIST
from utils import utils
from valid import valid

# Check config
if __file__.split('/')[-1] != conf['exp'] + '.py':
    print('Warning: conf[exp] doesn\'t match the file name')
# WandB
wandb.init(project="meta-cl")
wandb.config.update(conf)
# Model
net = getattr(model, conf['model']).Model()
net.to('cuda')
net.train()
net.apply(utils.init_weights)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=conf['lr'])

# Dataset
transform = transforms.Compose(
    [
     lambda x: x.reshape((28, 28)),
     lambda x: torch.FloatTensor(x),
     lambda x: x.unsqueeze(0)
     ])

trainset = SplitMNIST(type='train', valid=conf['valid'], transform=transform)
validset = SplitMNIST(type='valid', valid=conf['valid'], transform=transform)

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


        running_loss = running_loss / len(trainloader)
        valid_loss = valid(net, validset)
        wandb.log({'Train loss': running_loss, 'Valid accuracy': valid_loss})

    # print((time.time() - start))

print('Finished Training')