import dill
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import contflame.data.datasets as datasets
import torch
from torch import nn
from contflame.data.utils import MultiLoader
from torch.utils.data import DataLoader

import model

def train(model, optimizer, criterion, train_loader, config):
    model.train()

    correct = 0
    loss_sum = 0
    tot = 0

    for step, (data, targets) in enumerate(train_loader):
        data = data.to(config['device'])
        targets = targets.to(config['device'])
        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        _, preds = torch.max(outputs, dim=1)

        loss_sum += loss.item() * data.size(0)
        tot += data.size(0)
        correct += preds.eq(targets).sum().item()

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

        with torch.no_grad():
            outputs = model(data)
            loss = criterion(outputs, targets)

        _, preds = torch.max(outputs, dim=1)

        loss_sum += loss.item() * data.size(0)
        tot += data.size(0)
        correct += preds.eq(targets).sum().item()

    accuracy = correct / tot
    loss = loss_sum / tot

    return loss, accuracy

w = 0
def print_images(imgs, trgs, mean, std):
    global w
    for img, trg in zip(imgs, trgs):
        print(trg)

        std = [std[0] for _ in range(img.size(0))] if len(std) == 1 else std
        mean = [mean[0] for _ in range(img.size(0))] if len(mean) == 1 else mean

        for i in range(img.size(0)):
            img[i] = img[i] * std[i] + mean[i]

        img = img * 255
        img = img.cpu().detach().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = np.squeeze(img)
        img = img.astype(np.uint8)

        plt.imsave(f'./img{w}.png', img)
        w += 1

if __name__ == '__main__':
    with open('distill6', 'rb') as file:
        checkpoint = dill.load(file)

    config = checkpoint['config']

    run_config = config['run_config']
    model_config = config['model_config']
    param_config = config['param_config']
    data_config = config['data_config']
    log_config = config['log_config']

    criterion = nn.CrossEntropyLoss()

    net = getattr(model, model_config['arch']).Model(model_config)
    net.load_state_dict(checkpoint['init'])
    net.to(run_config['device'])

    Dataset = getattr(datasets, data_config['dataset'])
    testset = Dataset(dset='test', transform=data_config['test_transform'])
    testloader = DataLoader(testset, batch_size=256, shuffle=False, pin_memory=True, num_workers=data_config['num_workers'])


    buffer = checkpoint['dataset']
    lrs = checkpoint['lrs']

    bufferloader = MultiLoader([buffer], batch_size=len(buffer))

    mean, std = data_config['test_transform'].transforms[-1].mean, data_config['test_transform'].transforms[-1].std
    for x, y in bufferloader:
        print_images(x, y, mean, std)

    for epoch in range(param_config['epochs']):
        lr = lrs[epoch] if epoch < len(lrs) else lrs[-1]
        optimizer = torch.optim.SGD(net.parameters(), lr=np.log(1 + np.exp(lr)), )

        buffer_loss, buffer_accuracy = train(net, optimizer, criterion, bufferloader, run_config)
        test_loss, test_accuracy = test(net, criterion, testloader, run_config)

        metrics = {f'Test loss': test_loss,
                   f'Test accuracy': test_accuracy,
                   f'Buffer loss': buffer_loss,
                   f'Buffer accuracy': buffer_accuracy,
                   f'Epoch': epoch}
        print(metrics)