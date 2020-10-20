import torchvision.transforms as transforms
import torch

def valid(model, dataset):
    model.to('cuda')
    model.eval()

    validloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, pin_memory=True)

    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(validloader):
            images, labels = data
            images = images.to('cuda')
            labels = labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    model.train()
    return (100 * correct / total)