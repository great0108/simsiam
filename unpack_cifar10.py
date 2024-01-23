import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import matplotlib.pyplot as plt
import matplotlib.image as image
import numpy as np
import os
from multiprocessing import freeze_support

if __name__ == "__main__":
    freeze_support()
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./raw_Cifar10', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./raw_Cifar10', train=False, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=32)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=32)
    

    import os

    num_classes = 10
    number_per_class = {}

    for i in range(num_classes):
        number_per_class[i] = 0

    def custom_imsave(img, label, train=True):
        if train:
            path = 'Cifar10/train/' + str(label) + '/'
        else:
            path = 'Cifar10/val/' + str(label) + '/'

        if not os.path.exists(path):
            os.makedirs(path)
        
        img = img.numpy()
        img = np.transpose(img, (1, 2, 0))
        image.imsave(path + str(number_per_class[label]) + '.jpg', img)
        number_per_class[label] += 1

    def process():
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            print("[ Current Batch Index: " + str(batch_idx) + " ]")
            for i in range(inputs.size(0)):
                custom_imsave(inputs[i], targets[i].item())

        for batch_idx, (inputs, targets) in enumerate(test_loader):
            print("[ Current Batch Index: " + str(batch_idx) + " ]")
            for i in range(inputs.size(0)):
                custom_imsave(inputs[i], targets[i].item(), train=False)

    process()

    # check well done
    from torchvision.datasets import ImageFolder

    train_dataset = ImageFolder(root='./Cifar10/train', transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=32)
    test_dataset = ImageFolder(root='./Cifar10/val', transform=transform_train)
    test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=32)
