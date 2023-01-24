# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from matplotlib import pyplot as plt
import os
import numpy as np

# Training settings
batch_size = 64

# MNIST Dataset
train_dataset = datasets.MNIST(root='.Assignment2/data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='.Assignment2/data/',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)



def imshow(dataset):
    """Helper function to display an image"""
    label = []
    dirs = 'train_image'
    for title,i in enumerate(train_dataset):
        if i == 100:
            break
        img = i[0]
        label.append(i[1])
        img = img.clip(0, 1)
        img = img.cpu().numpy()
        plt.imshow(img.transpose(1, 2, 0))
        plt.axis("off")
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        path = dirs + '/' + str(title) + '.jpg'
        plt.savefig(path)
    np.save("label.npy", label)
    # plt.title(title)
    # plt.show()

imshow(train_dataset)

