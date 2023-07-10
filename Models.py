import torch
from torch import nn,optim
import torchvision
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=5,
            stride=2,
            padding=2),
            nn.BatchNorm2d(16),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(16,32,5,2,2),
            nn.BatchNorm2d(32),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.fc1=nn.Sequential(
            nn.Linear(32*14*14,120),
            nn.Sigmoid()
        )
        self.fc2=nn.Sequential(
            nn.Linear(120,84),
            nn.Sigmoid(),
            nn.Linear(84,5)
        )
    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        x=self.fc2(x)
        return x

# AlexNet
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=11,
                stride=4,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(32,32,3,padding=1),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.fc1=nn.Sequential(
            nn.Linear(1152,120),
            nn.ReLU(),
            nn.Linear(120,5)
        )
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        return x

# 使用预训练方法
def getNet():
    model=AlexNet()
    return model
