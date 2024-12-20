from __future__ import print_function
from tqdm import tqdm
from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import StepLR
from utils import MNIST_DATA, FitEvaluate


SEED = 10
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if cuda else "cpu")
print(device)



# Final Model
# Train Phase transformations
train_transforms = transforms.Compose([
                                      transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
                                      transforms.Resize((28, 28)),
                                      transforms.RandomRotation((-15.0, 15.0), fill=(1,)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                       ])

# Test Phase transformations
test_transforms = transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                       ])

#train & test data
mnist_data = MNIST_DATA(train_transforms,test_transforms)

#data stats
print("\n MNIST data stats \n")
print(mnist_data.stats())
#Visualizing Sample Images from the MNIST Dataset
print("\n Visualizing Sample Images from the MNIST Dataset \n")
mnist_data.showimages(num_of_images=60)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.convblock1 = nn.Sequential(
            self.depthwise_separable_conv(1, 10, kernel_size=3, padding=1),  # output = 28x28, receptive field = 3x3
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=1, bias=False, dilation=2),  # output = 28x28, receptive field = 7x7
            nn.BatchNorm2d(10),
            nn.ReLU(),
            self.depthwise_separable_conv(10, 10, kernel_size=3, padding=1),  # output = 28x28, receptive field = 9x9
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=1, bias=False, dilation=2),  # output = 28x28, receptive field = 15x15
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.05),
            self.depthwise_separable_conv(10, 10, kernel_size=3, padding=1),  # output = 28x28, receptive field = 17x17
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=1, bias=False, dilation=2),  # output = 28x28, receptive field = 23x23
            nn.BatchNorm2d(10),
            nn.ReLU(),
            self.depthwise_separable_conv(10, 13, kernel_size=3, padding=1),  # output = 28x28, receptive field = 25x25
            nn.BatchNorm2d(13),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2, 2)  # output = 14x14, receptive field = 50x50 (pooling doubles the receptive field)

        self.convblock2 = nn.Sequential(
            self.depthwise_separable_conv(13, 10, kernel_size=1, padding=0),  # output = 14x14, receptive field = 50x50
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=1, bias=False, dilation=2),  # output = 14x14, receptive field = 54x54
            nn.BatchNorm2d(10),
            nn.ReLU(),
            self.depthwise_separable_conv(10, 10, kernel_size=3, padding=1),  # output = 14x14, receptive field = 56x56
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=1, bias=False, dilation=2),  # output = 14x14, receptive field = 62x62
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.05),
            self.depthwise_separable_conv(10, 13, kernel_size=3, padding=1),  # output = 14x14, receptive field = 64x64
            nn.BatchNorm2d(13),
            nn.ReLU(),
            nn.Conv2d(in_channels=13, out_channels=13, kernel_size=(3, 3), padding=1, bias=False, dilation=3),  # output = 14x14, receptive field = 70x70
            nn.BatchNorm2d(13),
            nn.ReLU(),
            self.depthwise_separable_conv(13, 13, kernel_size=3, padding=0),  # output = 14x14, receptive field = 72x72
            nn.BatchNorm2d(13),
            nn.ReLU(),
            self.depthwise_separable_conv(13, 10, kernel_size=1, padding=0)  # output = 14x14, receptive field = 72x72
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # output_size = 1x1
    def depthwise_separable_conv(self, in_channels, out_channels, kernel_size=3, padding=1):
        """Depthwise Separable Convolution (Depthwise + Pointwise)"""
        # Depthwise Convolution
        depthwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                   kernel_size=kernel_size, padding=padding, groups=in_channels,
                                   bias=False)

        # Pointwise Convolution (1x1)
        pointwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=1, padding=0, bias=False)

        return nn.Sequential(depthwise_conv, pointwise_conv)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.pool1(x)
        x = self.convblock2(x)
        x = self.avgpool(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)




    

#Model Summary
model = Net().to(device)
print("\n Model Summary \n")
print(summary(model, input_size=(1, 28, 28)))


print("\n Training and testing the model \n")

optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
scheduler = StepLR(optimizer, step_size=4, gamma=0.4)

model_fiteval = FitEvaluate(model, device,mnist_data.train_loader,mnist_data.test_loader)
model_fiteval.epoch_training(optimizer, scheduler = scheduler)


