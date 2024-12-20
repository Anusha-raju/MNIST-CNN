import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Model 3

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

# Model 2

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )  # output = 26x26, receptive field = 3x3 (since padding=0 and kernel_size=3x3)

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )  # output = 24x24, receptive field = 5x5 (each 3x3 kernel expands the receptive field by 2)

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )  # output = 22x22, receptive field = 7x7 (each 3x3 kernel expands the receptive field by 2)

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2)  # output = 11x11, receptive field = 14x14 (pooling doubles the receptive field)

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )  # output = 11x11, receptive field = 14x14 (1x1 kernel does not change the receptive field)

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )  # output = 9x9, receptive field = 16x16 (each 3x3 kernel expands the receptive field by 2)

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )  # output = 7x7, receptive field = 20x20 (each 3x3 kernel expands the receptive field by 2)

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        )  # output = 7x7, receptive field = 20x20 (1x1 kernel does not change the receptive field)

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=7)
        )  # output_size = 1x1, receptive field = 26x26 (the global average pool covers the entire input)

        self.dropout = nn.Dropout(0.01)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.dropout(x)
        x = self.convblock2(x)
        x = self.dropout(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.dropout(x)
        x = self.convblock4(x)
        x = self.dropout(x)
        x = self.convblock5(x)
        x = self.dropout(x)
        x = self.convblock6(x)
        x = self.dropout(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

# Model 1

class Net1(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        )  # output = 26x26, receptive field = 3x3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        )  # output = 24x24, receptive field = 5x5 (each 3x3 kernel expands by 2)

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        )  # output = 22x22, receptive field = 7x7 (each 3x3 kernel expands by 2)

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2)  # output = 11x11, receptive field = 14x14 (pooling doubles the receptive field)

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        )  # output = 11x11, receptive field = 14x14 (1x1 kernel doesn't change receptive field)

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        )  # output = 9x9, receptive field = 16x16 (each 3x3 kernel expands by 2)

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        )  # output = 7x7, receptive field = 20x20 (each 3x3 kernel expands by 2)

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        )  # output = 7x7, receptive field = 20x20 (1x1 kernel doesn't change receptive field)

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(7, 7), padding=0, bias=False),
        )  # output = 1x1x10, receptive field = 26x26 (7x7 kernel on 7x7 input results in 1x1 output covering the entire input)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
