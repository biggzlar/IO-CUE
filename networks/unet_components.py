import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise separable convolution for reduced parameters and faster computation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class StridedConvBlock(nn.Module):
    """Replaces MaxPool2d + Conv2d with a single strided convolution"""
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                             stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class EfficientUpsample(nn.Module):
    """Efficient upsampling block using transposed convolution with reduced channels"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DepthwiseSeparableConv2d(out_channels + out_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.up(x)
        skip = nn.functional.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class OptimizedHead(nn.Module):
    """Optimized head using depthwise separable convolutions"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv2d(in_channels, in_channels // 2)
        self.conv2 = DepthwiseSeparableConv2d(in_channels // 2, in_channels // 4)
        self.conv3 = nn.Conv2d(in_channels // 4, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.bn2 = nn.BatchNorm2d(in_channels // 4)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x 