"""
    Definition of harmonic Residual Networks.

    Licensed under the BSD License [see LICENSE for details].

    Written by Matej Ulicny, based on torchvision implementation:
    https://github.com/pytorch/vision/tree/master/torchvision/models
"""

import torch.nn as nn
import sys
sys.path.insert(0, '../')
from harmonic import Harm2d


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


def harm3x3(in_planes, out_planes, stride=1, level=None):
    """3x3 harmonic convolution with padding"""
    return Harm2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                               bias=False, use_bn=False, level=level)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, harm=True, level=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if harm:
            self.harm1 = harm3x3(planes, planes, stride, level=level)
            self.harm2 = harm3x3(planes, planes, level=level)
        else:
            self.conv1 = conv3x3(planes, planes, stride)
            self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.harm1(x) if hasattr(self, 'harm1') else self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.harm2(out) if hasattr(self, 'harm2') else self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, harm=True, level=None):
        super(Bottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        if harm:
            self.harm2 = harm3x3(planes, planes, stride, level=level)
        else:
            self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.harm2(out) if hasattr(self, 'harm2') else self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, harm_root=True, harm_res_blocks=True, pool=None, levels=[None, None, None, None]):
        super(ResNet, self).__init__()
        self.inplanes = 64
        
        root_stride = 2 if pool in ['avg', 'max'] else 4
        if harm_root:
            self.harm1 = Harm2d(3, 64, kernel_size=7, stride=root_stride, padding=3,
                               bias=False, use_bn=True)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=root_stride, padding=3, bias=False)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        if pool == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        elif pool == 'max':
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], harm=harm_res_blocks, level=levels[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, harm=harm_res_blocks, level=levels[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, harm=harm_res_blocks, level=levels[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, harm=harm_res_blocks, level=levels[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) and m.affine:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, harm=True, level=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, harm, level))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, harm=harm, level=level))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.harm1(x) if hasattr(self, 'harm1') else self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if hasattr(self, 'pool'):
            x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

