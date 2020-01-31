"""
    Definition of harmonic VGG networks.

    Licensed under the BSD License [see LICENSE for details].

    Written by Matej Ulicny, based on torchvision implementation:
    https://github.com/pytorch/vision/tree/master/torchvision/models
"""

import torch.nn as nn
from harmonic import Harm2d


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if not m.weight is None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, harm_root=True, harm_res_blocks=True):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if i == 0:
                conv2d = Harm2d(in_channels, v, kernel_size=3, padding=1, use_bn=True) if harm_root else nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            else:
                conv2d = Harm2d(in_channels, v, kernel_size=3, padding=1) if harm_res_blocks else nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, **kwargs):
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm, harm_root=['harm_root'], harm_res_blocks=kwargs['harm_res_blocks']))
    return model

def vgg11(**kwargs):
    return _vgg('vgg11', 'A', False, **kwargs)


def vgg11_bn(**kwargs):
    return _vgg('vgg11_bn', 'A', True, **kwargs)


def vgg13(**kwargs):
    return _vgg('vgg13', 'B', False, **kwargs)


def vgg13_bn( **kwargs):
    return _vgg('vgg13_bn', 'B', True, **kwargs)


def vgg16(**kwargs):
    return _vgg('vgg16', 'D', False, **kwargs)


def vgg16_bn(**kwargs):
    return _vgg('vgg16_bn', 'D', True, **kwargs)


def vgg19(**kwargs):
    return _vgg('vgg19', 'E', False, **kwargs)


def vgg19_bn(**kwargs):
    return _vgg('vgg19_bn', 'E', True, **kwargs)

