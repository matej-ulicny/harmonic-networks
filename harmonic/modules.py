"""
    Harmonic block definition.

    Licensed under the BSD License [see LICENSE for details].

    Written by Matej Ulicny
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def dct_filters(k=3, groups=1, expand_dim=1, level=None, DC=True, l1_norm=True):
    if level is None:
        nf = k**2 - int(not DC) 
    else:
       	if level <= k:
            nf = level*(level+1)//2 - int(not DC) 
       	else:
       	    r = 2*k-1 - level
       	    nf = k**2 - r*(r+1)//2 - int(not DC)
    filter_bank = np.zeros((nf, k, k), dtype=np.float32)
    m = 0
    for i in range(k):
        for j in range(k):
            if (not DC and i == 0 and j == 0) or (not level is None and i + j >= level):
                continue
            for x in range(k):
                for y in range(k):
                    filter_bank[m, x, y] = math.cos((math.pi * (x + .5) * i) / k) * math.cos((math.pi * (y + .5) * j) / k)
            if l1_norm:
                filter_bank[m, :, :] /= np.sum(np.abs(filter_bank[m, :, :]))
            else:
                ai = 1.0 if i > 0 else 1.0 / math.sqrt(2.0)
                aj = 1.0 if j > 0 else 1.0 / math.sqrt(2.0)
                filter_bank[m, :, :] *= (2.0 / k) * ai * aj
            m += 1
    filter_bank = np.tile(np.expand_dims(filter_bank, axis=expand_dim), (groups, 1, 1, 1))
    return torch.FloatTensor(filter_bank)


class Harm2d(nn.Module):

    def __init__(self, ni, no, kernel_size, stride=1, padding=0, bias=True, dilation=1, use_bn=False, level=None, DC=True, groups=1):
        super(Harm2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.dct = nn.Parameter(dct_filters(k=kernel_size, groups=ni if use_bn else 1, expand_dim=1 if use_bn else 0, level=level, DC=DC), requires_grad=False)
        
        nf = self.dct.shape[0] // ni if use_bn else self.dct.shape[1]
        if use_bn:
            self.bn = nn.BatchNorm2d(ni*nf, affine=False)
            self.weight = nn.Parameter(nn.init.kaiming_normal_(torch.Tensor(no, ni // self.groups * nf, 1, 1), mode='fan_out', nonlinearity='relu'))
        else:
            self.weight = nn.Parameter(nn.init.kaiming_normal_(torch.Tensor(no, ni // self.groups, nf, 1, 1), mode='fan_out', nonlinearity='relu'))
        self.bias = nn.Parameter(nn.init.zeros_(torch.Tensor(no))) if bias else None

    def forward(self, x):
        if not hasattr(self, 'bn'):
            filt = torch.sum(self.weight * self.dct, dim=2)
            x = F.conv2d(x, filt, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
            return x
        else:
            x = F.conv2d(x, self.dct, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=x.size(1))
            x = self.bn(x)
            x = F.conv2d(x, self.weight, bias=self.bias, padding=0, groups=self.groups)
            return x

