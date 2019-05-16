"""
    Util functions for harmonic Wide Residual Network.

    The code is based on pytorch implementation of WRN:
    https://github.com/szagoruyko/wide-residual-networks/tree/master/pytorch

    2019 Matej Ulicny
"""

import torch
from torch.nn.init import kaiming_normal_
import torch.nn.functional as F
from torch.nn.parallel._functions import Broadcast
from torch.nn.parallel import scatter, parallel_apply, gather
from functools import partial
from nested_dict import nested_dict
import numpy as np
import math

def dct_filters(n=3, groups=1, expand_dim=1, level=None, DC=True, l1_norm=True):
    if level is None:
        filter_bank = np.zeros((n**2-int(not DC), n, n), dtype=np.float32)
    else:
        filter_bank = np.zeros((level*(level+1)//2-int(not DC), n, n), dtype=np.float32)
    m = 0
    for i in range(n):
        for k in range(n):
            if (not DC and i == 0 and k == 0) or (not level is None and i + k >= level):
                continue
            ai = 1.0 if i > 0 else 1.0 / math.sqrt(2.0)
            ak = 1.0 if k > 0 else 1.0 / math.sqrt(2.0)
            for x in range(n):
                for y in range(n):
                    filter_bank[m, x, y] = math.cos((math.pi * (x + .5) * i) / n) * math.cos((math.pi * (y + .5) * k) / n)
            if l1_norm:
                filter_bank[m, :, :] /= np.sum(np.abs(filter_bank[m, :, :]))
            else:
                filter_bank[m, :, :] *= (2.0 / n) * ai * ak
            m += 1
    filter_bank = np.tile(np.expand_dims(filter_bank, axis=expand_dim), (groups,1,1,1))
    return torch.FloatTensor(filter_bank)


def cast(params, dtype='float'):
    if isinstance(params, dict):
        return {k: cast(v, dtype) for k,v in params.items()}
    else:
        return getattr(params.cuda() if torch.cuda.is_available() else params, dtype)()


def conv_params(ni, no, k=1):
    return kaiming_normal_(torch.Tensor(no, ni, k, k))

def dct_params(ni, no, nf):
    return kaiming_normal_(torch.Tensor(no, ni, nf, 1, 1))

def linear_params(ni, no):
    return {'weight':kaiming_normal_(torch.Tensor(no, ni)), 'bias': torch.zeros(no)}


def bnparams(n, affine=True):
    paramdict = {'running_mean': torch.zeros(n),
                 'running_var': torch.ones(n)}
    if affine:
        paramdict.update({'weight': torch.rand(n),
                          'bias': torch.zeros(n)})
    return paramdict


def data_parallel(f, input, params, mode, device_ids, output_device=None):
    assert isinstance(device_ids, list)
    if output_device is None:
        output_device = device_ids[0]

    if len(device_ids) == 1:
        return f(input, params, mode)

    params_all = Broadcast.apply(device_ids, *params.values())
    params_replicas = [{k: params_all[i + j*len(params)] for i, k in enumerate(params.keys())}
                       for j in range(len(device_ids))]

    replicas = [partial(f, params=p, mode=mode)
                for p in params_replicas]
    inputs = scatter([input], device_ids)
    outputs = parallel_apply(replicas, inputs)
    return gather(outputs, output_device)


def flatten(params):
    return {'.'.join(k): v for k, v in nested_dict(params).items_flat() if v is not None}


def batch_norm(x, params, base, mode, affine=True):
    return F.batch_norm(x, weight=params[base + '.weight'] if affine else None,
                        bias=params[base + '.bias'] if affine else None,
                        running_mean=params[base + '.running_mean'],
                        running_var=params[base + '.running_var'],
                        training=mode)


def print_tensor_dict(params):
    kmax = max(len(key) for key in params.keys())
    for i, (key, v) in enumerate(params.items()):
        print(str(i).ljust(5), key.ljust(kmax + 3), str(tuple(v.shape)).ljust(23), torch.typename(v), v.requires_grad)


def set_requires_grad_except_bn_(params):
    for k, v in params.items():
        if not k.endswith('running_mean') and not k.endswith('running_var') and k.find('dct') == -1:
            v.requires_grad = True

