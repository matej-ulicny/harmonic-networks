"""
    Definition of partially harmonic Wide Residual Network.

    Licensed under the BSD License [see LICENSE for details].

    Written by Matej Ulicny, based on implementation by Sergey Zagoruyko:
"""

import torch
import torch.nn.functional as F
import utils


def resnet(depth, width, num_classes, dropout):
    assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
    n = (depth - 4) // 6
    widths = [int(v * width) for v in (16, 32, 64)]

    def gen_harmonic_params(ni, no, k, normalize=False):
        paramdict = {'conv': utils.conv_params(ni*k*k, no, 1)}
        if normalize:
            paramdict.update({'bn': utils.bnparams(ni*k*k, affine=False)})
        return paramdict

    def gen_block_params(ni, no):
        return {
            'conv0': utils.conv_params(ni, no, 3),
            'conv1': utils.conv_params(no, no, 3),
            'bn0': utils.bnparams(ni),
            'bn1': utils.bnparams(no),
            'convdim': utils.conv_params(ni, no, 1) if ni != no else None,
        }

    def gen_group_params(ni, no, count):
        return {'block%d' % i: gen_block_params(ni if i == 0 else no, no)
                for i in range(count)}

    flat_params = utils.cast(utils.flatten({
        'dct': utils.dct_filters(n=3, groups=3),
        'harmonic0': gen_harmonic_params(3, 16, k=3, normalize=True),
        'group0': gen_group_params(16, widths[0], n),
        'group1': gen_group_params(widths[0], widths[1], n),
        'group2': gen_group_params(widths[1], widths[2], n),
        'bn': utils.bnparams(widths[2]),
        'fc': utils.linear_params(widths[2], num_classes),
    }))

    utils.set_requires_grad_except_bn_(flat_params)

    def harmonic_block(x, params, base, mode, stride=1, padding=1):
        y = F.conv2d(x, params['dct'], stride=stride, padding=padding, groups=x.size(1))
        if base + '.bn.running_mean' in params:
            y = utils.batch_norm(y, params, base + '.bn', mode, affine=False)
        z = F.conv2d(y, params[base + '.conv'], padding=0)
        return z

    def block(x, params, base, mode, stride):
        o1 = F.relu(utils.batch_norm(x, params, base + '.bn0', mode), inplace=True)
        y = F.conv2d(o1, params[base + '.conv0'], stride=stride, padding=1)
        o2 = F.relu(utils.batch_norm(y, params, base + '.bn1', mode), inplace=True)
        if dropout > 0:
            o2 = F.dropout(o2, p=dropout, training=mode, inplace=False)
        z = F.conv2d(o2, params[base + '.conv1'], stride=1, padding=1)
        if base + '.convdim' in params:
            return z + F.conv2d(o1, params[base + '.convdim'], stride=stride)
        else:
            return z + x

    def group(o, params, base, mode, stride):
        for i in range(n):
            o = block(o, params, '%s.block%d' % (base,i), mode, stride if i == 0 else 1)
        return o

    def f(input, params, mode):
        x = harmonic_block(input, params, 'harmonic0', mode, stride=1, padding=1)
        g0 = group(x, params, 'group0', mode, 1)
        g1 = group(g0, params, 'group1', mode, 2)
        g2 = group(g1, params, 'group2', mode, 2)
        o = F.relu(utils.batch_norm(g2, params, 'bn', mode))
        o = F.avg_pool2d(o, 8, 1, 0)
        o = o.view(o.size(0), -1)
        o = F.linear(o, params['fc.weight'], params['fc.bias'])
        return o

    return f, flat_params
