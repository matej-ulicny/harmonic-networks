"""
    Training code for harmonic Wide Residual Networks.

    Licensed under the BSD License [see LICENSE for details].

    Written by Matej Ulicny, based on implementation by Sergey Zagoruyko:
    https://github.com/szagoruyko/wide-residual-networks/tree/master/pytorch
"""

import argparse
import os
import json
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import SGD
import torch.utils.data
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import torchnet as tnt
from torchnet.engine import Engine
from utils import cast, data_parallel, print_tensor_dict
from torch.backends import cudnn
from models.harm_wrn_stl import resnet

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Wide Residual Networks')
# Model options
parser.add_argument('--model', default='resnet', type=str)
parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--width', default=1, type=float)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--level', default=None, type=int)
parser.add_argument('--dataset', default='STL10', type=str)
parser.add_argument('--dataroot', default='../data/stl/', type=str)
parser.add_argument('--fold', default=-1, type=int)
parser.add_argument('--dtype', default='float', type=str)
parser.add_argument('--groups', default=1, type=int)
parser.add_argument('--nthread', default=1, type=int)
parser.add_argument('--seed', default=1, type=int)

# Training options
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--weight_decay', default=0.0005, type=float)
parser.add_argument('--nesterov', action='store_true', default=False)
parser.add_argument('--epoch_step', default='[300,400,600,800]', type=str,
                    help='json list with epochs to drop lr on')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--note', default='', type=str)

# Device options
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--save', default='', type=str,
                    help='save parameters and logs in this folder')
parser.add_argument('--ngpu', default=1, type=int,
                    help='number of GPUs to use for training')
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')


def create_dataset(opt, train):


    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                    np.array([63.0, 62.1, 66.7]) / 255.0)
    ])
    if train:
        transform = T.Compose([
            T.Pad(12, padding_mode='reflect'),
            T.RandomHorizontalFlip(),
            T.RandomCrop(96),
            transform
        ])
    return datasets.STL10(opt.dataroot, split="train" if train else "test", download=True, transform=transform)


def main():
    opt = parser.parse_args()
    print('parsed options:', vars(opt))
    epoch_step = json.loads(opt.epoch_step)
    log_step = 5
    if opt.fold >= 0 and opt.fold <= 9:
        log_step *= 5
        epoch_step = [ep*5 for ep in epoch_step]
        opt.epochs *= 5
 
    num_classes = 10

    torch.manual_seed(opt.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    def create_iterator(mode):
        if opt.fold < 0 or opt.fold > 9:
            return DataLoader(create_dataset(opt, mode), opt.batch_size, shuffle=mode,
                              num_workers=opt.nthread, pin_memory=torch.cuda.is_available())
        if mode:
            folds = np.loadtxt('fold_indices.txt', dtype=np.int64)
            fold = folds[opt.fold]
            fold = torch.from_numpy(fold)
        return DataLoader(create_dataset(opt, mode), opt.batch_size, sampler=SubsetRandomSampler(fold) if mode else None,
                          num_workers=opt.nthread, pin_memory=torch.cuda.is_available())

    train_loader = create_iterator(True)
    test_loader = create_iterator(False)

    kwargs = {}
    if not opt.level is None:
        kwargs.update({'level': opt.level})
    f, params = resnet(opt.depth, opt.width, num_classes, opt.dropout, **kwargs)
    def create_optimizer(opt, lr):
        print('creating optimizer with lr = ', lr)
        return SGD([v for v in params.values() if v.requires_grad], lr, momentum=0.9, weight_decay=opt.weight_decay, nesterov=opt.nesterov)

    optimizer = create_optimizer(opt, opt.lr)

    epoch = 0
    if opt.resume != '':
        state_dict = torch.load(opt.resume)
        epoch = state_dict['epoch']
        params_tensors = state_dict['params']
        for k, v in params.items():
            if k in params_tensors:
                v.data.copy_(params_tensors[k])
        optimizer.load_state_dict(state_dict['optimizer'])

    print('\nParameters:')
    print_tensor_dict(params)

    n_parameters = sum(p.numel() for p in params.values() if p.requires_grad)
    print('\nTotal number of parameters:', n_parameters)

    meter_loss = tnt.meter.AverageValueMeter()
    classacc = tnt.meter.ClassErrorMeter(accuracy=True)
    timer_train = tnt.meter.TimeMeter('s')
    timer_test = tnt.meter.TimeMeter('s')

    if not os.path.exists(opt.save):
        os.mkdir(opt.save)

    def h(sample):
        inputs = cast(sample[0], opt.dtype)
        targets = cast(sample[1], 'long')
        y = data_parallel(f, inputs, params, sample[2], list(range(opt.ngpu))).float()
        return F.cross_entropy(y, targets), y

    def log(t, state):
        torch.save(dict(params={k: v for k, v in params.items() if k.find('dct') == -1}, epoch=t['epoch'], 
                   optimizer=state['optimizer'].state_dict()), os.path.join(opt.save, 'model.pt7'))
        z = vars(opt).copy()
        z.update(t)
        with open(os.path.join(opt.save, 'log.txt'), 'a') as flog:
            flog.write('json_stats: ' + json.dumps(z) + '\n')
        print(z)

    def on_sample(state):
        state['sample'].append(state['train'])

    def on_forward(state):
        loss = float(state['loss'])
        classacc.add(state['output'].data, state['sample'][1])
        meter_loss.add(loss)
        if state['train']:
            state['iterator'].set_postfix(loss=loss)

    def on_start(state):
        state['epoch'] = epoch

    def on_start_epoch(state):
        classacc.reset()
        meter_loss.reset()
        timer_train.reset()
        state['iterator'] = tqdm(train_loader, dynamic_ncols=True)
        epoch = state['epoch'] + 1
        if epoch in epoch_step:
            lr = state['optimizer'].param_groups[0]['lr']
            state['optimizer'] = create_optimizer(opt, lr * opt.lr_decay_ratio)

    def on_end_epoch(state):
        if state['epoch'] % log_step == 0:
            train_loss = meter_loss.value()
            train_acc = classacc.value()
            train_time = timer_train.value()
            meter_loss.reset()
            classacc.reset()
            timer_test.reset()

            with torch.no_grad():
                engine.test(h, test_loader)

            test_acc = classacc.value()[0]
            print(log({
                "train_loss": train_loss[0],
                "train_acc": train_acc[0],
                "test_loss": meter_loss.value()[0],
                "test_acc": test_acc,
                "epoch": state['epoch'],
                "num_classes": num_classes,
                "n_parameters": n_parameters,
                "train_time": train_time,
                "test_time": timer_test.value(),
            }, state))
            print('==> id: %s (%d/%d), test_acc: \33[91m%.2f\033[0m' %
                  (opt.save, state['epoch'], opt.epochs, test_acc))

    engine = Engine()
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.hooks['on_start'] = on_start
    engine.train(h, train_loader, opt.epochs, optimizer)


if __name__ == '__main__':
    main()
