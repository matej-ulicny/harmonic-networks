#!/user/bin/python
# coding=utf-8
import os, sys
import numpy as np
from PIL import Image
import shutil
import argparse
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_loader import BSDSLoader
from functions import sigmoid_cross_entropy_loss, cross_entropy_loss, fixed_weight_cross_entropy_loss
from torch.utils.data import DataLoader, sampler
from utils import Logger, Averagvalue, save_checkpoint, load_vgg16pretrain, load_harm_vgg16pretrain
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname
from models import HED, HEDSmall

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=1, type=int, metavar='BT',
                    help='batch size')
# =============== optimizer
parser.add_argument('--lr', '--learning_rate', default=1e-6, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float,
                    metavar='W', help='default weight decay')
parser.add_argument('--stepsize', default=4, type=int, 
                    metavar='SS', help='learning rate step size')
parser.add_argument('--gamma', '--gm', default=0.1, type=float,
                    help='learning rate decay parameter: Gamma')
parser.add_argument('--maxepoch', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--itersize', default=10, type=int,
                    metavar='IS', help='iter size')
# =============== misc
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print_freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--gpu', default='0', type=str,
                    help='GPU ID')
parser.add_argument('--harmonic', action='store_true',
                    help='use harmonic blocks')
parser.add_argument('--small', action='store_true',
                    help='build small network')
parser.add_argument('--pretrained', action='store_true',
                    help='initialize model with pretrained weights')
parser.add_argument('--use-cfg', action='store_true',
                    help='use predefined training hyperparameters')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--output', help='output folder', default='output/HED')
# ================ dataset
parser.add_argument('--dataset', help='root folder of dataset', default='data/HED-BSDS')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

THIS_DIR = abspath(dirname(__file__))
OUT_DIR = join(THIS_DIR, args.output)
if not isdir(OUT_DIR):
  os.makedirs(OUT_DIR)

def main():
    args.cuda = True
    # dataset
    train_dataset = BSDSLoader(root=args.dataset, split="train")
    test_dataset = BSDSLoader(root=args.dataset, split="test")
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=4, drop_last=True,shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        num_workers=4, drop_last=True,shuffle=False)
    with open(join(args.dataset, 'test.lst'), 'r') as f:
        test_list = f.readlines()
    test_list = [split(i.rstrip())[1] for i in test_list]
    assert len(test_list) == len(test_loader), "%d vs %d" % (len(test_list), len(test_loader))

    # default hyperparameters
    if args.use_cfg:
        if args.pretrained and not args.small:
            args.stepsize = 2
            args.lr = 0.001 if args.harmonic else 0.0002
        elif args.small:
            args.stepsize = 6
            args.lr = 0.005 if args.harmonic else 0.001
        else:
            args.stepsize = 4
            args.lr = 0.0005 if args.harmonic else 0.0002
        args.maxepoch = args.stepsize + 1
        
    # model
    model = HEDSmall(harmonic=args.harmonic) if args.small else HED(harmonic=args.harmonic)
    model.cuda()
    model.apply(weights_init)
    if args.pretrained and not args.small:
        if args.harmonic:    
            load_harm_vgg16pretrain(model)
        else:
            load_vgg16pretrain(model)
    
    #tune lr
    net_parameters_id = {}
    
    if args.pretrained and not args.small:
        for pname, p in model.named_parameters():
            if pname in ['conv1_1.weight','conv1_2.weight',
                         'conv2_1.weight','conv2_2.weight',
                         'conv3_1.weight','conv3_2.weight','conv3_3.weight',
                         'conv4_1.weight','conv4_2.weight','conv4_3.weight',
                         'conv5_1.weight','conv5_2.weight','conv5_3.weight']:
                print(pname, 'lr:1 de:1')
                if 'conv1-5.weight' not in net_parameters_id:
                    net_parameters_id['conv1-5.weight'] = []
                net_parameters_id['conv1-5.weight'].append(p)
            elif pname in ['conv1_1.bias','conv1_2.bias',
                           'conv2_1.bias','conv2_2.bias',
                           'conv3_1.bias','conv3_2.bias','conv3_3.bias',
                           'conv4_1.bias','conv4_2.bias','conv4_3.bias',
                           'conv5_1.bias','conv5_2.bias','conv5_3.bias']:
                print(pname, 'lr:2 de:0')
                if 'conv1-5.bias' not in net_parameters_id:
                    net_parameters_id['conv1-5.bias'] = []
                net_parameters_id['conv1-5.bias'].append(p)     
            elif pname in ['score_dsn1.weight','score_dsn2.weight','score_dsn3.weight',
                           'score_dsn4.weight','score_dsn5.weight']:
                print(pname, 'lr:0.01 de:1')
                if 'score_dsn_1-5.weight' not in net_parameters_id:
                    net_parameters_id['score_dsn_1-5.weight'] = []
                net_parameters_id['score_dsn_1-5.weight'].append(p)
            elif pname in ['score_dsn1.bias','score_dsn2.bias','score_dsn3.bias',
                           'score_dsn4.bias','score_dsn5.bias']:
                print(pname, 'lr:0.02 de:0')
                if 'score_dsn_1-5.bias' not in net_parameters_id:
                    net_parameters_id['score_dsn_1-5.bias'] = []
                net_parameters_id['score_dsn_1-5.bias'].append(p)
            elif pname in ['score_final.weight']:
                print(pname, 'lr:0.001 de:1')
                if 'score_final.weight' not in net_parameters_id:
                    net_parameters_id['score_final.weight'] = []
                net_parameters_id['score_final.weight'].append(p)
            elif pname in ['score_final.bias']:
                print(pname, 'lr:0.002 de:0')
                if 'score_final.bias' not in net_parameters_id:
                    net_parameters_id['score_final.bias'] = []
                net_parameters_id['score_final.bias'].append(p)
        param_groups = [
                {'params': net_parameters_id['conv1-5.weight']      , 'lr': args.lr*1    , 'weight_decay': args.weight_decay},
                {'params': net_parameters_id['conv1-5.bias']        , 'lr': args.lr*2    , 'weight_decay': 0.},
                {'params': net_parameters_id['score_dsn_1-5.weight'], 'lr': args.lr*0.01 , 'weight_decay': args.weight_decay},
                {'params': net_parameters_id['score_dsn_1-5.bias']  , 'lr': args.lr*0.02 , 'weight_decay': 0.},
                {'params': net_parameters_id['score_final.weight']  , 'lr': args.lr*0.001, 'weight_decay': args.weight_decay},
                {'params': net_parameters_id['score_final.bias']    , 'lr': args.lr*0.002, 'weight_decay': 0.}
            ]
    else:
        net_parameters_id = {'weights': [], 'biases': []}
        for pname, p in model.named_parameters():
            if 'weight' in pname:
                net_parameters_id['weights'].append(p)
            elif 'bias' in pname:
                net_parameters_id['biases'].append(p)
        param_groups = [
                {'params': net_parameters_id['weights'], 'weight_decay': args.weight_decay},
                {'params': net_parameters_id['biases'], 'weight_decay': 0.}
            ]

    optimizer = torch.optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
    

    if args.resume:
        if isfile(args.resume): 
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'"
                  .format(args.resume))
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch']
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # log
    log = Logger(join(OUT_DIR, 'log.txt'))
    sys.stdout = log

    train_loss = []
    train_loss_detail = []
    for epoch in range(args.start_epoch, args.maxepoch):
        if epoch == 0:
            print("Performing initial testing...")
            test(model, test_loader, epoch=epoch, test_list=test_list,
                 save_dir = join(OUT_DIR, 'initial-testing-record'))

        tr_avg_loss, tr_detail_loss = train(
            train_loader, model, optimizer, epoch,
            save_dir = join(OUT_DIR, 'epoch-%d-training-record' % epoch))
        test(model, test_loader, epoch=epoch, test_list=test_list,
            save_dir = join(OUT_DIR, 'epoch-%d-testing-record' % epoch))
        log.flush() # write log
        # Save checkpoint
        save_file = os.path.join(OUT_DIR, 'checkpoint_epoch{}.pth'.format(epoch))
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
                         }, filename=save_file)
        scheduler.step() # will adjust learning rate
        # save train/val loss/accuracy, save every epoch in case of early stop
        train_loss.append(tr_avg_loss)
        train_loss_detail += tr_detail_loss


def train(train_loader, model, optimizer, epoch, save_dir):
    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()
    # switch to train mode
    model.train()
    end = time.time()
    epoch_loss = []
    counter = 0
    for i, (image, label) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        image, label = image.cuda(), label.cuda()
        outputs = model(image)
        loss = torch.zeros(1).cuda()
        for o in outputs:
            loss = loss + cross_entropy_loss(o, label)       
        counter += 1
        loss = loss / args.itersize
        loss.backward()
        if counter == args.itersize:
            optimizer.step()
            optimizer.zero_grad()
            counter = 0
        # measure accuracy and record loss
        losses.update(loss.item(), image.size(0))
        epoch_loss.append(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
        # display and logging
        if not isdir(save_dir):
            os.makedirs(save_dir)
        if i % args.print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch+1, args.maxepoch, i, len(train_loader)) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(
                       loss=losses)
            print(info)
            if i % (10*args.print_freq) == 0:
                outputs.append(label)
                _, _, H, W = outputs[0].shape
                all_results = torch.zeros((len(outputs), 1, H, W))
                for j in range(len(outputs)):
                    all_results[j, 0, :, :] = outputs[j][0, 0, :, :]
                torchvision.utils.save_image(all_results, join(save_dir, "iter-%d.jpg" % i))

    return losses.avg, epoch_loss

def test(model, test_loader, epoch, test_list, save_dir):
    model.eval()
    if not isdir(save_dir):
        os.makedirs(save_dir)
    for idx, image in enumerate(test_loader):
        image = image.cuda()
        # rescale image to [0, 255] and then substract the mean
        _, _, H, W = image.shape
        results = model(image)
        result = torch.squeeze(results[-1].detach()).cpu().numpy()
        results_all = torch.zeros((len(results), 1, H, W))
        for i in range(len(results)):
            results_all[i, 0, :, :] = results[i]
        filename = splitext(test_list[idx])[0]
        torchvision.utils.save_image(results_all, join(save_dir, "%s.jpg" % filename))
        result_b = Image.fromarray(((1 - result) * 255).astype(np.uint8))
        result = Image.fromarray((result * 255).astype(np.uint8))
        result.save(join(save_dir, "%s.png" % filename))
        result_b.save(join(save_dir, "%s.jpg" % filename))
        print("Running test [%d/%d]" % (idx + 1, len(test_loader)))

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.weight.data.shape[0] == 1 and m.weight.data.shape[1] > 5: # dsn weights
            m.weight.data.zero_()
        if m.weight.data.shape == torch.Size([1, 5, 1, 1]): # fusion weights
            torch.nn.init.constant_(m.weight, 0.2)
        if m.bias is not None:
            m.bias.data.zero_()

if __name__ == '__main__':
    main()
