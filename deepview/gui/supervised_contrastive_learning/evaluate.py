# This Python file uses the following encoding: utf-8
from __future__ import print_function

import sys
import argparse
import time
import math

# import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet
from losses import SupConLoss

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from deepview.clustering_pytorch.datasets import (
    Batch,
    prepare_all_data,
)
from deepview.clustering_pytorch.nnet.common_config import (
    get_criterion,
    get_optimizer,
    get_model,
    # adjust_learning_rate,
)
from tqdm import tqdm

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # parser.add_argument('--print_freq', type=int, default=10,
    #                     help='print frequency')
    # parser.add_argument('--save_freq', type=int, default=50,
    #                     help='save frequency')
    # todo 这个变量需要制作文本输入框
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    # todo 这个变量需要制作文本输入框
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of training epochs')

    # optimization
    # todo 这个变量需要制作文本输入框
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # # model dataset
    # parser.add_argument('--model', type=str, default='resnet50')
    # parser.add_argument('--dataset', type=str, default='cifar10',
    #                     choices=['cifar10', 'cifar100', 'path'], help='dataset')
    # parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    # parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    # parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    # parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    # todo 这个变量需要制作下拉框
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    return opt


def set_loader(opt):
    # 相同代码在clustering pytorch/core的train.py出现
    data_path = r'C:\Users\dell\Desktop\ss-cc-2024-08-05\unsupervised-datasets\allDataSet'
    select_filenames = ['Omizunagidori2018_raw_data_9B36578_lb0009_25Hz.pkl']
    data_len = 180
    data_column = ['acc_x', 'acc_y', 'acc_z', 'GPS_velocity', 'GPS_bearing']
    batch_size = 1028
    augment = True

    train_dataloader, num_channel = prepare_all_data(data_path,
                                                     select_filenames,
                                                     data_len,
                                                     data_column,
                                                     batch_size,
                                                     augment,
                                                     device,
                                                     labeled_flag=True)
    return train_dataloader, num_channel  # trainloader修改label，从batch,len,1变成batch，去掉无标签数据


def set_model(opt):
    # 相同代码在clustering pytorch/core的train.py出现
    net_type = 'AE_CNN'
    p_setup = 'simclr'
    num_channel = 5  # 根据模型文件pkl的名字，找到使用多少传感器，计算总列数
    data_len = 180  # 根据模型文件pkl的名字

    model = get_model(p_backbone=net_type,
                      p_setup=p_setup,
                      num_channel=num_channel,
                      data_len=data_len)
    model = model.to(device)

    criterion = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    # if torch.cuda.is_available():
    if 'cuda' in device.type:
        if torch.cuda.device_count() > 1:
            # model.encoder = torch.nn.DataParallel(model.encoder)
            model.backbone = torch.nn.DataParallel(model.backbone)
        # model = model.cuda()
        # criterion = criterion.cuda()
        cudnn.benchmark = True
    model = model.to(device)
    criterion = criterion.to(device)
    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    # for idx, (images, labels) in enumerate(train_loader):
    for idx, (aug_sample1, aug_sample2, timestamp, labels) in enumerate(tqdm(train_loader)):
        aug_sample1 = aug_sample1.to(dtype=torch.double)
        aug_sample2 = aug_sample2.to(dtype=torch.double)
        data_time.update(time.time() - end)

        images = torch.cat([aug_sample1, aug_sample2], dim=0)
        # if torch.cuda.is_available():
        #     images = images.cuda(non_blocking=True)
        #     labels = labels[:,0,0].cuda(non_blocking=True)  # remove unlabeled data.
        images = images.to(device, non_blocking=True)
        labels = labels[:, int(labels.shape[1] / 2), 0].to(device, non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)  # batch, 2, 128
        if opt.method == 'SupCon':
            loss = criterion(features, labels)
        elif opt.method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


    return losses.avg


def main():
    # todo 参数希望放到\unsup-models\iteration-0\dAug4 文件夹下，生成。yaml文件
    opt = parse_option()

    # build data loader
    train_loader, _ = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # Load the saved state
    full_model_path_new = r'C:\Users\dell\Desktop\ss-cc-2024-08-05\unsup-models\iteration-0\ssAug5\AE_CNN_epoch29_datalen180_gps-acceleration_%s.pth' % opt.method

    checkpoint = torch.load(full_model_path_new)

    # Restore model state
    model.load_state_dict(checkpoint['model'])

    # Restore optimizer state
    optimizer.load_state_dict(checkpoint['optimizer'])

    loss = train(train_loader, model, criterion, optimizer, 0, opt)

if __name__ == '__main__':
    main()
