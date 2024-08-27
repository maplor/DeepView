
from __future__ import print_function

import sys
import argparse
import time
import math

# import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

# from util import TwoCropTransform, AverageMeter
# from util import adjust_learning_rate, warmup_learning_rate
# from util import set_optimizer, save_model
from deepview.gui.supervised_contrastive_learning.util import TwoCropTransform, AverageMeter
from deepview.gui.supervised_contrastive_learning.util import adjust_learning_rate, warmup_learning_rate
from deepview.gui.supervised_contrastive_learning.util import set_optimizer, save_model
# from networks.resnet_big import SupConResNet
from deepview.gui.supervised_contrastive_learning.losses import SupConLoss

import os
from sklearn.manifold import TSNE
import pyqtgraph as pg
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')  # or 'Agg', 'Qt5Agg', etc.
# import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from deepview.clustering_pytorch.datasets import (
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
p_setup = 'simclr'
if p_setup == 'simclr':
    AUGMENT = True  # 也存到yaml文件或者opt里
else:
    AUGMENT = False  # 是否用data augment，作为参数存储


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # todo 这个变量需要制作文本输入框
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    # todo 这个变量需要制作文本输入框
    parser.add_argument('--epochs', type=int, default=1,
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
    # parser.add_argument('--syncBN', action='store_true',
    #                     help='using synchronized batch normalization')
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


def set_loader(augment=False, labeled_flag=False):
    '''
    从select data中读取pkl文件路径（我加了一列label_flag，可以用在勾选框上）
    '''
    select_filenames = ['Omizunagidori2018_raw_data_9B36578_lb0009_25Hz.pkl']
    # 相同代码在clustering pytorch/core的train.py出现
    data_path = r'C:\Users\user\Desktop\fast-test-2024-08-04\unsupervised-datasets\allDataSet'
    data_len = 180
    data_column = ['acc_x', 'acc_y', 'acc_z', 'GPS_velocity', 'GPS_bearing']
    batch_size = 1028

    '''
    deepview/clustering_pytorch/datasets/factory.py
    class data_loader_umineko(Dataset), if self.augment 处，
    选择两个def gen_aug(sample, ssh_type)里有的augment方法，可以重复
    '''
    train_dataloader, num_channel = prepare_all_data(data_path,
                                                     select_filenames,
                                                     data_len,
                                                     data_column,
                                                     batch_size,
                                                     augment,
                                                     device,
                                                     labeled_flag=labeled_flag)
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

    # load existing model
    ## 相似代码在__init__.py 文件 1360行左右，def featureExtraction
    # full_model_path = os.path.join(self.cfg["project_path"], unsup_model_path, self.model_path)
    full_model_path = r'C:\Users\user\Desktop\fast-test-2024-08-04\unsup-models\iteration-0\fastAug4\AE_CNN_epoch0_datalen180_gps-acceleration.pth'

    # if torch.cuda.is_available():
    if 'cuda' in device.type:
        state_dict = torch.load(full_model_path)
        filtered_state_dict = {k: v for k, v in state_dict.items() if 'backbone' in k}

        # Extract and load only the feature extractor part
        new_state_dict = model.state_dict()
        for name, param in filtered_state_dict.items():
            if 'backbone' in name:
                new_state_dict[name] = param
        model.load_state_dict(new_state_dict)

    else:
        state_dict = torch.load(full_model_path, map_location=torch.device('cpu'))
        filtered_state_dict = {k: v for k, v in state_dict.items() if 'backbone' in k}

        # Extract and load only the feature extractor part
        new_state_dict = model.state_dict()
        for name, param in filtered_state_dict.items():
            if 'backbone' in name:
                new_state_dict[name] = param
        model.load_state_dict(new_state_dict)

    criterion = SupConLoss(temperature=opt.temp)

    # # enable synchronized Batch Normalization
    # if opt.syncBN:
    #     model = apex.parallel.convert_syncbn_model(model)

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
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()

    for idx, (aug_sample1, aug_sample2, timestamp, labels, label_flags) in enumerate(tqdm(train_loader)):
        aug_sample1 = aug_sample1.to(dtype=torch.double)
        aug_sample2 = aug_sample2.to(dtype=torch.double)

        images = torch.cat([aug_sample1, aug_sample2], dim=0)
        images = images.to(device, non_blocking=True)
        labels = labels[:, int(labels.shape[1] / 2), 0].to(device, non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features, backboneout = model(images)

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
        # batch_time.update(time.time() - end)
        # end = time.time()


    return losses.avg


# def evaluate(model, epoch, train_loader, fig_name):
#     model.eval()

#     # get all data:
#     representation_list, flag_list, label_list = [], [], []

#     for idx, (aug_sample1, aug_sample2, timestamp, labels, label_flags) in enumerate(tqdm(train_loader)):
#         aug_sample1 = aug_sample1.to(dtype=torch.double)
#         aug_sample2 = aug_sample2.to(dtype=torch.double)
#         images = torch.cat([aug_sample1, aug_sample2], dim=0)
#         images = images.to(device, non_blocking=True)
#         features, backboneout = model(images)
#         representation_list.append(backboneout[0].detach().cpu().numpy().reshape(aug_sample1.shape[0], -1))
#         flag_list.append(label_flags[:, 0, 0].detach().cpu().numpy())
#         label_list.append(labels[:, 0, 0].detach().cpu().numpy())

#     # PCA latent representation to shape=(2, len) PCA降维到形状为 (2, len)
#     repre_concat = np.concatenate(representation_list)
#     repre_reshape = repre_concat.reshape(repre_concat.shape[0], -1)
#     repre_tsne = reduce_dimension_with_tsne(repre_reshape)
#     flag_concat = np.concatenate(flag_list)
#     label_concat = np.concatenate(label_list)

#     # Create a scatter plot
#     plt.figure(figsize=(8, 6))

#     real_label_idxs = np.where(flag_concat == 0)[0]
#     x = repre_tsne[real_label_idxs, 0]
#     y = repre_tsne[real_label_idxs, 1]
#     plt.scatter(x, y, color='grey', alpha=0.5, label='unlabeled')

#     real_label_idxs = np.where(flag_concat == 1)[0]
#     x = repre_tsne[real_label_idxs, 0]
#     y = repre_tsne[real_label_idxs, 1]
#     # plt.scatter(x, y, color='blue', alpha=0.5, label='labeled')
#     real_label_concat = label_concat[real_label_idxs]
#     color_dict = {0: 'blue', 1: 'red', 2: 'green', 3: 'yellow'}
#     for i in range(len(real_label_concat)):
#         plt.scatter(x[int(i)], y[int(i)],
#                     color=color_dict[real_label_concat[int(i)]],
#                     alpha=0.5)

#     # Add labels and title
#     plt.xlabel('X-axis Label')
#     plt.ylabel('Y-axis Label')
#     plt.title('supervised contrastive latent, %s' % str(epoch))
#     plt.legend()
#     plt.savefig('%s_%s.png' % (fig_name, str(epoch)))

def evaluate(model, epoch, train_loader, fig_name):
    model.eval()

    # get all data:
    representation_list, flag_list, label_list = [], [], []

    for idx, (aug_sample1, aug_sample2, timestamp, labels, label_flags) in enumerate(tqdm(train_loader)):
        aug_sample1 = aug_sample1.to(dtype=torch.double)
        aug_sample2 = aug_sample2.to(dtype=torch.double)
        images = torch.cat([aug_sample1, aug_sample2], dim=0)
        images = images.to(device, non_blocking=True)
        features, backboneout = model(images)
        representation_list.append(backboneout[0].detach().cpu().numpy().reshape(aug_sample1.shape[0], -1))
        flag_list.append(label_flags[:, 0, 0].detach().cpu().numpy())
        label_list.append(labels[:, 0, 0].detach().cpu().numpy())

    # PCA latent representation to shape=(2, len) PCA降维到形状为 (2, len)
    repre_concat = np.concatenate(representation_list)
    repre_reshape = repre_concat.reshape(repre_concat.shape[0], -1)
    repre_tsne = reduce_dimension_with_tsne(repre_reshape)
    flag_concat = np.concatenate(flag_list)
    label_concat = np.concatenate(label_list)

    return repre_tsne, flag_concat, label_concat

    # # Create a scatter plot
    # plt.figure(figsize=(8, 6))

    # real_label_idxs = np.where(flag_concat == 0)[0]
    # x = repre_tsne[real_label_idxs, 0]
    # y = repre_tsne[real_label_idxs, 1]
    # plt.scatter(x, y, color='grey', alpha=0.5, label='unlabeled')

    # real_label_idxs = np.where(flag_concat == 1)[0]
    # x = repre_tsne[real_label_idxs, 0]
    # y = repre_tsne[real_label_idxs, 1]
    # # plt.scatter(x, y, color='blue', alpha=0.5, label='labeled')
    # real_label_concat = label_concat[real_label_idxs]
    # color_dict = {0: 'blue', 1: 'red', 2: 'green', 3: 'yellow'}
    # for i in range(len(real_label_concat)):
    #     plt.scatter(x[int(i)], y[int(i)],
    #                 color=color_dict[real_label_concat[int(i)]],
    #                 alpha=0.5)

    # # Add labels and title
    # plt.xlabel('X-axis Label')
    # plt.ylabel('Y-axis Label')
    # plt.title('supervised contrastive latent, %s' % str(epoch))
    # plt.legend()
    # plt.savefig('%s_%s.png' % (fig_name, str(epoch)))





#  __init__.py 相同代码
def reduce_dimension_with_tsne(array, method='tsne'):
    # tsne or pca
    tsne = TSNE(n_components=2)  # 创建TSNE对象，降维到2维
    reduced_array = tsne.fit_transform(array)  # 对数组进行降维
    return reduced_array


def main():
    # todo 参数希望放到\unsup-models\iteration-0\dAug4 文件夹下，生成。yaml文件
    # opt = parse_option()
    opt_dict = parse_options()
    opt = argparse.Namespace(**opt_dict)

    # build data loader
    train_loader, _ = set_loader(augment=AUGMENT, labeled_flag=True)

    # build model and criterion
    model, criterion = set_model(opt)
    # build optimizer
    optimizer = set_optimizer(opt, model)

    # training routine
    '''
    opt.epochs参数从Parameter选框中读取
    当点击 Apply SCL按钮后运行下面程序
    '''
    epoch = 0
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
    # evaluate and plot
    train_loader, _ = set_loader(augment=AUGMENT, labeled_flag=False)
    '''
    第一张New scatter map生成方式
    '''
    evaluate(model, epoch, train_loader, fig_name='supContrast')

    # standard contrastive learning
    opt.method = 'SimCLR'
    for epoch in range(1, opt.epochs + 1):
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
    # evaluate and plot
    '''
    第2张New scatter map生成方式
    '''
    evaluate(model, epoch, train_loader, fig_name='Contrast')

    # save the last model
    ## 将新模型保存在旧的模型所在目录，后面加上opt.method标志
    full_model_path_new = r'C:\Users\user\Desktop\fast-test-2024-08-04\unsup-models\iteration-0\fastAug5\AE_CNN_epoch29_datalen180_gps-acceleration_%s.pth' % opt.method
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, full_model_path_new)


def parse_options():
    # 默认参数
    default_params = {
        'batch_size': 256,
        'num_workers': 16,
        'epochs': 1,
        'learning_rate': 0.05,
        'lr_decay_epochs': '700,800,900',
        'lr_decay_rate': 0.1,
        'weight_decay': 1e-4,
        'momentum': 0.9,
        'method': 'SupCon',
        'temp': 0.07,
        'cosine': False,
        'warm': False,
        'trial': '0'
    }

    # 处理 lr_decay_epochs
    iterations = default_params['lr_decay_epochs'].split(',')
    default_params['lr_decay_epochs'] = [int(it) for it in iterations]

    # warm-up for large-batch training
    if default_params['batch_size'] > 256:
        default_params['warm'] = True
    
    if default_params['warm']:
        default_params['model_name'] = 'model_warm'
        default_params['warmup_from'] = 0.01
        default_params['warm_epochs'] = 10
        if default_params['cosine']:
            eta_min = default_params['learning_rate'] * (default_params['lr_decay_rate'] ** 3)
            default_params['warmup_to'] = eta_min + (default_params['learning_rate'] - eta_min) * (
                1 + math.cos(math.pi * default_params['warm_epochs'] / default_params['epochs'])) / 2
        else:
            default_params['warmup_to'] = default_params['learning_rate']

    return default_params




def handle_new_scatter_map_data():
    pass




def test():
    from PySide6 import QtWidgets
    # todo 参数希望放到\unsup-models\iteration-0\dAug4 文件夹下，生成。yaml文件
    # opt = parse_option()
    app = QtWidgets.QApplication([])

    opt_dict = parse_options()


    opt = argparse.Namespace(**opt_dict)

    map_1 = pg.GraphicsLayoutWidget(show=True, title="Autoencoder Latent Representation")
    map_2 = pg.GraphicsLayoutWidget(show=True, title="Autoencoder Latent Representation")



    # build data loader
    train_loader, _ = set_loader(augment=AUGMENT, labeled_flag=True)

    # build model and criterion
    model, criterion = set_model(opt)
    # build optimizer
    optimizer = set_optimizer(opt, model)

    # training routine
    '''
    opt.epochs参数从Parameter选框中读取
    当点击 Apply SCL按钮后运行下面程序
    '''
    epoch = 0
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
    # evaluate and plot
    train_loader, _ = set_loader(augment=AUGMENT, labeled_flag=False)
    '''
    第一张New scatter map生成方式
    '''
    plot_1 = map_1.addPlot(title=f"supContrast,{epoch}")

    repre_tsne, flag_concat, label_concat = evaluate(model, epoch, train_loader, fig_name='supContrast')

    real_label_idxs_unlabeled_1 = np.where(flag_concat == 0)[0]
    x_unlabeled_1 = repre_tsne[real_label_idxs_unlabeled_1, 0]
    y_unlabeled_1 = repre_tsne[real_label_idxs_unlabeled_1, 1]
    scatter1 = pg.ScatterPlotItem(x=x_unlabeled_1, y=y_unlabeled_1, pen=pg.mkPen(None),
                                  brush=pg.mkBrush(128, 128, 128, 128), size=5,
                                  data=real_label_idxs_unlabeled_1)

    plot_1.addItem(scatter1)

    real_label_idxs_labeled_1 = np.where(flag_concat == 1)[0]
    x_labeled_1 = repre_tsne[real_label_idxs_labeled_1, 0]
    y_labeled_1 = repre_tsne[real_label_idxs_labeled_1, 1]
    # plt.scatter(x, y, color='blue', alpha=0.5, label='labeled')
    real_label_concat = label_concat[real_label_idxs_labeled_1]
    color_dict = {0: 'blue', 1: 'red', 2: 'green', 3: 'yellow'}
    brushes = [pg.mkBrush(color_dict[label]) for label in real_label_concat]
    scatter2 = pg.ScatterPlotItem(x=x_labeled_1, y=y_labeled_1, pen=pg.mkPen(None),
                                  brush=brushes, size=5, data=real_label_idxs_labeled_1)
    plot_1.addItem(scatter2)

    # standard contrastive learning
    opt.method = 'SimCLR'
    for epoch in range(1, opt.epochs + 1):
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
    # evaluate and plot
    '''
    第2张New scatter map生成方式
    '''
    plot_2 = map_2.addPlot(title=f"Contrast,{epoch}")

    repre_tsne, flag_concat, label_concat = evaluate(model, epoch, train_loader, fig_name='Contrast')
    real_label_idxs_unlabeled_2 = np.where(flag_concat == 0)[0]
    x_unlabeled_2 = repre_tsne[real_label_idxs_unlabeled_2, 0]
    y_unlabeled_2 = repre_tsne[real_label_idxs_unlabeled_2, 1]
    scatter3 = pg.ScatterPlotItem(x=x_unlabeled_2, y=y_unlabeled_2, pen=pg.mkPen(None),
                                  brush=pg.mkBrush(128, 128, 128, 128), size=5,
                                  data=real_label_idxs_unlabeled_2)

    plot_2.addItem(scatter3)

    real_label_idxs_labeled_2 = np.where(flag_concat == 1)[0]
    x_labeled_2 = repre_tsne[real_label_idxs_labeled_2, 0]
    y_labeled_2 = repre_tsne[real_label_idxs_labeled_2, 1]
 
    real_label_concat = label_concat[real_label_idxs_labeled_2]
    color_dict = {0: 'blue', 1: 'red', 2: 'green', 3: 'yellow'}
    brushes2 = [pg.mkBrush(color_dict[label]) for label in real_label_concat]
    scatter4 = pg.ScatterPlotItem(x=x_labeled_2, y=y_labeled_2, pen=pg.mkPen(None),
                                  brush=brushes2, size=5, data=real_label_idxs_labeled_2)
    plot_2.addItem(scatter4)

    # # Connect signals
    # scatter1.sigClicked.connect(lambda _, points: on_click(points))
    # scatter2.sigClicked.connect(lambda _, points: on_click(points))
    # scatter3.sigClicked.connect(lambda _, points: on_click(points))
    # scatter4.sigClicked.connect(lambda _, points: on_click(points))



    # def change_points_properties(self, indices, new_size, new_color):
    #     global last_modified_points
    #     # Restore original properties of last modified points
    #     for p, original_size, original_brush in last_modified_points:
    #         p.setSize(original_size)
    #         p.setBrush(original_brush)

    #     last_modified_points = []  # Clear the list

    #     # Change properties of new points
    #     for scatter in [scatter1, scatter2, scatter3, scatter4]:
    #         points = scatter.points()
    #         for p in points:
    #             if p.data() in indices:
    #                 # Save current properties
    #                 original_size = p.size()
    #                 original_brush = p.brush()
    #                 last_modified_points.append((p, original_size, original_brush))

    #                 # Change to new properties
    #                 p.setSize(new_size)
    #                 p.setBrush(pg.mkBrush(new_color))

    # def on_click(self, points):
    #     idxs = []  # Initialize idxs list
    #     for p in points:
    #         idx = p.data()
    #         idxs.append(idx)  # Add idx to idxs list
    #     change_points_properties(idxs, new_size=14, new_color=(255, 0, 0, 255))

    # save the last model
    ## 将新模型保存在旧的模型所在目录，后面加上opt.method标志
    # full_model_path_new = r'C:\Users\user\Desktop\fast-test-2024-08-04\unsup-models\iteration-0\fastAug5\AE_CNN_epoch29_datalen180_gps-acceleration_%s.pth' % opt.method
    # state = {
    #     'opt': opt,
    #     'model': model.state_dict(),
    #     'optimizer': optimizer.state_dict(),
    #     'epoch': epoch,
    # }
    # torch.save(state, full_model_path_new)
    app.exec()


if __name__ == '__main__':
    test()
    # main()
