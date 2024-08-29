
import math
import os
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from deepview.clustering_pytorch.datasets.factory import (
extend_data_column,
sliding_window,
process_pressure_sensor,
)
from deepview.gui.supervised_cl.train.losses import SupConLoss
from deepview.gui.supervised_cl.util import AverageMeter
from deepview.gui.label_with_interactive_plot.utils import reduce_dimension_with_tsne
# from deepview.gui.supervised_cl.util import warmup_learning_rate

def get_window_data_scl(data, data_columns, len_sw):
    extend_column = extend_data_column(data_columns)

    # process the pressure column by reducing 1013.25hPa
    data = process_pressure_sensor(data, extend_column)
    tmp = sliding_window(data[extend_column],
                         len_sw)  # temp:['acc_x', 'acc_y', 'acc_z', 'timestamp', 'labelid', 'domain']

    selected_data = tmp[:, :, :-3]
    label_flag = tmp[:, :, -3:-2]
    timestamp = tmp[:, :, -2:-1]
    label = tmp[:, :, -1:]
    return selected_data, label_flag, timestamp, label

def load_model_parameters(model, full_model_path, device='cpu'):
    state_dict = torch.load(full_model_path, map_location=torch.device(device))
    # state_dict = torch.load(full_model_path, map_location=torch.device('cpu'))
    filtered_state_dict = {k: v for k, v in state_dict.items() if 'backbone' in k}

    # Extract and load only the feature extractor part
    new_state_dict = model.state_dict()
    for name, param in filtered_state_dict.items():
        if 'backbone' in name:
            new_state_dict[name] = param
    model.load_state_dict(new_state_dict)

    return model

def set_optimizer(model, learning_rate, momentum, weight_decay):
    optimizer = optim.SGD(model.parameters(),
                          lr=learning_rate,
                          momentum=momentum,
                          weight_decay=weight_decay)
    return optimizer

def get_scl_criterion_opt(model, device):
    criterion = SupConLoss(temperature=0.07)

    if 'cuda' in device.type:
        if torch.cuda.device_count() > 1:
            # model.encoder = torch.nn.DataParallel(model.encoder)
            model.backbone = torch.nn.DataParallel(model.backbone)
        # model = model.cuda()
        # criterion = criterion.cuda()
        cudnn.benchmark = True
    model = model.to(device)
    criterion = criterion.to(device)
    # build optimizer
    optimizer = set_optimizer(model, 0.05, 0.9, 1e-4)
    return model, criterion, optimizer

def train(train_loader, model, method, criterion, optimizer, epoch, num_epochs, device):
    """one epoch training"""
    model.train()

    # batch_time = AverageMeter()
    losses = AverageMeter()

    for idx, (aug_sample1, aug_sample2, timestamp, labels, label_flags) in enumerate(tqdm(train_loader)):
        aug_sample1 = aug_sample1.to(dtype=torch.double)
        aug_sample2 = aug_sample2.to(dtype=torch.double)

        images = torch.cat([aug_sample1, aug_sample2], dim=0)
        images = images.to(device, non_blocking=True)
        labels = labels[:, int(labels.shape[1] / 2), 0].to(device, non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        # warmup_learning_rate(epoch, idx, len(train_loader), optimizer)
        warmup_learning_rate(epoch, num_epochs, bsz, idx, len(train_loader), optimizer)

        # compute loss
        features, backboneout = model(images)

        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)  # batch, 2, 128
        if method == 'Supervised_SimCLR':
            loss = criterion(features, labels)
        elif method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(method))

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


def evaluate(model, train_loader, device):
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

    # # Create a scatter plot
    # plt.figure(figsize=(8, 6))
    #
    # real_label_idxs = np.where(flag_concat == 0)[0]
    # x = repre_tsne[real_label_idxs, 0]
    # y = repre_tsne[real_label_idxs, 1]
    # plt.scatter(x, y, color='grey', alpha=0.5, label='unlabeled')
    #
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
    #
    # # Add labels and title
    # plt.xlabel('X-axis Label')
    # plt.ylabel('Y-axis Label')
    # plt.title('supervised contrastive latent, %s' % str(epoch))
    # plt.legend()
    # plt.savefig('%s_%s.png' % (fig_name, str(epoch)))
    return repre_tsne, flag_concat, label_concat

def training_parameters():
    # warm_epochs = 10
    # warmup_from = 0.01
    learning_rate = 0.05
    lr_decay_rate = 0.1
    return learning_rate, lr_decay_rate


# def adjust_learning_rate(args, optimizer, epoch):
#     lr = args.learning_rate
#     if args.cosine:
#         eta_min = lr * (args.lr_decay_rate ** 3)
#         lr = eta_min + (lr - eta_min) * (
#                 1 + math.cos(math.pi * epoch / args.epochs)) / 2
#     else:
#         steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
#         if steps > 0:
#             lr = lr * (args.lr_decay_rate ** steps)
#
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#
def adjust_learning_rate(optimizer, epoch, nepochs):
    lr, lr_decay_rate = training_parameters()
    # lr = args.learning_rate

    eta_min = lr * (lr_decay_rate ** 3)
    lr = eta_min + (lr - eta_min) * (
            1 + math.cos(math.pi * epoch / nepochs)) / 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
#     if args.warm and epoch <= args.warm_epochs:
#         p = (batch_id + (epoch - 1) * total_batches) / \
#             (args.warm_epochs * total_batches)
#         lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)
#
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr

def warmup_learning_rate(epoch, num_epochs, batch_size, batch_id, total_batches, optimizer):
    warm_epochs = 10
    warmup_from = 0.01
    learning_rate, lr_decay_rate = training_parameters()

    # warm-up for large-batch training,
    if batch_size > 256:
        eta_min = learning_rate * (lr_decay_rate ** 3)
        warmup_to = eta_min + (learning_rate - eta_min) * (
                1 + math.cos(math.pi * warm_epochs / num_epochs)) / 2

        if epoch <= warm_epochs:
            p = (batch_id + (epoch - 1) * total_batches) / \
                (warm_epochs * total_batches)
            lr = warmup_from + p * (warmup_to - warmup_from)

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

