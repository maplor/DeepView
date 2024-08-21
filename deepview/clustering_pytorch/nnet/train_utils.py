# This Python file uses the following encoding: utf-8
"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import numpy as np
from deepview.clustering_pytorch.nnet.util import (
    AverageMeter,
    ProgressMeter,
    gen_aug,
)
# from util import gen_aug
from tqdm import tqdm


def simclr_train(train_loader, model, criterion, optimizer, epoch):
    """
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),
                             [losses],
                             prefix="Epoch: [{}]".format(epoch))

    model.train()

    for i, batch in enumerate(train_loader):
        images = batch['image']
        images_augmented = batch['image_augmented']
        b, c, h, w = images.size()  # batch64, channel3, height32, width32
        input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)  # input_.shape=b,2,c,h,w
        input_ = input_.view(-1, c, h, w)  # out.shape=b*2,c,h,w
        input_ = input_.cuda(non_blocking=True)
        targets = batch['target'].cuda(non_blocking=True)

        output = model(input_).view(b, 2, -1)  # output.shape=b,2,128, split the first dim into 2 parts
        loss = criterion(output)
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)


def simclr_train_time_series(train_loader, model, criterion, optimizer, epoch, device):
    """
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    """
    losses = AverageMeter('Loss', ':.4e')

    model.train()

    for i, (aug_sample1, aug_sample2, timestamp, label) in enumerate(tqdm(train_loader)):
        # aug_sample1 = gen_aug(sample, 't_warp')  # t_warp, out.shape=batch64,width3,height900
        # aug_sample2 = gen_aug(sample, 'negate')  # negate
        # aug_sample1 = aug_sample1.to(device=device, non_blocking=True, dtype=torch.double)
        # aug_sample2 = aug_sample2.to(device=device, non_blocking=True, dtype=torch.double)
        aug_sample1 = aug_sample1.to(dtype=torch.double)
        aug_sample2 = aug_sample2.to(dtype=torch.double)

        b, l, d = aug_sample1.size()  # batch64, length180, dim6
        input_ = torch.cat([aug_sample1.unsqueeze(1), aug_sample2.unsqueeze(1)], dim=1)  # input_.shape=b,2,l,d
        input_ = input_.view(-1, l, d)  # out.shape=b*2,l,d
        # input_ = input_.cuda(non_blocking=True)
        # input_ = input_.to(device='cuda', non_blocking=True, dtype=torch.float)
        # targets = batch['target'].cuda(non_blocking=True)

        output = model(input_).view(b, 2, -1)  # output.shape=b,2,128, split the first dim into 2 parts
        loss = criterion(output)
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return


def simclr_eval_time_series(train_loader, model, device):
    model.eval()

    representation_list = []
    sample_list = []
    for i, (sample, timestamp, label) in enumerate(train_loader):
        sample = sample.to(device=device, non_blocking=True, dtype=torch.double)
        b, l, d = sample.size()  # batch64, channel3, height32, width32
        input_ = torch.cat([sample.unsqueeze(1), sample.unsqueeze(1)], dim=1)  # input_.shape=b,2,c,h,w
        input_ = input_.view(-1, l, d)  # out.shape=b*2,c,h,w

        output = model(input_).view(b, 2, -1)  # output.shape=b,2,128, split the first dim into 2 parts
        tmp_representation = output[:, 0, :].detach().cpu().numpy()
        representation_list.append(tmp_representation)
        sample_list.append(sample.cpu().numpy())

    return representation_list, sample_list


# autoencoder training and test
def AE_train_time_series(train_loader, model, criterion, optimizer, epoch, device):
    """
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    """
    losses = AverageMeter('Loss', ':.4e')

    model.train()

    for i, (sample, timestamp, label, label_flag) in enumerate(tqdm(train_loader)):
        # aug_sample1 = gen_aug(sample, 't_warp')  # t_warp, out.shape=batch64,width3,height900
        # reshape data by adding channel to 1, and transpose height and width
        # sample = sample.to(device=device, non_blocking=True, dtype=torch.float)
        sample = sample.to(dtype=torch.float)

        # input of autoencoder will be 3D, the backbone is 1d-cnn
        x_encoded, output = model(sample)  # x_encoded.shape=batch512,outchannel128,len13
        loss = criterion(sample, output)
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print('loss of the ' + str(epoch) + '-th training epoch is :' + losses.__str__())
    print('loss of the ' + str(epoch) + '-th training epoch is :' + losses.__str__())
    return


def AE_eval_time_series(train_loader, model, device):
    model.eval()

    representation_list = []
    sample_list, timestamp_list, label_list, domain_list, timestr_list, flag_list = [], [], [], [], [], []
    for i, (sample, timestamp, label, label_flag) in enumerate(train_loader):
        sample = sample.to(device=device, non_blocking=True, dtype=torch.float)

        # input of autoencoder will be 3D, the backbone is 1d-cnn
        x_encoded, output = model(sample)  # x_encoded.shape=batch512,outchannel128,len13

        # x_encoded, output = model(input_).view(b, 2, -1)  # output.shape=b,2,128, split the first dim into 2 parts
        tmp_representation = x_encoded.detach().cpu().numpy()
        representation_list.append(tmp_representation)
        sample_list.append(sample.cpu().numpy())
        timestamp_list.append(timestamp)
        # timestr_list.append(timestr)
        label_list.append(label)
        flag_list.append(label_flag)
        # domain_list.append(domain)

    return representation_list, sample_list, timestamp_list, label_list, flag_list

def AE_eval_time_series_labelflag(train_loader, model, device):
    model.eval()

    representation_list = []
    sample_list, timestamp_list, label_list, domain_list, timestr_list, flag_list = [], [], [], [], [], []
    for i, (sample, timestamp, label, label_flag) in enumerate(train_loader):
        sample = sample.to(device=device, non_blocking=True, dtype=torch.float)
        # input_ = (sample).permute(0, 2, 1)  # input.shape=b512,3channel,90width
        # input_ = sample

        # input of autoencoder will be 3D, the backbone is 1d-cnn
        x_encoded, output = model(sample)  # x_encoded.shape=batch512,outchannel128,len13

        # x_encoded, output = model(input_).view(b, 2, -1)  # output.shape=b,2,128, split the first dim into 2 parts
        tmp_representation = x_encoded.detach().cpu().numpy()
        representation_list.append(tmp_representation)
        sample_list.append(sample.cpu().numpy())
        timestamp_list.append(timestamp)
        # timestr_list.append(timestr)
        label_list.append(label)
        flag_list.append(label_flag.detach().cpu().numpy())
        # domain_list.append(domain)

    return representation_list, sample_list, timestamp_list, label_list, flag_list

# ----------------------------------------step 2: clustering--------------------------------------

def scan_train(train_loader, model, criterion, optimizer, epoch, update_cluster_head_only=False):
    """
    Train w/ SCAN-Loss
    """
    total_losses = AverageMeter('Total Loss', ':.4e')
    consistency_losses = AverageMeter('Consistency Loss', ':.4e')
    entropy_losses = AverageMeter('Entropy', ':.4e')
    progress = ProgressMeter(len(train_loader),
                             [total_losses, consistency_losses, entropy_losses],
                             prefix="Epoch: [{}]".format(epoch))

    if update_cluster_head_only:
        model.eval()  # No need to update BN
    else:
        model.train()  # Update BN

    for i, batch in enumerate(train_loader):
        # Forward pass
        anchors = batch['anchor'].cuda(non_blocking=True)
        neighbors = batch['neighbor'].cuda(non_blocking=True)

        if update_cluster_head_only:  # Only calculate gradient for backprop of linear layer
            with torch.no_grad():
                anchors_features = model(anchors, forward_pass='backbone')
                neighbors_features = model(neighbors, forward_pass='backbone')
            anchors_output = model(anchors_features, forward_pass='head')
            neighbors_output = model(neighbors_features, forward_pass='head')

        else:  # Calculate gradient for backprop of complete network
            anchors_output = model(anchors)
            neighbors_output = model(neighbors)

            # Loss for every head
        total_loss, consistency_loss, entropy_loss = [], [], []
        for anchors_output_subhead, neighbors_output_subhead in zip(anchors_output, neighbors_output):
            total_loss_, consistency_loss_, entropy_loss_ = criterion(anchors_output_subhead,
                                                                      neighbors_output_subhead)
            total_loss.append(total_loss_)
            consistency_loss.append(consistency_loss_)
            entropy_loss.append(entropy_loss_)

        # Register the mean loss and backprop the total loss to cover all subheads
        total_losses.update(np.mean([v.item() for v in total_loss]))
        consistency_losses.update(np.mean([v.item() for v in consistency_loss]))
        entropy_losses.update(np.mean([v.item() for v in entropy_loss]))

        total_loss = torch.sum(torch.stack(total_loss, dim=0))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)


def selflabel_train(train_loader, model, criterion, optimizer, epoch, ema=None):
    """
    Self-labeling based on confident samples
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), [losses],
                             prefix="Epoch: [{}]".format(epoch))
    model.train()

    for i, batch in enumerate(train_loader):
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)

        with torch.no_grad():
            output = model(images)[0]
        output_augmented = model(images_augmented)[0]

        loss = criterion(output, output_augmented)
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ema is not None:  # Apply EMA to update the weights of the network
            ema.update_params(model)
            ema.apply_shadow(model)

        if i % 25 == 0:
            progress.display(i)
