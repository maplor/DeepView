
# This Python file uses the following encoding: utf-8
import argparse
import logging
import os
import threading
import warnings
from pathlib import Path

import torch
from deepview.clustering_pytorch.util.logging import setup_logging
from deepview.clustering_pytorch.config import load_config
from deepview.clustering_pytorch.datasets import (
    Batch,
    # DatasetFactory,
    prepare_all_data,
    # prepare_single_data,
)
from deepview.clustering_pytorch.nnet.common_config import (
    get_criterion,
    get_optimizer,
    get_model,
    adjust_learning_rate,
)

from deepview.clustering_pytorch.nnet.train_utils import (
    AE_train_time_series,
    simclr_train_time_series,
)
import json
from PySide6.QtWidgets import QProgressBar


class LearningRate(object):
    def __init__(self, cfg):
        self.steps = cfg["multi_step"]
        self.current_step = 0

    def get_lr(self, iteration):
        lr = self.steps[self.current_step][0]
        if iteration == self.steps[self.current_step][1]:
            self.current_step += 1

        return lr

def get_batch_spec(cfg):
    num_joints = cfg["num_joints"]
    batch_size = cfg["batch_size"]
    return {
        Batch.inputs: [batch_size, None, None, 3],
        Batch.part_score_targets: [batch_size, None, None, num_joints],
        Batch.part_score_weights: [batch_size, None, None, num_joints],
        Batch.locref_targets: [batch_size, None, None, num_joints * 2],
        Batch.locref_mask: [batch_size, None, None, num_joints * 2],
    }

def train(
    sensor_dict,
    progress_update,
    config_yaml,
    select_filenames,
    net_type,
    lr,
    batch_size,
    num_epochs,
    data_len,
    data_column
):
    # data_column_list = json.loads(data_column.replace('\'', '"'))
    start_path = os.getcwd()
    try:
        os.chdir(
            str(Path(config_yaml).parents[0])
        )  # switch to folder of config_yaml (for logging)
    except:
        from deepview.utils import auxiliaryfunctions
        auxiliaryfunctions.create_folders_from_string(str(Path(config_yaml).parents[0]))
    setup_logging()

    cfg = load_config(config_yaml)
    project_path = cfg['project_path']
    from deepview.utils import auxiliaryfunctions
    data_path = os.path.join(project_path, auxiliaryfunctions.get_unsupervised_set_folder())

    # device = 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # xia, dataloader,将Train network tab中选中的文件传入这个函数， todo 目前写死了
    if net_type in ['AE_CNN', 'shortAE']:
        augment = False
    elif net_type in ['SIMCLR_LSTM']:
        augment = True
    else:
        augment = False
    train_dataloader, num_channel = prepare_all_data(data_path,
                                                     select_filenames,
                                                     data_len,
                                                     data_column,
                                                     batch_size,
                                                     augment,
                                                     device)

    # print(optimizer)
    # -------------------------核心的模型训练部分，计算loss----------------------------

    # get model, todo, combine to above
    # 相同代码在supervised contrastive learning的runscl出现
    if net_type.upper() in ['AE_CNN'.upper(), 'shortAE'.upper()]:
        p_setup = 'autoencoder'
    elif net_type.upper() in ['simclr'.upper()]:
        p_setup = 'simclr'
    else:
        raise ValueError('Unknown framework type: %s' % net_type)

    model = get_model(p_backbone=net_type,
                      p_setup=p_setup,
                      num_channel=num_channel,
                      data_len=data_len)  # set backbone model=ResNet18, SSL=simclr, weight
    model = model.to(device)

    # Criterion
    criterion = get_criterion(p_setup)
    criterion = criterion.to(device)

    # Optimizer and scheduler
    p_opti = 'sgd'
    optimizer = get_optimizer(p_opti, model)

    # training
    start_epoch = 0
    # i = 1  # 应该一只鸟一个trainloader，暂时全拼接到一起
    # print('Starting %s-th bird data' % str(i))
    for epoch in range(start_epoch, num_epochs):
        progress_update.emit(int((epoch+1)/num_epochs * 100))
        # Adjust lr
        lr = adjust_learning_rate(lr, optimizer, epoch, p_scheduler='cosine', p_epochs=num_epochs)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train: the same as simclr
        print('Train ...')
        if p_setup == 'autoencoder':
            AE_train_time_series(train_dataloader, model, criterion, optimizer, epoch, device)
        elif p_setup == 'simclr':
            simclr_train_time_series(train_dataloader, model, criterion, optimizer, epoch, device)

    sensor_str = ''
    for i in data_column:
        for sensor, sensor_axis in sensor_dict.items():
            if i in sensor_axis:
                tmp = sensor
            elif i in ['GPS_velocity', 'GPS_bearing']:
                tmp = 'gps'
            else:
                continue
        # tmp = [k for k, v in sensor_dict.items() if i in v][0]  # get key according to value
        if tmp not in sensor_str:
            sensor_str = tmp + '-' + sensor_str

    # Save final model, xx\aaa-bbb-2024-04-24\unsup-models\iteration-0\aaaApr24\train
    # column_list = '-'.join(data_column).replace('_','')
    print('Saving model at: ' + net_type + '_epoch%s' % str(epoch)\
          + '_datalen%s_' % str(cfg['data_length']) \
          + sensor_str[:-1] + '.pth')
    torch.save(model.state_dict(), net_type + '_epoch%s' % str(epoch) +\
               '_datalen%s_' % str(cfg['data_length']) + sensor_str[:-1] + '.pth')

    # return to original path.
    os.chdir(str(start_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to yaml configuration file.")
    cli_args = parser.parse_args()

    train(Path(cli_args.config).resolve())
