
# import argparse
import os
import pickle
from pathlib import Path
# from typing import List
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
import torch
from deepview.utils import auxiliaryfunctions
from deepview.clustering_pytorch.util.logging import setup_logging
from deepview.clustering_pytorch.config import load_config
from deepview.clustering_pytorch.nnet.common_config import (
    # get_criterion,
    # get_optimizer,
    get_model,
    # adjust_learning_rate,
)
from deepview.clustering_pytorch.datasets import (
    Batch,
    # DatasetFactory,
    # prepare_all_data,
    prepare_single_data,
)

def evaluate_network(
    config,
    testdata_path='',
    plotting=False,
    show_errors=True,
    gputouse=None,
    modelprefix="",
    per_keypoint_evaluation: bool = False,
):

    # 1 get pkl raw data from labeled-data
    ## Read file path for pose_config file. >> pass it on
    cfg = auxiliaryfunctions.read_config(config)
    modelfoldername = auxiliaryfunctions.get_model_folder(cfg, modelprefix=modelprefix)
    modelconfigfile = Path(
        os.path.join(
            cfg["project_path"], str(modelfoldername), "test", "model_cfg.yaml"
        )
    )

    os.chdir(
        str(Path(modelconfigfile).parents[0])
    )  # switch to folder of config_yaml (for logging)
    setup_logging()

    if testdata_path == '':
        rawdata_files = auxiliaryfunctions.get_labeled_data_folder(cfg)
        # with open(rawdata_files[0], 'rb') as f:
        #     rawdata = pickle.load(f)
        train_dataloader = prepare_single_data(rawdata_files[0])
    else:
        train_dataloader = prepare_single_data(testdata_path)

    # 2 get model from dv-models
    modelcfg = load_config(modelconfigfile)
    net_type = modelcfg["net_type"]
    # project_path = cfg['project_path']
    model_path = list(
        Path(
        os.path.join(
            cfg["project_path"], str(modelfoldername), "train")
        ).glob('*.pth')
    )[0]
    device = 'cpu'
    model = get_model(p_backbone=net_type, p_setup='autoencoder')  # set backbone model=ResNet18, SSL=simclr, weight
    model = model.to(device)
    print('Restart from checkpoint')
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.to(device)

    # 3 evaluating network
    representation_list, sample_list, timestamp_list, label_list, domain_list \
        = AE_eval_time_series(train_dataloader, model, device)

    # 4 save results to plot
    if testdata_path == '':
        with open(os.path.join(rawdata_files[0].parent, rawdata_files[0].stem+'_toplot.pkl'), 'wb') as f:
            pickle.dump([representation_list, sample_list, timestamp_list, label_list, domain_list], f)
    else:
        with open(os.path.join(os.path.dirname(testdata_path),
                               os.path.splitext(os.path.basename(testdata_path))[0]+'_toplot.pkl'), 'wb') as f:
            pickle.dump([representation_list, sample_list, timestamp_list, label_list, domain_list], f)
    return



def AE_eval_time_series(train_loader, model, device):

    model.eval()

    representation_list = []
    sample_list, timestamp_list, label_list, domain_list, timestr_list = [], [], [], [], []
    for i, (sample, timestamp, label, domain) in enumerate(train_loader):
        sample = sample.to(device=device, non_blocking=True, dtype=torch.float)
        # input_ = (sample).permute(0, 2, 1)  # input.shape=b512,3channel,90width
        input_ = sample

        # input of autoencoder will be 3D, the backbone is 1d-cnn
        x_encoded, output = model(input_)  # x_encoded.shape=batch512,outchannel128,len13

        # x_encoded, output = model(input_).view(b, 2, -1)  # output.shape=b,2,128, split the first dim into 2 parts
        tmp_representation = x_encoded.detach().cpu().numpy()
        representation_list.append(tmp_representation)
        sample_list.append(sample.cpu().numpy())
        timestamp_list.append(timestamp)
        # timestr_list.append(timestr)
        label_list.append(label)
        domain_list.append(domain)

    return representation_list, sample_list, timestamp_list, label_list, domain_list
