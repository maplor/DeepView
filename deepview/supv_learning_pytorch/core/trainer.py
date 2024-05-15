"""
A trainer for deep learning models

Otsuka et al., (2024) Methods in Ecology and Evolution
"Exploring deep Learning techniques for wild animal behaviour classification using animal-borne accelerometers"

"""

import os
import glob
import copy
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from timm.scheduler import CosineLRScheduler
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score, precision_score, recall_score
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
# from . import models, utils
from deepview.supv_learning_pytorch import models
from deepview.supv_learning_pytorch.utils import utils
from deepview.supv_learning_pytorch.utils.pytorchtools import EarlyStopping
from tqdm import tqdm


MODEL_LIST_01 = [
    "mlp",
    "cnn", 
    "lstm", 
    "dcl",
    "dcl-v3", # DCL mixup after LSTM version
    "dcl-sa", 
    "resnet-l-sa", 
    "transformer",
    "cnn-ae-wo",
    "cnn-ae",
]

def setup_model(cfg):
    # model configuration
    if cfg.model.model_name == "cnn": # CNN
        model = models.cnn.CNN(cfg)
    elif cfg.model.model_name == 'lstm': # LSTM
        model = models.lstm.LSTM(cfg)
    elif cfg.model.model_name == 'dcl': # DCL
        model = models.dcl.DeepConvLSTM(cfg)
    elif cfg.model.model_name == 'dcl-v3': # Mixup After LSTM layer
        model = models.dcl_v3.DeepConvLSTM3(cfg)
    elif cfg.model.model_name == 'dcl-sa': # DCLSA
        model = models.dcl_sa.DeepConvLSTMSelfAttn(cfg)
    elif cfg.model.model_name == 'resnet-l-sa': # DCLSA-RN (ResNet version of DCLSA)
        model = models.resnet_l_sa.resnet_lstm_selfattn(cfg) 
    elif cfg.model.model_name == 'transformer':
        model = models.transformer.Transformer(cfg)
    elif cfg.model.model_name in ['cnn-ae']: # CNN_AE5 for unsupervised pre-training
        model = models.cnn_ae_v5.CNN_AE5(cfg)
    elif cfg.model.model_name in ["cnn-ae-wo"]: # CNN_AE6 for hyperparameter tuning
        model = models.cnn_ae_v6.CNN_AE6(cfg)
    else:
        raise Exception(
            f"cfg.model.model_name: {cfg.model.model_name} is not appropriate.")

    return model


def setup_scheduler(cfg, optimizer):
    # scheduler
    if cfg.train.scheduler in ["CosineLR", True]:
        # https://timm.fast.ai/SGDR
        scheduler = CosineLRScheduler(
            optimizer, 
            t_initial=cfg.train.n_epoch, 
            lr_min=1e-6, 
            warmup_t=0, 
            warmup_lr_init=0, 
            warmup_prefix=False)
    elif cfg.train.scheduler == "Plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.5,
            patience=7, # default 10
            threshold=1e-4,
            threshold_mode='rel',
            cooldown=0,
            min_lr=0,
            eps=1e-8,
            verbose=False)
    else:
        scheduler = None
        
    return scheduler


def train(
    model,
    optimizer,
    criterion,
    train_loader,
    test_loader,
    DEVICE,
    cfg
):

    best_val_f1 = 0
    best_model = None

    scheduler = setup_scheduler(cfg, optimizer)

    # initialization
    epoch_list = []
    train_loss_list = []
    train_acc_list = []
    train_f1_list = []
    test_loss_list = []
    test_acc_list = []
    test_f1_list = []
    learning_rate_list = []
    early_stopping_coutner_list = []
    patience_list = []
    
    label_species = utils.get_label_species(cfg)
    
    # path to "best_model_weights.pt"
    path = Path(
        cfg.path.log.rootdir,
        cfg.path.log.checkpoints.dir,
        cfg.path.log.checkpoints.fname,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    # log.info(f"early stopping directory: {path}")
    early_stopping = EarlyStopping(patience=cfg.train.patience, path=path)

    # define your training loop; iterates over the number of epochs
    for epoch in tqdm(range(cfg.train.n_epoch)):

        # ---------------------------------------------------------------------------
        # Training
        # ---------------------------------------------------------------------------
        model.train()
        train_losses = []
        train_preds = []
        train_gt = []
        num_batch = 0
        num_batch_with_na = 0
        
        for idx, (sample, target) in enumerate(train_loader):
            
            # initialize optimizer
            optimizer.zero_grad()
            
            #############################################
            # Data preparation
            #############################################
            
            # -------------------------------------
            # Unsupervised Pre-training of CNN-AE
            # -------------------------------------
            if cfg.model.model_name in ["cnn-ae"] and cfg.model.pretrain == True:
                if sample.size(0) != cfg.train.batch_size:
                    continue # skip
                if torch.sum(torch.isnan(sample)) > 0:
                    num_batch_with_na += 1
                    continue # skip
                window_size = len(sample[0][0]) 
                # print(f"window_size: {window_size}") # window_size: 50
                # print(f"sample.shape: {sample.shape}") # sample.shape: torch.Size([30, 20, 50, 3])
                sample = sample.reshape(cfg.train.batch_size * 20, window_size, 3) # (BS, N_windows, T, CH) -> (BS, T, CH)
                target = target.reshape(cfg.train.batch_size * 20, 1, 50) # (BS, 1, T)
                x = sample.transpose(1, 2).unsqueeze(3) # (BS, CH, T, 1)
                y = target.transpose(1, 2) # (BS, T, 1)
            # -------------------------------------
            # Supervised Model Training
            # -------------------------------------
            else:
                x = sample.transpose(1, 3) # (BS, 1, T, CH) -> (BS, CH, T, 1)
                y = target.transpose(1, 2) # (BS, 1, T) -> (BS, T, 1)
            inputs = x.to(device=DEVICE, dtype=torch.float)
            targets = y.to(device=DEVICE, dtype=torch.long)
            targets = utils.convert_torch_labels(targets, label_species)
            if "flatten-linear" in cfg.model.out_layer_type:
                    # reshape the target (BS, T, 1) -> (BS,)
                    targets_list = []
                    for i in range(0, targets.shape[0]):
                        # print(f"{i}: {targets[i][-1]}")
                        targets_list.extend(targets[i][-1])
                    _targets = torch.tensor(targets_list)
                    targets = _targets.to(device=DEVICE, dtype=torch.long)
                    # Cross Entropy loss inputs: train_output: (B, C) / targets: (B)
                    # train_loss = criterion(train_output, targets)
            # else: # 1×1 conv layer as output
                # Cross Entropy loss inputs: train_output: (B, C, T, 1) / targets: (B, T, 1)
                # train_loss = criterion(train_output, targets)
            
            # check the inputs and targets shape
            # if epoch == 0 and idx == 0:
                # log.info(f"inputs.shape: {inputs.shape}")   # -> (BS, CH, T, 1)
                # log.info(f"targets.shape: {targets.shape}") # -> (BS, T, 1) or (BS,)
            
            
            #############################################
            # Feed Data to the model and calculate loss 
            #############################################
            
            # -------------------------------------
            # Unsupervised Pre-training of CNN-AE
            # -------------------------------------
            if cfg.model.model_name == "cnn-ae" and cfg.model.pretrain == True:
                train_output, x_decoded, x_encoded = model(inputs)
                train_loss = torch.nn.functional.mse_loss(inputs, x_decoded) # Reconstruction loss
            # -------------------------------------
            # Supervised Model Training
            # -------------------------------------
            else: # supervised training
                if cfg.model.model_name in MODEL_LIST_01:
                    # Manifold Mixup
                    if cfg.train.manifold_mixup:
                        train_output, y_onehot, features = model(
                            inputs, 
                            targets, 
                            mixup=True,
                            mixup_alpha=cfg.train.mixup_alpha
                        )
                        
                        # bce_loss = torch.nn.BCELoss().cuda()
                        softmax = torch.nn.Softmax(dim=1).cuda()

                        if cfg.train.mixup_targets_argmax == True:
                            # -----------------------------------------------------------------------
                            # reweighted targets (one-hot) -> Label encoding (n_classes = 6)
                            # -----------------------------------------------------------------------
                            # sample-wise: reshape (B, T, 1, 6) -> (B, 6, T, 1) in the model (see dl_model.py)
                            # dl-wabc: (B, T, 1, 6) -> (B, 6, T, 1) in the model (see dl_model.py)
                            targets = torch.argmax(y_onehot, dim=1) # one-hot encoding -> label encoding
                            if epoch == 0 and idx == 0:
                                print(f"targets.shape: {targets.shape}")
                            train_loss = criterion(train_output, targets) # Cross Entropy loss
                        else:
                            # One-hot encoding (n_classes = 6)
                            # update targets after mixup to calculate train scores for log
                            train_loss = criterion(train_output, y_onehot) # Cross Entropy loss
                            targets = torch.argmax(y_onehot, dim=1) # one-hot encoding -> label encoding
                    else:
                        train_output, _, features = model(
                            inputs, 
                            targets,
                            mixup=False,
                            mixup_alpha=cfg.train.mixup_alpha
                        )
                        # Cross Entropy loss
                        train_loss = criterion(train_output, targets)
                else:
                    raise Exception(f'cfg.model.model_name "{cfg.model.model_name}" is not appropriate.')
                
                # if epoch == 0 and idx == 0:
                #     log.info(f"train_output.shape: {train_output.shape}")   # -> (BS, CH, T, 1)
                #     log.info(f"targets.shape: {targets.shape}") # -> (BS, T, 1) or (BS,)
            
            # back propagation                
            train_loss.backward()
            optimizer.step()
            
            # prediction
            train_output = torch.nn.functional.softmax(train_output, dim=1)
            train_losses.append(train_loss.item())
            y_preds = np.argmax(
                train_output.cpu().detach().numpy(),
                axis=1).flatten()
            train_preds = np.concatenate(
                (np.array(train_preds, int), 
                 np.array(y_preds, int)))
            
            # ground truth
            y_true = targets.cpu().numpy().flatten()
            train_gt = np.concatenate(
                (np.array(train_gt, int), 
                 np.array(y_true, int)))
            
            # batch counter
            num_batch += 1
            
        # save model weights
        model_save_path = Path(
            cfg.path.log.rootdir, 
            cfg.path.log.checkpoints.dir,
            f"model_weights_epoch_{epoch:0=3}.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model_state_dict': model.state_dict()}, model_save_path)
        
        # evaluation
        train_loss_mean = np.mean(train_losses)
        if cfg.model.model_name == "cnn-ae" and cfg.model.pretrain == True:
            m1 = f"Train Loss: {np.sum(train_losses):.6f}"
            # log.info(f'{m1}')
            train_accuracy = 0
            train_f1 = 0
            # log.info(f'num_batch: {num_batch}')
            # log.info(f'num_batch_with_na: {num_batch_with_na}')
        else:
            train_accuracy = jaccard_score(
                train_gt, train_preds, average='macro')
            train_precision = precision_score(
                train_gt, train_preds, average='macro')
            train_recall = recall_score(train_gt, train_preds, average='macro')
            train_f1 = f1_score(train_gt, train_preds, average='macro')
            m1 = f"Train Loss: {train_loss_mean:.3f}"
            m2 = f"Acc: {train_accuracy:.3f}"
            m3 = f"Precision: {train_precision:.3f}"
            m4 = f"Recall: {train_recall:.3f}"
            m5 = f"F1: {train_f1:.3f}"
            # log.info(f'{m1} {m2} {m3} {m4} {m5}')
            print(f'{m1} {m2} {m3} {m4} {m5}')
            
            # check number of batches
            # log.info(f'num_batch: {num_batch}')
            # log.info(f'num_samples: {num_batch*cfg.train.batch_size}')
        
        learning_rate_tmp = optimizer.param_groups[0]['lr']

        if cfg.train.scheduler in ["CosineLR"]:
            scheduler.step(epoch)
        elif cfg.train.scheduler in ["Plateau"]:
            scheduler.step()


        # ---------------------------------------------------------------------------
        # Test result
        # ---------------------------------------------------------------------------
        test_preds = []
        test_gt = []
        test_losses = []
        model.eval()
        with torch.no_grad():
            n_batches = 0
            for idx, (sample, target) in enumerate(test_loader):

                #############################################
                # Data preparation
                #############################################
                if cfg.model.model_name in ["cnn-ae"] and cfg.model.pretrain == True:
                    if sample.size(0) != cfg.train.batch_size:
                        continue  # skip
                    if torch.sum(torch.isnan(sample)) > 0:
                        continue  # skip
                    window_size = len(sample[0][0])
                    sample = sample.reshape(cfg.train.batch_size * 20, window_size, 3)  # (BS, T, CH)
                    target = target.reshape(cfg.train.batch_size * 20, 1, 50)  # (BS, 1, T)
                    x = sample.transpose(1, 2).unsqueeze(3)  # (BS, CH, T, 1)
                    y = target.transpose(1, 2)  # (BS, T, 1)
                else:
                    x = sample.transpose(1, 3)
                    y = target.transpose(1, 2)

                inputs = x.to(DEVICE, dtype=torch.float)
                targets = y.to(DEVICE, dtype=torch.long)
                targets = utils.convert_torch_labels(targets, label_species)

                if "flatten-linear" in cfg.model.out_layer_type:
                    # Reshape the targets
                    targets_list = []
                    for i in range(0, targets.shape[0]):
                        # print(f"{i}: {targets[i][-1]}")
                        targets_list.extend(targets[i][-1])
                    _targets = torch.tensor(targets_list)
                    targets = _targets.to(device=DEVICE, dtype=torch.long)

                #############################################
                # Feed Data to the model and calculate loss
                #############################################
                if cfg.model.model_name == "cnn-ae" and cfg.model.pretrain == True:
                    test_output, x_decoded, x_encoded = model(inputs)
                    test_loss = torch.nn.functional.mse_loss(inputs, x_decoded)
                else:
                    test_output, _, _ = model(
                        inputs,
                        targets,
                        mixup=False,
                        mixup_alpha=cfg.train.mixup_alpha
                    )
                    test_loss = criterion(test_output, targets)
                test_losses.append(test_loss.item())

                # prediction
                test_output = torch.nn.functional.softmax(test_output, dim=1)
                y_preds = np.argmax(
                    test_output.cpu().numpy(),
                    axis=1).flatten()
                test_preds = np.concatenate(
                    (np.array(test_preds, int),
                     np.array(y_preds, int)))

                # ground truth
                y_true = targets.cpu().numpy().flatten()
                test_gt = np.concatenate(
                    (np.array(test_gt, int),
                     np.array(y_true, int)))

                n_batches += 1

            # evaluation
            test_loss_mean = np.mean(test_losses)
            if cfg.model.model_name == "cnn-ae" and cfg.model.pretrain == True:
                m1 = f"Test Loss: {np.sum(test_losses):.6f}"
                # log.info(f'{m1}')
                print(f'{m1}')
                test_accuracy = 0
                test_f1 = 0
            else:
                test_accuracy = jaccard_score(test_gt, test_preds, average='macro')
                test_precision = precision_score(test_gt, test_preds, average='macro')
                test_recall = recall_score(test_gt, test_preds, average='macro')
                test_f1 = f1_score(test_gt, test_preds, average='macro')
                m1 = f"Test Loss:   {test_loss_mean:.3f}"
                m2 = f"Test Acc: {test_accuracy:.3f}"
                m3 = f"Test Precision: {test_precision:.3f}"
                m4 = f"Test Recall: {test_recall:.3f}"
                m5 = f"macro F1: {test_f1:.3f}"
                # log.info(f'{m1} {m2} {m3} {m4} {m5}')
                print(f'{m1} {m2} {m3} {m4} {m5}')


        # early stopping
        # log.info(
        #     f'EarlyStopping counter: {early_stopping.counter} out of {early_stopping.patience}')
        early_stopping_coutner_list.append(early_stopping.counter)
        patience_list.append(early_stopping.patience)

        # learning rate
        learning_rate_list.append(learning_rate_tmp)

        # monitor training process 
        epoch_list.append(epoch)
        train_loss_list.append(train_loss_mean)
        train_acc_list.append(train_accuracy)
        train_f1_list.append(train_f1)
        val_loss_list.append(val_loss_mean)
        val_acc_list.append(val_accuracy)
        val_f1_list.append(val_f1)
        test_loss_list.append(test_loss_mean)
        test_acc_list.append(test_accuracy)
        test_f1_list.append(test_f1)

        # the best model based on F1-score (this is not used for analysis)
        if cfg.model.model_name == "cnn-ae" and cfg.model.pretrain == True:
            best_model = copy.deepcopy(model)
        else: 
            current_val_f1 = f1_score(val_gt, val_preds, average='macro')
            if current_val_f1 > best_val_f1:
                best_model = copy.deepcopy(model)
                best_val_f1 = current_val_f1

        # if early_stopping.early_stop:
        #     log.info("Early Stopping")
        #     break

    # DataFrame for monitoring training curve
    df_log = pd.DataFrame(list(zip(epoch_list,
                                   train_loss_list,
                                   train_acc_list,
                                   train_f1_list,
                                   val_loss_list,
                                   val_acc_list,
                                   val_f1_list,
                                   test_loss_list,
                                   test_acc_list,
                                   test_f1_list,
                                   learning_rate_list,
                                   early_stopping_coutner_list,
                                   patience_list)),
                          columns=["epoch",
                                   "train_loss",
                                   "train_acc",
                                   "train_f1",
                                   "val_loss",
                                   "val_acc",
                                   "val_f1",
                                   "test_loss",
                                   "test_acc",
                                   "test_f1",
                                   "learning_rate",
                                   "early_stopping_counter",
                                   "patience"])

    # if hasattr(train_loader, "unload"):
    #     train_loader.unload()
    # if hasattr(val_loader, "unload"):
    #     val_loader.unload()

    return best_model, df_log


# prediction
def test(
    best_model,
    optimizer,
    criterion,
    test_loader,
    DEVICE,
    cfg
):

    test_preds = []
    test_gt = []
    test_losses = []    
    label_species = utils.get_label_species(cfg)
    best_model.eval()
    with torch.no_grad():
        for idx, (sample, target) in enumerate(test_loader):
            x = sample.transpose(1, 3)
            y = target.transpose(1, 2)
            inputs = x.to(DEVICE, dtype=torch.float)
            targets = y.to(DEVICE, dtype=torch.long)
            targets = utils.convert_torch_labels(targets, label_species)
            test_output, _, _ = best_model(
                inputs, 
                targets,
                mixup=False,
                mixup_alpha=cfg.train.mixup_alpha)
            test_loss = criterion(test_output, targets)
            test_output = torch.nn.functional.softmax(test_output, dim=1)
            test_losses.append(test_loss.item())
            # prediction
            y_preds = np.argmax(
                test_output.cpu().numpy(),
                axis=1).flatten()
            test_preds = np.concatenate(
                (np.array(test_preds, int), 
                 np.array(y_preds, int)))
            # ground truth
            y_true = targets.cpu().numpy().flatten()
            test_gt = np.concatenate(
                (np.array(test_gt, int), 
                 np.array(y_true, int)))
            
        test_loss_mean = np.mean(test_losses)
        test_accuracy = jaccard_score(test_gt, test_preds, average='macro')
        test_precision = precision_score(test_gt, test_preds, average='macro')
        test_recall = recall_score(test_gt, test_preds, average='macro')
        test_f1 = f1_score(test_gt, test_preds, average='macro')
        
        m1 = f"Test Loss: {test_loss_mean:.3f}"
        m2 = f"Acc: {test_accuracy:.3f}"
        m3 = f"Precision: {test_precision:.3f}"
        m4 = f"Recall: {test_recall:.3f}"
        m5 = f"F1: {test_f1:.3f}"
        print(f'{m1} {m2} {m3} {m4} {m5}')

    y_gt = test_gt
    y_pred = test_preds

    cm, df_cm, fig_cm = utils.plot_confusion_matrix(y_gt, 
                                                    y_pred, 
                                                    cfg, 
                                                    figsize=(7,6))

    return y_gt, y_pred, cm, df_cm, fig_cm


# prediction + features
def test2(
    best_model,
    optimizer,
    criterion,
    test_loader,
    DEVICE,
    cfg
):

    test_preds = []
    test_gt = []
    test_losses = []
    features = None
    label_species = utils.get_label_species(cfg)
    best_model.eval()
    with torch.no_grad():
        for idx, (sample, target) in enumerate(test_loader):
            
            x = sample.transpose(1, 3)
            y = target.transpose(1, 2)
            inputs = x.to(DEVICE, dtype=torch.float)
            targets = y.to(DEVICE, dtype=torch.long)
            targets = utils.convert_torch_labels(targets, label_species)
            
            if "flatten-linear" in cfg.model.out_layer_type:
                # Reshape the targets
                targets_list = []
                for i in range(0, targets.shape[0]):
                    # print(f"{i}: {targets[i][-1]}")
                    targets_list.extend(targets[i][-1])
                _targets = torch.tensor(targets_list)
                targets = _targets.to(device=DEVICE, dtype=torch.long)
            
            # feed data to the model
            test_output, y_onehot, feature = best_model(
                inputs, 
                targets,
                mixup=False,
                mixup_alpha=cfg.train.mixup_alpha
            )
            
            # calculate loss
            test_loss = criterion(test_output, targets)
            test_output = torch.nn.functional.softmax(test_output, dim=1)
            test_losses.append(test_loss.item())
            # prediction
            y_preds = np.argmax(
                test_output.cpu().numpy(),
                axis=1).flatten()
            test_preds = np.concatenate(
                (np.array(test_preds, int), 
                 np.array(y_preds, int)))
            # ground truth
            y_true = targets.cpu().numpy().flatten()
            test_gt = np.concatenate(
                (np.array(test_gt, int), 
                 np.array(y_true, int)))
            
            if features is None:
                features = feature
            else:
                features = torch.cat((features, feature), 0)
            
        test_loss_mean = np.mean(test_losses)
        test_accuracy = jaccard_score(test_gt, test_preds, average='macro')
        test_precision = precision_score(test_gt, test_preds, average='macro')
        test_recall = recall_score(test_gt, test_preds, average='macro')
        test_f1 = f1_score(test_gt, test_preds, average='macro')
        
        m1 = f"Test Loss: {test_loss_mean:.3f}"
        m2 = f"Acc: {test_accuracy:.3f}"
        m3 = f"Precision: {test_precision:.3f}"
        m4 = f"Recall: {test_recall:.3f}"
        m5 = f"F1: {test_f1:.3f}"
        print(f'{m1} {m2} {m3} {m4} {m5}')

    y_gt = test_gt
    y_pred = test_preds

    cm, df_cm, fig_cm = utils.plot_confusion_matrix(y_gt, 
                                                    y_pred, 
                                                    cfg, 
                                                    figsize=(6.5, 5.5))

    return y_gt, y_pred, features, cm, df_cm, fig_cm


# Reconstruction visualization for Autoencoder models
def test3(
    best_model,
    optimizer,
    criterion,
    test_loader,
    DEVICE,
    cfg
):
    window_plot_counter = 0
    
    test_preds = []
    test_gt = []
    test_losses = []
    features = None
    label_species = utils.get_label_species(cfg)
    best_model.eval()
    with torch.no_grad():
        for idx, (sample, target) in enumerate(test_loader):
            x = sample.transpose(1, 3)
            y = target.transpose(1, 2)
            inputs = x.to(DEVICE, dtype=torch.float)
            targets = y.to(DEVICE, dtype=torch.long)
            targets = utils.convert_torch_labels(targets, label_species)
            
            if "flatten-linear" in cfg.model.out_layer_type:
                # Reshape the targets
                targets_list = []
                for i in range(0, targets.shape[0]):
                    # print(f"{i}: {targets[i][-1]}")
                    targets_list.extend(targets[i][-1])
                _targets = torch.tensor(targets_list)
                targets = _targets.to(device=DEVICE, dtype=torch.long)
            
            test_output, x_decoded, x_encoded = best_model(
                inputs, 
                targets,
                mixup=False,
                mixup_alpha=cfg.train.mixup_alpha)
            feature = x_encoded
            
            for i in range(0, targets.shape[0]): # all samples in a batch
                
                if window_plot_counter >= 10:
                    continue
                
                target_behavior_list = [0, 1, 2, 3, 4, 5]
                if targets[i] in target_behavior_list:
                    print(f"inputs.shape: {inputs.shape}")
                    print(f"targets.shape: {target.shape}")
                    print(targets[i])
                    
                    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 3))
                    ax0 = utils.plot_window_ax(
                        ax0, 
                        sample[i], 
                        None, 
                        "Input window"
                    )
                    print(f"x_decoded.shape: {x_decoded.shape}")
                    ax1 = utils.plot_window_ax(
                        ax1,
                        x_decoded.transpose(1, 3).cpu()[i], 
                        None, 
                        "Decoded window"
                    )
                    plt.show()
                    plt.close()
                    window_plot_counter += 1
                    
                    input_tmp = sample[i]
                    distance_tmp_list = []
                    num_axes = 3
                    for axis in range(0, num_axes):
                        reconstruction_tmp = x_decoded.transpose(1, 3).cpu()[i]
                        a = np.reshape(np.array(input_tmp[0].transpose(1, 0)[axis]), (-1,1))
                        b = np.reshape(np.array(reconstruction_tmp[0].transpose(1, 0)[axis]), (-1,1))
                        distance_tmp, path = fastdtw(a, b, dist=euclidean)
                        distance_tmp_list.append(distance_tmp)
                    print(f"distance_tmp_list: {distance_tmp_list}")
            
            test_loss = criterion(test_output, targets)
            test_output = torch.nn.functional.softmax(test_output, dim=1)
            test_losses.append(test_loss.item())
            
            # prediction
            y_preds = np.argmax(
                test_output.cpu().numpy(),
                axis=1).flatten()
            test_preds = np.concatenate(
                (np.array(
                    test_preds, int), np.array(
                    y_preds, int)))
            
            # ground truth
            y_true = targets.cpu().numpy().flatten()
            test_gt = np.concatenate(
                (np.array(
                    test_gt, int), np.array(
                    y_true, int)))
            
            if features is None:
                features = feature
            else:
                features = torch.cat((features, feature), 0)
            
        test_loss_mean = np.mean(test_losses)
        test_accuracy = jaccard_score(test_gt, test_preds, average='macro')
        test_precision = precision_score(test_gt, test_preds, average='macro')
        test_recall = recall_score(test_gt, test_preds, average='macro')
        test_f1 = f1_score(test_gt, test_preds, average='macro')
        
        m1 = f"Test Loss: {test_loss_mean:.3f}"
        m2 = f"Acc: {test_accuracy:.3f}"
        m3 = f"Precision: {test_precision:.3f}"
        m4 = f"Recall: {test_recall:.3f}"
        m5 = f"F1: {test_f1:.3f}"
        print(f'{m1} {m2} {m3} {m4} {m5}')

    y_gt = test_gt
    y_pred = test_preds

    cm, df_cm, fig_cm = utils.plot_confusion_matrix(
        y_gt, 
        y_pred, 
        cfg, 
        figsize=(7, 6)
    )

    return y_gt, y_pred, features, cm, df_cm, fig_cm

def generate_condition_list(issue, ex, dataset, model_name):
    
    common_base_dir = "../../data/model-output"
    path0 = f"{issue}/{ex}"
    path1 = f"{dataset}"
    path2 = f"{model_name}"
    path3 = f"*"
    
    condition_target = os.path.join(common_base_dir, path0, path1, path2, path3)
    print(f"condition_target: {condition_target}")
    condition_list = sorted(glob.glob(condition_target))
    condition_list = [os.path.basename(s) for s in condition_list]
    
    return condition_list


def generate_test_config_target(issue, ex, dataset, model_name, condition, seed):
    
    common_base_dir = "../../data/model-output"
    path0 = f"{issue}/{ex}"
    path1 = f"{dataset}"
    path2 = f"{model_name}"
    path3 = f"{condition}"
    path4 = f"seed{seed}"
    path5 = "**/config.yaml"
    
    condition_target = os.path.join(common_base_dir, path0, path1, path2, path3)
    condition_list = sorted(glob.glob(condition_target))
    condition_list = [os.path.basename(s) for s in condition_list]
    
    config_target = os.path.join(common_base_dir, path0, path1, path2, path3, path4, path5)
    path_list = [path0, path1, path2, path3, path4, path5]
    
    return config_target, path_list



def generate_results_save_path(path_list, checkpoints_fname=None):
    
    results_save_roodir = "../../data/test-results"
    
    path0 = path_list[0]
    path1 = path_list[1]
    path2 = path_list[2]
    path3 = path_list[3]
    path4 = path_list[4]
    
    results_save_dir = f"{results_save_roodir}/{path0}/{path1}/{path2}"
    f_basename = f"{path3}_{path4}"
    if checkpoints_fname is not None:
        _checkpoints_fname = checkpoints_fname.replace(".pt", "")
        f_basename = f"{f_basename}_{_checkpoints_fname}"
    f_basename = f_basename.replace("-", "_")

    print(f"results_save_dir: {results_save_dir}")
    print(f"f_basename: {f_basename}")
    
    return results_save_dir, f_basename


def load_and_setup_test_config(config_path, TEST_CUDA_ID, checkpoints_fname=None):
    
    cfg = OmegaConf.load(config_path)
    print(f"test_animal_id: {cfg.dataset.test_animal_id}")
    print(cfg.dataset.labelled.animal_id_list.test)

    cfg.train.cuda = TEST_CUDA_ID
    DEVICE = torch.device('cuda:' + str(cfg.train.cuda) if torch.cuda.is_available() else 'cpu')
    print(f"DEVICE: {DEVICE}")
    
    cfg.train.data_augmentation = False
    cfg.dataset.da_type = None

    if cfg.debug == True:
        print("Note: You are loading a config file for debug !")
        cfg.debug = False
    # print(cfg.debug)
    
    if checkpoints_fname is None:
        if cfg.path.log.checkpoints.fname != "best_model_weights.pt":
            cfg.path.log.checkpoints.fname = "best_model_weights.pt"
    else:
        cfg.path.log.checkpoints.fname = checkpoints_fname
        print(f"cfg.path.log.checkpoints.fname is overwritten as: {cfg.path.log.checkpoints.fname}")

    return cfg, DEVICE


def setup_test_loader(cfg):
    from ..utils.utils import prep_dataloaders_for_supervised_learning
    (
        _, _, test_loader
    ) = prep_dataloaders_for_supervised_learning(cfg, test_only=True)
    print("len(test_loader): ", len(test_loader))
    print("len(test_loader.dataset): ", len(test_loader.dataset))
    if hasattr(test_loader, "load"): 
        test_loader.load()
        
    return test_loader


def setup_test_model(cfg, config_path, DEVICE):
    
    model = setup_model(cfg)
    # if i == 0:
    #     print(model)
    best_model = copy.deepcopy(model)
    best_model.to(DEVICE)
        
    best_model_path = os.path.join(
        os.path.dirname(config_path), 
        cfg.path.log.checkpoints.dir, 
        cfg.path.log.checkpoints.fname
    )
    print(f"best_model_path: {best_model_path}")
    
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    
    # print(checkpoint.keys())
    
    if "model_state_dict" in checkpoint:
        best_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        best_model.load_state_dict(checkpoint)
        
    return best_model


def test_optimizer_criterion_setup(cfg, model):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.lr, 
        weight_decay=cfg.train.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    
    return optimizer, criterion


def test_setup(cfg, config_path, DEVICE):
    # test_loader
    test_loader = setup_test_loader(cfg)
    
    # Load best model weights for this test animal
    best_model = setup_test_model(cfg, config_path, DEVICE) 
    
    # initialize the optimizer and loss
    optimizer, criterion = test_optimizer_criterion_setup(cfg, best_model)
    
    return test_loader, best_model, optimizer, criterion


def generate_test_results_set(
    y_gt_all, 
    y_pred_all, 
    test_animal_id_all,
    cfg
):
    # F1-score DataFrame
    df_test_scores_all = utils.generate_test_score_df(
        cfg,
        y_gt_all, 
        y_pred_all,  
        "ALL"
    )
    # display(df_test_scores_all)
    
    # Ground Truth and Prediction
    data_dict = {
        'y_gt': y_gt_all,
        'y_pred': y_pred_all, 
        'test_id': test_animal_id_all,
    }
    df_gt_pred_all = pd.DataFrame(data=data_dict)
    
    # Confusion Matrix
    cm, df_cm, fig_cm = utils.plot_confusion_matrix(
        y_gt_all, 
        y_pred_all, 
        cfg, 
        figsize=(7, 6)
    )
    
    return df_test_scores_all, df_gt_pred_all, fig_cm

    
def save_test_results_set(
    results_save_dir,
    f_basename,
    df_test_score_all, 
    df_gt_pred_all
):
    
    # F1-score
    dir_path = f"{results_save_dir}/test-score"
    os.makedirs(dir_path, exist_ok=True)
    path = f"{dir_path}/test_score_{f_basename}.csv"
    df_test_score_all.to_csv(path, index=False)
    
    # Ground Truth and Prediction
    dir_path = f"{results_save_dir}/y-gt-y-pred"
    os.makedirs(dir_path, exist_ok=True)
    path = f"{dir_path}/y_gt_y_pred_{f_basename}.csv"
    df_gt_pred_all.to_csv(path, index=False)