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
import sklearn.metrics as metrics

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
    max_iter,
    optimizer,
    criterion,
    train_loader,
    test_loader,
    DEVICE,
    sup_path
):

    best_val_f1 = 0
    best_model = None

    # scheduler = setup_scheduler(cfg, optimizer)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=max_iter,
        lr_min=1e-6,
        warmup_t=0,
        warmup_lr_init=0,
        warmup_prefix=False)

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

    early_stopping = EarlyStopping(patience=7, path=sup_path)

    # define your training loop; iterates over the number of epochs
    for epoch in tqdm(range(max_iter)):

        # ---------------------------------------------------------------------------
        # Training
        # ---------------------------------------------------------------------------
        model.train()
        train_losses = []
        total, correct = 0, 0
        output_list, target_list = [], []
        for idx, (sample, target) in enumerate(train_loader):
            inputs = sample.to(torch.float)
            # inputs = torch.transpose(sample.unsqueeze(3), 2, 1)
            output = model(inputs)
            loss = criterion(output.reshape(-1, output.shape[-1]),
                             target.reshape(-1).long())

            output_list.append(output)
            target_list.append(target[:, 0])

            # zero accumulated gradients
            optimizer.zero_grad()
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

            # calculate accuracy
            target = target.reshape(-1).long()
            output = output.reshape(-1, output.shape[-1])
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (target == predicted).sum()

        scheduler.step(epoch)
        accuracy = float(correct) * 100.0 / total

        output_concat = torch.concatenate(output_list)
        target_concat = torch.concat(target_list)
        top_p, top_class = output_concat.topk(1, dim=1)
        f1score = metrics.f1_score(target_concat.long().cpu(),
                                   top_class.cpu(),
                                   average='weighted') * 100.0

        print(f'Test Loss     : {np.mean(train_losses):.4f} \t| \t accuracy     : {accuracy:2.4f}| \t weighted macro f1     : {f1score:2.4f}\n')

        # save model weights
        model_save_path = Path(
            sup_path,
            f"model_weights_epoch_{epoch:0=3}.pt")
        # path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model_state_dict': model.state_dict()}, model_save_path)


        # ---------------------------------------------------------------------------
        # Test result
        # ---------------------------------------------------------------------------

        model.eval()
        total_loss = 0
        total = 0
        correct = 0
        prds = None
        trgs = None
        output_list, target_list = [], []
        with torch.no_grad():
            n_batches = 0
            for idx, (sample, target) in enumerate(test_loader):
                n_batches += 1

                sample = sample.to(torch.float)
                # sample = torch.transpose(sample.unsqueeze(3), 2, 1)
                out = model(sample)

                output_list.append(out)
                target_list.append(target)
                # reshape output
                target = target.reshape(-1).long()
                output = out.reshape(-1, out.shape[-1])
                loss = criterion(output, target)
                # loss = criterion(out, target[:, 0, :].reshape(-1).long())
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum()

                if prds is None:
                    prds = predicted
                    trgs = target
                    # feats = features[:, :]
                else:
                    prds = torch.cat((prds, predicted))
                    trgs = torch.cat((trgs, target))
                    # feats = torch.cat((feats, features), 0)

            acc_test = float(correct) * 100.0 / total

            output_concat = torch.concatenate(output_list)
            output_concat = output_concat.reshape(-1, output_concat.shape[-1])
            target_concat = torch.concat(target_list)
            target_concat = target_concat.reshape(-1)
            # target_concat = target_concat.reshape(-1, target_concat.shape[-1])
            top_p, top_class = output_concat.topk(1, dim=1)
            f1score = metrics.f1_score(target_concat.long().cpu(),
                                       top_class.cpu(),
                                       average='weighted') * 100.0

            print(
                f'Test Loss     : {total_loss / n_batches:.4f}\t | \tTest Accuracy     : {acc_test:2.4f} \t| \t macro f1     : {f1score:2.4f}\n')

        early_stopping_coutner_list.append(early_stopping.counter)
        patience_list.append(early_stopping.patience)


    return best_model




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