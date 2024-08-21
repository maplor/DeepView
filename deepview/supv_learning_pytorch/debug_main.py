# This Python file uses the following encoding: utf-8
import torch
import hydra
from omegaconf import DictConfig
from utils.utils import setup_train_val_test_animal_id_list, setup_dataloaders_supervised_learning
from deepview.supv_learning_pytorch.core.supv_trainer import setup_model
from deepview.supv_learning_pytorch.core.trainer import train
from pathlib import Path

all_animal_id_list = ["OM1802"
    , "OM1803"
    , "OM1804"
    , "OM1805"
    , "OM1806"
    , "OM1807"
    , "OM1808"
    , "OM1809"
    , "OM1810"
    , "OM1811"
    , "OM1901"
    , "OM2001"
    , "OM2002"
    , "OM2003"
                      # - "OM2004", # sensor broken?
    , "OM2005"
    , "OM2006"
    , "OM2101"
    , "OM2102"
    , "OM2103"
    , "OM2201"
    , "OM2202"
    , "OM2203"
    , "OM2204"
    , "OM2205"
    , "OM2206"
    , "OM2207"
    , "OM2208"
    , "OM2209"
    , "OM2210"
    , "OM2211"
    , "OM2212"
    , "OM2213"
    , "OM2214"]  # om.yaml of ostuka code


@hydra.main(version_base=None,
            config_path="./config",
            config_name="config_dl.yaml"
            )
def main(cfg: DictConfig):
    # DEVICE = 'cpu'
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device using %s' % DEVICE)
    # initialization of train, val, and test animal id list
    test_animal_id_list = ["OM2214"]
    (
        train_animal_id_list,
        val_animal_id_list,
        test_animal_id_list
    ) = setup_train_val_test_animal_id_list(
        all_animal_id_list,
        test_animal_id_list
    )

    # load datasets and prepare dataloader
    (
        train_loader_balanced,
        val_loader,
        test_loader
    ) = setup_dataloaders_supervised_learning(
        cfg,
        train_animal_id_list,
        val_animal_id_list,
        test_animal_id_list,
        train=True,
        train_balanced=True
    )

    # setup model
    model = setup_model(cfg)
    model.to(DEVICE)  # send model to GPU
    print(f'model:\n {model}')

    # initialize the optimizer and loss
    if cfg.train.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay
        )
    elif cfg.train.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            # lr=cfg.train.lr,
            lr=0.01,
            momentum=0.9,
            weight_decay=0
        )

    # loss function
    criterion = torch.nn.CrossEntropyLoss()

    # run training, and validate result using validation data
    best_model, df_log = train(
        model,
        optimizer,
        criterion,
        train_loader_balanced,
        val_loader,
        test_loader,
        DEVICE,
        cfg
    )

    # save the training log
    path = Path(cfg.path.log.rootdir, cfg.path.log.logfile.csv)
    df_log.to_csv(path, index=False)

    return


if __name__ == '__main__':
    main()
